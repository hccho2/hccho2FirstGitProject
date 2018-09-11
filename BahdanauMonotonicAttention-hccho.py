# -*- coding: utf-8 -*-
# https://github.com/craffel/mad
# BahdanauMonotonicAttention
"""
BahdanauAttention     -->    _bahdanau_score   --> wrapped_probability_fn = probability_fn = tf.nn.softmax

BahdanauMonotonicAttention   -->     _bahdanau_score  --> wrapped_probability_fn = _monotonic_probability_fn('hard','parallel','recursiive')

"""


"""Example soft monotonic alignment decoder implementation.
This file contains an example TensorFlow implementation of the approach
described in ``Online and Linear-Time Attention by Enforcing Monotonic
Alignments''.  The function monotonic_attention covers the algorithms in the
paper and should be general-purpose.  monotonic_alignment_decoder can be used
directly in place of tf.nn.seq2seq.attention_decoder.  This implementation
attempts to deviate as little as possible from tf.nn.seq2seq.attention_decoder,
in order to facilitate comparison between the two decoders.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten
from tensorflow.python.util.nest import is_sequence


def safe_cumprod(x, **kwargs):
    """Computes cumprod in logspace using cumsum to avoid underflow."""
    return tf.exp(tf.cumsum(tf.log(tf.clip_by_value(x, 1e-10, 1)), **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
    # p_choose_i: (batch_size, encoder_seq_length), 각각의 원손는 sigmoid를 취한 값이기 때문에 0~1의 값을 가진다.
    
    
    """Compute monotonic attention distribution from choosing probabilities.
    Monotonic attention implies that the input sequence is processed in an
    explicitly left-to-right manner when generating the output sequence.  In
    addition, once an input sequence element is attended to at a given output
    timestep, elements occurring before it cannot be attended to at subsequent
    output timesteps.  This function generates attention distributions according
    to these assumptions.  For more information, see ``Online and Linear-Time
    Attention by Enforcing Monotonic Alignments''.
    Args:
      p_choose_i: Probability of choosing input sequence/memory element i.  Should
        be of shape (batch_size, input_sequence_length), and should all be in the
        range [0, 1].
      previous_attention: The attention distribution from the previous output
        timestep.  Should be of shape (batch_size, input_sequence_length).  For
        the first output timestep, preevious_attention[n] should be [1, 0, 0, ...,
        0] for all n in [0, ... batch_size - 1].
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.
          * 'recursive' uses tf.scan to recursively compute the distribution.
            This is slowest but is exact, general, and does not suffer from
            numerical instabilities.
          * 'parallel' uses parallelized cumulative-sum and cumulative-product
            operations to compute a closed-form solution to the recurrence
            relation defining the attention distribution.  This makes it more
            efficient than 'recursive', but it requires numerical checks which
            make the distribution non-exact.  This can be a problem in particular
            when input_sequence_length is long and/or p_choose_i has entries very
            close to 0 or 1.
          * 'hard' requires that the probabilities in p_choose_i are all either 0
            or 1, and subsequently uses a more efficient and exact solution.
    Returns:
      A tensor of shape (batch_size, input_sequence_length) representing the
      attention distributions for each sequence in the batch.
    Raises:
      ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
    """
    if mode == "recursive":
        batch_size = tf.shape(p_choose_i)[0]
        # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
        shifted_1mp_choose_i = tf.concat( [tf.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
        # Compute attention distribution recursively as
        # q[i] = (1 - p_choose_i[i])*q[i - 1] + previous_attention[i]
        # attention[i] = p_choose_i[i]*q[i]
        attention = p_choose_i*tf.transpose(tf.scan(
            # Need to use reshape to remind TF of the shape between loop iterations
            lambda x, yz: tf.reshape(yz[0]*x + yz[1], (batch_size,)),
            # Loop variables yz[0] and yz[1]
            [tf.transpose(shifted_1mp_choose_i), tf.transpose(previous_attention)],
            # Initial value of x is just zeros
            tf.zeros((batch_size,))))
    elif mode == "parallel":
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
        # Compute recurrence relation solution
        attention = p_choose_i*cumprod_1mp_choose_i*tf.cumsum(
            previous_attention /
            # Clip cumprod_1mp to avoid divide-by-zero
            tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
    elif mode == "hard":
        # Remove any probabilities before the index chosen last time step
        p_choose_i *= tf.cumsum(previous_attention, axis=1)
        # Now, use exclusive cumprod to remove probabilities after the first
        # chosen index, like so:
        # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
        # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
        # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
        attention = p_choose_i*tf.cumprod(1 - p_choose_i, axis=1, exclusive=True)
    else:
        raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
    return attention


def monotonic_alignment_decoder(decoder_inputs, initial_state, attention_states, cell, output_size=None,num_heads=1, loop_function=None, dtype=None, scope=None,
                                initial_state_attention=False, sigmoid_noise_std_dev=0.,initial_energy_bias=0., initial_energy_gain=None, hard_sigmoid=False):
    """RNN decoder with monotonic alignment for the sequence-to-sequence model.
    In this context "monotonic alignment" means that, during decoding, the RNN can
    look up information in the additional tensor attention_states, and it does
    this by focusing on a few entries from the tensor.  The attention mechanism
    used here is such that the first element in attention_states which has a high
    coefficient is likely to be chosen, and the subsequent attentions will only
    look at items from attention_state after the one chosen at a previous step.
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.
      sigmoid_noise_std_dev: Standard deviation of pre-sigmoid additive noise.  To
        ensure that the model produces hard alignments, this should be set larger
        than 0.
      initial_energy_bias: Initial value for bias scalar in energy computation.
        Setting this value negative (e.g. -4) ensures that the initial attention
        is spread out across the encoder states at the beginning of training,
        which can facilitate convergence.
      initial_energy_gain: Initial gain term scalar in energy computation.
        Setting this value too large may result in the attention sigmoids becoming
        saturated and losing the learning signal.  By default, it is set to
        1/sqrt(attn_size).
      hard_sigmoid: Whether to use a hard sigmoid when computing attention
        probabilities.  This should be set to False during training, and True
        during testing to simulate linear time/online computation.
    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(
        scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = tf.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape( attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v, b, r, g = [], [], [], []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            init = tf.random_normal_initializer(stddev=1./attention_vec_size)
            v.append(tf.get_variable( "AttnV_%d" % a, [attention_vec_size], initializer=init))
            r.append(tf.get_variable( "AttnR_%d" % a, [],initializer=tf.constant_initializer(initial_energy_bias)))
            b.append(tf.get_variable("AttnB_%d" % a, [attention_vec_size], initializer=tf.zeros_initializer()))
            if initial_energy_gain is None:
                initial_energy_gain = np.sqrt(1./attention_vec_size)
            g.append(tf.get_variable("AttnG_%d" % a, [], initializer=tf.constant_initializer(initial_energy_gain)))

        state = initial_state

        def attention(query, previous_attentions):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            alignments = []
            if is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = tf.concat(1, query_list)
            for a in range(num_heads):
                with tf.variable_scope("Attention_%d" % a) as scope:
                    previous_attention = previous_attentions[a]
                    y = tf.contrib.layers.linear(query, attention_vec_size, scope=scope, biases_initializer=None)
                    y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                    normed_v = g[a]*v[a]/tf.norm(v[a])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(normed_v*tf.tanh(hidden_features[a] + y + b[a]), [2, 3])
                    s += r[a]
                    if hard_sigmoid:
                        # At test time (i.e. not computing gradients), use hard sigmoid
                        # Note that s are pre-sigmoid logits, so thresholding around 0
                        # is equivalent to thresholding the probability around 0.5
                        a = tf.cast(tf.greater(s, 0.), s.dtype)
                        attention = monotonic_attention(a, previous_attention, "hard")
                    else:
                        a = tf.nn.sigmoid(s + sigmoid_noise_std_dev*tf.random_normal(tf.shape(s)))
                        attention = monotonic_attention(a, previous_attention, "recursive")
                    alignments.append(attention)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(tf.reshape(attention, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(tf.reshape(d, [-1, attn_size]))
            return ds, alignments

        outputs = []
        prev = None
        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype) for _ in range(num_heads)]
        # Initialize the first alignment to dirac distributions which will cause
        # the attention to compute the right thing without special casing
        all_alignments = [ [tf.one_hot(tf.zeros((batch_size,), tf.int32), attn_length, dtype=dtype) for _ in range(num_heads)]]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns, alignments = attention(initial_state, all_alignments[-1])
            all_alignments.append(alignments)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            input_size = input_size.value
            x = tf.contrib.layers.linear(tf.concat([inp] + attns, 1), input_size,
                                         reuse=i > 0, scope=tf.get_variable_scope())
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    attns, alignments = attention(state, all_alignments[-1])
            else:
                attns, alignments = attention(state, all_alignments[-1])
            all_alignments.append(alignments)

            with tf.variable_scope("AttnOutputProjection"):
                output = tf.contrib.layers.linear(
                    tf.concat([cell_output] + attns, 1), output_size, reuse=i > 0,
                    scope=tf.get_variable_scope())
            if loop_function is not None:
                prev = output
            outputs.append(output)
    return outputs, state
def test_monotonic_alignment_decoder():
    """Test for utils.learning_to_emit_decoder."""
    with tf.Session() as sess:
        with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
            cell = tf.contrib.rnn.GRUCell(7)
            
            # seq_length, batch_size, encoder_dim : time major
            inp = [tf.constant(0.5, shape=[3, 4])] * 2  # input.    inp = [tf.constant([[0.5, 0.5,0.5, 0.5],[0.5, 0.5,0.5, 0.5],[0.5, 0.5,0.5, 0.5]]),tf.constant([[0.5, 0.5,0.5, 0.5],[0.5, 0.5,0.5, 0.5],[0.5, 0.5,0.5, 0.5]])]
            
            enc_outputs, enc_state = tf.contrib.rnn.static_rnn( cell, inp, dtype=tf.float32)
            attn_states = tf.concat([tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)  # ==> time major에서 ==> (batch_size, seq_length, hidden_dim)
            
            
            dec_inp = [tf.constant(0.4, shape=[3, 4])] * 5
            dec_cell = tf.contrib.rnn.GRUCell(7)
            
            dec, mem = monotonic_alignment_decoder(dec_inp, initial_state=enc_state, attention_states=attn_states, cell = dec_cell, output_size=4)
            sess.run([tf.global_variables_initializer()])
            res = sess.run(dec)
            assert len(res) == 5
            assert res[0].shape == (3, 4)
 
            res = sess.run([mem])
            assert res[0].shape == (3, 7)


def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
def hard_attention():
    A = np.array([[-1.3456318 , -1.1016366 ,  0.53299403,  0.42781317],
           [-2.0774524 ,  1.4898142 ,  0.5875847 , -1.6679052 ],
           [ 2.042389  ,  0.8909064 ,  0.03359929, -0.83841914]]).astype(np.float32)
    A = sigmoid(A)
    tf.reset_default_graph()
    previous_attention = tf.one_hot(tf.zeros((3,),tf.int32),4)
    
    sess = tf.Session()
    for i in range(3):
        #p_choose_i = tf.convert_to_tensor(sigmoid(np.random.normal(0,1,[3,4]).astype(np.float32))) 
        p_choose_i = tf.convert_to_tensor(np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0]]).astype(np.float32)) 
        p_choose_i *= previous_attention
        temp = tf.cumprod(1 - p_choose_i, axis=1, exclusive=True)
        attention = p_choose_i*temp
        x,y,z =sess.run([p_choose_i,temp,attention])
        print("step ", i )
        print("p_choose_i", x,)
        print("temp", y,)
        print("attention", z,)
        previous_attention = attention
    
def parallel_attention():
    previous_attention = tf.one_hot(tf.zeros((3,),tf.int32),4)
    sess = tf.Session()
    history = []
    for i in range(10):
        p_choose_i = tf.convert_to_tensor(sigmoid(np.random.normal(0,1,[3,4]).astype(np.float32))) 
        cumprod_1mp_choose_i = tf.contrib.seq2seq.safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
        attention = p_choose_i*cumprod_1mp_choose_i*tf.cumsum(previous_attention / tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
        attention_ = sess.run(attention)
        print(i, attention_)
        history.append(attention_)
        previous_attention = attention
    
    history = np.array(history).transpose(1,0,2)
    print(np.sum(history[0],axis=1))

def recursive_attention():
    previous_attention = tf.one_hot(tf.zeros((3,),tf.int32),4)
    sess = tf.Session()
    history = []
    for i in range(10):    
        p_choose_i = tf.convert_to_tensor(sigmoid(np.random.normal(0,1,[3,4]).astype(np.float32))) 
        batch_size = p_choose_i.shape[0].value or tf.shape(p_choose_i)[0]
        shifted_1mp_choose_i = tf.concat([tf.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
        attention = p_choose_i*tf.transpose(tf.scan(lambda x, yz: tf.reshape(yz[0]*x + yz[1], (batch_size,)),
                                                                       [tf.transpose(shifted_1mp_choose_i),tf.transpose(previous_attention)],
                                                                       tf.zeros((batch_size,))))
        attention_ = sess.run(attention)
        print(i, attention_)
        history.append(attention_)
        previous_attention = attention        
    history = np.array(history).transpose(1,0,2)   
    print(np.sum(history[0],axis=1))
    
def parallel_attention_numpy():
    batch_size=3
    p_choose_i = np.array([[[-0.89901773,  0.14183508, -0.43932167,  0.99357405,0.01001145,  1.34721523, -0.38266134, -0.31384029,0.24819669, -1.05571939],
    [-0.34286925,  1.95730185, -0.36198071,  0.86635974, -0.73570819, -1.45275514, -0.62655579,  0.46074289,  0.22833848, -1.54738561],
    [-1.99476573, -0.36442733,  2.02440133,  1.14772577,  0.01852836, -1.40128036, -0.54021285, -0.33474558, -0.34070014, -1.11878183]],
    [[-2.05640416,  0.72517592,  0.2531164 ,  0.66428936, -1.02601423, -0.89712032, -0.01042617, -0.4191656 ,  0.6548876 , -1.29142762],
    [ 0.30953507, -0.16832032, -1.18473549,  0.07200521,  0.24944816,  1.77674227,  0.65923626, -0.1727309 ,  0.41155359, -0.59379435],
    [ 1.44942389,  0.44197312, -0.51229049,  2.49422301, -0.91184975,  0.13487913,  0.29189776,  0.02945988,  1.57689731,  1.25581209]],
    [[ 1.46187947, -0.68268157,  1.38675132,  0.84723606, -1.30993363,  0.14201748, -1.847896  ,  0.54726278,  1.17751669,  0.68767254],
    [-0.12334747, -0.67706039,  0.81253611,  1.18548635, -0.22540487,  0.17888472,  0.1285538 ,  0.46056531,  0.02206509, -0.39119461],
    [-0.60105562,  1.40515868, -0.43853388, -0.87129582, -0.00505184, -0.16821598,  0.59898874,  1.23691615,  2.12198066,  0.22141299]],
    [[ 0.82872332, -0.04661657,  0.64132042, -0.4977081 , -0.43390763, -0.59754169, -0.10574198, -1.1444557 , -1.19829561, -1.2535122 ],
    [-0.34574755, -0.91017162, -1.81622602, -0.52265621, -0.10214899, -0.37669624, -0.55936436, -0.03374816,  0.45923689, -0.03319437],
    [-1.31260342, -0.19351793,  1.03563597,  0.41327658,  1.09949367,  1.18045847,  1.23929454, -1.15535779, -0.31079625, -0.56205703]],
    [[-0.47976513,  0.67557319, -0.66927814,  0.47478865, -0.18589161,  0.44397139,  2.03388436, -1.35983802,  0.81342536, -0.74512464],
    [ 0.7204559 , -1.46816947, -1.37991966,  0.30536252, -2.28578877,  0.18252803, -0.30200202, -0.19353009, -1.08347742, -0.42056578],
    [ 1.00618286,  0.08953775,  0.00282596,  0.13088284,  0.96673115, -0.52491113, -0.90667701, -0.41551252,  0.10978281, -1.33076673]],
    [[-1.23876635,  0.56415582, -1.1629744 ,  0.04056541,  0.2215322 , -0.18852159,  0.02717084,  0.77659747,  0.01619746,  1.03257522],
    [ 0.40994443,  0.78494598, -2.06820927,  0.9496637 , -0.243087  , -0.13149841, -2.50164756, -0.28846233, -1.40576018, -0.93940815],
    [ 0.74577234, -1.73604482, -0.80904104, -0.29026098, -1.19616067,  1.07409706, -1.83301568, -0.64789491,  0.51007962,  0.06333512]],
    [[-0.73922986, -1.17759878,  1.33320424, -1.76674376,  0.8367948 ,  0.55242259,  0.30124086,  0.51145035,  1.15348453,  0.16924186],
    [ 1.04797747,  0.3110594 ,  2.21662223,  0.08206117, -2.79636798,  0.98739423, -1.75438147, -0.60314009,  1.04436402,  0.70232486],
    [-1.07322745, -2.00554488, -0.17485827,  0.49806884, -0.2114042 , -0.81874392,  1.49001887,  1.42146966,  0.25549828,  0.46842105]],
    [[ 0.20906318, -1.13665509,  1.40834536, -1.01611393, -0.10968219,  0.22068963, -1.32153574,  0.94595086, -1.55675788, -2.13587673],
    [-0.28589113, -1.01645871, -1.70474014, -0.55955685,  0.44933885, -0.97751688, -1.31903962,  0.10947197, -1.14452173, -0.3321419 ],
    [ 1.56814131, -0.5881521 , -0.1832418 , -2.12663002, -2.3773412 ,  1.85963246, -0.18693106, -0.05903214,  1.41461495,  0.97961292]],
    [[ 0.42443063,  0.2390123 ,  0.63539649, -0.38132858,  0.72211599,  0.45663955, -0.614955  , -0.18606778, -0.05001624, -1.89753785],
    [ 0.34574263, -0.48885766, -0.03101046, -2.12924774,  0.3800159 ,  0.96259782,  0.26243147, -1.41574141, -0.74604093, -1.78796894],
    [-1.02706114, -1.10307845,  0.51202942,  0.54355624, -2.31511281, -2.15539538,  0.60562796,  1.13959043,  1.16698276, -1.03740561]],
    [[-0.12682132,  2.09207679,  0.03802798, -1.55206444,  0.73403656,  0.25959113, -0.35208918,  0.14817594,  0.60981187, -0.77832624],
    [ 1.63225482,  0.68377941, -0.69771809, -0.24278797,  0.14772713,  2.36880138, -1.17846297, -0.57687073, -1.59992605,  1.56378731],
    [-0.48335045, -0.51257482, -0.95629677,  0.93302412,  0.55679952, -1.4379083 , -1.56107105, -1.10102182,  0.25985005,  0.75902881]]])
    p_choose_i = sigmoid(p_choose_i)
    tmp = np.zeros(batch_size)
    previous_attention = change_one_hot_label(tmp,10)
    history = []
    
    for j in range(10): 
        tmp=np.ones(batch_size).reshape(batch_size,1)
        cumprod_1mp_choose_i = np.cumprod(np.concatenate((tmp,1-p_choose_i[j][:,:-1]),axis=1), axis=1)
        attention = p_choose_i[j]*cumprod_1mp_choose_i*np.cumsum(previous_attention / np.clip(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
        
        #attention = softmax(attention)  <--- hccho idea
        attention = attention/np.sum(attention,axis=1,keepdims=True)   <--- hccho idea
        
        #print(j, attention)
        history.append(attention)
        previous_attention = attention
    
    history = np.array(history).transpose(1,0,2)
    print(history)        
    
def change_one_hot_label(X,n_dim=10):
    X = X.astype(np.int32)
    T = np.zeros((X.size, n_dim))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))  
if __name__ == '__main__':
    #test_monotonic_alignment_decoder()
    #parallel_attention()
    #recursive_attention()
    parallel_attention_numpy()
    
    