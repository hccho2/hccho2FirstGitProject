# coding: utf-8
# user defined Wrapper
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseMonotonicAttentionMechanism
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _monotonic_probability_fn,_bahdanau_score
from tensorflow.python.layers.core import Dense
import functools


tf.reset_default_graph()
SOS_token = 0
EOS_token = 4
class MyRnnWrapper(RNNCell):
    # property(output_size, state_size) 2개와 call을 정의하면 된다.
    def __init__(self,name,state_dim):
        super(MyRnnWrapper, self).__init__(name=name)
        self.sate_size = state_dim

    def build(self, inputs_shape):
        # 필요한 trainable variable이 있으면 여기서 생성하고, self.built = True로 설정하면 된다.
        # 이곳을 잘 정의하면 BasicRNNCell, GRUCell, BasicLSTMCell같은 것을 만들 수 있다.
        # 여기서 선언한 Variable들을 아래의 call에서 input, state와 엮어서 필요한 계산을 하면된다.
        # BasicRNNCell, GRUCell 소스 코드를 보면 그렇데 되어 있다.
        
        # helper에서 필요한 정보를 받아서 inputs_shape를 받아오는 것이다.
        self.inputs_shape = inputs_shape.as_list()
        self.built = True

    @property
    def output_size(self):
        return 4  # input_dim *2

    @property
    def state_size(self):
        return self.sate_size  

    # 다음의 call은 내부적으로 __call__과 연결되어 있다.
    def call(self, inputs, state):
        # 이 call 함수를 통해 cell과 cell이 연결된다.
        # input에 필요에 따라, 원하는 작업을 하면 된다.
        xxx = tf.get_variable("zxx",[1],dtype=tf.float32)
        cell_output = tf.concat([inputs,inputs],axis=-1)
        next_state = state + 0.11
        return cell_output, next_state 


    # zero_state는 반드시 재정의해야 하는 것은 아니다. 필요에 따라...
    def zero_state(self,batch_size,dtype=tf.float32):
        return tf.ones([batch_size,self.sate_size],dtype)  # test 목적으로 1을 넣어 봄


class MyRnnWrapper2(RNNCell):
    # property(output_size, state_size) 2개와 call을 정의하면 된다.
    # 일반적인 User defined Wrapper는 cell(eg GRUCell)을 입력받아 필요한 작업을 추가하는 방식이다.
    # 입력받지 않으면, init에서 필요한 cell을 만들면 된다.
    def __init__(self,cell,name,hidden_dim):
        super(MyRnnWrapper2, self).__init__(name=name)
        self.sate_size = hidden_dim
        self.cell = cell # 
    @property
    def output_size(self):
        # 아래 call에서 intput과 state를 concat하는 방식이기 때문에, output size는 input dim + state size가 된다. 그런데, input dim을 어떻게 알수 있나?
        # 이런 경우는 input dim을 알아야 하는 특수한 경우이기 때문에, init에서 input dim을 입력받아야 한다.
        return 2 + self.sate_size  

    @property
    def state_size(self):
        return self.sate_size  

    # 다음의 call은 내부적으로 __call__과 연결되어 있다.
    def call(self, inputs, state):
        # 이 call 함수를 통해 cell과 cell이 연결된다.
        # input에 필요에 따라, 원하는 작업을 하면 된다.
        fc_outputs = tf.layers.dense(inputs,units=10,name='myFC')
        cell_out, cell_state = self.cell(fc_outputs,state)
        cell_output = tf.concat([inputs,cell_out],axis=-1)
        next_state = state + 0.11
        return cell_output, next_state 


    # zero_state는 반드시 재정의해야 하는 것은 아니다. 필요에 따라...
    def zero_state(self,batch_size,dtype=tf.float32):
        return tf.ones([batch_size,self.sate_size],dtype)  # test 목적으로 1을 넣어 봄
     
    
class MyRnnHelper(Helper):
    # property(batch_size,sample_ids_dtype,sample_ids_shape)이 정의되어야 하고, initialize,sample,next_inputs이 정의되어야 한다.
    def __init__(self,embedding,batch_size,output_dim):
        self._embedding = embedding
        self._batch_size = batch_size
        self._output_dim = output_dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    def next_inputs(self, time, outputs, state,sample_ids, name=None):   # time+1을 위한 input을 만든다., outputs,state,sample_ids는 time step에서의 결과이다.
        # 넘어오는 sample_ids는 sample 함수에어 계산된어 넘어온 값이다.   <----- 이런 계산은 BasicDecoder의 'step' 함수에서 이루어 진다.
        # next input을 계산하기 위해서 sample_ids를 이용하거나, outpus를 이용하거나 선택하면 된다.
        
        
        finished = (time + 1 >= 7)    # finished = (time + 1 >= [7,8,9])
        #next_inputs = outputs[:, -self._output_dim:]*2
        next_inputs = tf.nn.embedding_lookup(self._embedding,sample_ids)
        #next_inputs = tf.zeros_like(next_inputs)
        return (finished, next_inputs, state)  #finished==True이면 next_inputs,state는 의미가 없다.

    def initialize(self, name=None):
        # 시작하는 input을 정의한다.
        # return finished, first_inputs. finished는 시작이니까, 무조건 False
        # first_inputs는 예를 위해서, SOS_token으로 만들어 보았다.
        return (tf.tile([False], [self._batch_size]), tf.nn.embedding_lookup(self._embedding,tf.tile([SOS_token], [self._batch_size])))  

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them



class BahdanauMonotonicAttention_hccho(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Bahadanau-style energy function.

    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the memory
    it can't attend to any prior points at subsequence output timesteps.  It
    achieves this by using the _monotonic_probability_fn instead of softmax to
    construct its attention distributions.  Since the attention scores are passed
    through a sigmoid, a learnable scalar bias parameter is applied after the
    score function and before the sigmoid.  Otherwise, it is equivalent to
    BahdanauAttention.  This approach is proposed in

    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 score_mask_value=None,
                 sigmoid_noise=0.,
                 sigmoid_noise_seed=None,
                 score_bias_init=0.,
                 mode="parallel",
                 dtype=None,
                 name="BahdanauMonotonicAttentionHccho"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
            for `_monotonic_probability_fn` for more information.
          sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
          score_bias_init: Initial value for score bias scalar.  It's recommended to
            initialize this to a negative value when the length of the memory is
            large.
          mode: How to compute the attention distribution.  Must be one of
            'recursive', 'parallel', or 'hard'.  See the docstring for
            `tf.contrib.seq2seq.monotonic_attention` for more information.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
        """
        # Set up the monotonic probability fn with supplied parameters
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention_hccho, self).__init__(
            query_layer=Dense(num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=Dense(num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(
            None, "bahdanau_monotonic_hccho_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            
            # processed_query: (N,num_units)  ==> self._keys와 더하기 위해서  _bahdanau_score 내부에서 (N,1,num_units) 으로 변환.
            # self._keys: (N, encoder_dim, num_units)
            score = _bahdanau_score(processed_query, self._keys, self._normalize)     # keys 가 memory임
            score_bias = tf.get_variable("attention_score_bias", dtype=processed_query.dtype, initializer=self._score_bias_init)

            #alignments_bias = tf.get_variable("alignments_bias", shape = state.get_shape()[-1],dtype=processed_query.dtype, initializer=tf.zeros_initializer())  # hccho
            alignments_bias = tf.get_variable("alignments_bias", shape = (1),dtype=processed_query.dtype, initializer=tf.zeros_initializer())  # hccho

            score += score_bias
        alignments = self._probability_fn(score, state)   #BahdanauAttention에서 _probability_fn = softmax

        next_state = alignments   # 다음 alignment 계산에 사용할 state 값
        # hccho. alignment가 attention 계산에 직접 사용된다.
        alignments = tf.nn.relu(alignments+alignments_bias)
        alignments = alignments/(tf.reduce_sum(alignments,axis=-1,keepdims=True) + 1.0e-12 )  # hccho 수정


        return alignments, next_state



def wapper_test():
    vocab_size = 5
    x_data = np.array([[SOS_token, 3, 3, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[1,2,0,3,2,EOS_token],[3,2,3,3,1,EOS_token],[3,1,1,2,0,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =4
    seq_length = x_data.shape[1]
    embedding_dim = 2
    
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = False
    with tf.variable_scope('test') as scope:
        # Make rnn
        #cell = MyRnnWrapper("xxx",hidden_dim)
        cell = MyRnnWrapper2(tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim),"xxx",hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        
        #aaa = cell(inputs,initial_state)
        
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
        else:
            #helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
            helper = MyRnnHelper(embedding,batch_size,embedding_dim)
    
        output_layer = Dense(output_dim, name='output_projection')
        
        #BasicDecoder는 clas
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다.
        # dynamic_decode는 class가 아니고 function임
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
    
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,"\n argmax --> ",o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
    

def wapper_attention_test():
    vocab_size = 5
    x_data = np.array([[SOS_token, 3, 3, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[1,2,0,3,2,EOS_token],[3,2,3,3,1,EOS_token],[3,1,1,2,0,EOS_token]],dtype=np.int32)

    
    
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =4
    seq_length = x_data.shape[1]
    embedding_dim = 2
    
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = False
    alignment_history_flag = False
    with tf.variable_scope('test') as scope:
        # Make rnn
        #cell = MyRnnWrapper("xxx",hidden_dim)
        cell = MyRnnWrapper2(tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim),"xxx",hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
        
        
        
        #######################################################
        
        encoder_outputs = tf.ones([batch_size,20,30])
        #encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32))
        input_lengths = [20]*batch_size
        
        # attention
        #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=True)
        #attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=True)
        attention_mechanism = BahdanauMonotonicAttention_hccho(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=True)
        
        
        
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=13,alignment_history=alignment_history_flag,output_attention=True)
    
        #######################################################
    
        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        
        
        
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
        else:
            #helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
            helper = MyRnnHelper(embedding,batch_size,embedding_dim)
    
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다.
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
        
        opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            loss_,_ =sess.run([loss,opt])
            print("{} loss: = {}".format(i,loss_))        

        ######################################################
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,"\n argmax --> ",o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size   

if __name__ == "__main__":
    wapper_test()
    #wapper_attention_test()

