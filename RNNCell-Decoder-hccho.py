# coding: utf-8
# user defined Wrapper
import tensorflow as tf
import numpy as np
import collections
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
import functools


tf.reset_default_graph()
vocab_size = 5
SOS_token = 0
EOS_token = 4

class CustomDecoderOutput(collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
    # BasicDecoderOutput의 return에 "token_output"이 추가되어 있다.
    # class BasicDecoderOutput(collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
    pass


class CustomDecoder(decoder.Decoder):
    """Custom sampling decoder.

    Allows for stop token prediction at inference time
    and returns equivalent loss in training time.

    Note:
    Only use this decoder with Tacotron 2 as it only accepts tacotron custom helpers
    """

    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize CustomDecoder.
        Args:
                cell: An `RNNCell` instance.
                helper: A `Helper` instance.
                initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
                        The initial state of the RNNCell.
                output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
                        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
                        to storing the result or sampling.
        Raises:
                TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        rnn_cell_impl.assert_like_rnncell(type(cell), cell)
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None
                        and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                            "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                            lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                            size)
            layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
                            output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        #BasicDecoderOutput에 비해, "token_output"을 위한 항목이 추가되어 있다.
        # Return the cell output and the id
        return CustomDecoderOutput(
                        rnn_output=self._rnn_output_size(),
                        token_output=self._helper.token_output_size,
                        sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return CustomDecoderOutput(
                        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
                        tf.float32,
                        self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
                name: Name scope for any created operations.
        Returns:
                `(finished, first_inputs, initial_state)`.
        """
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """Perform a custom decoding step.
        Enables for dyanmic <stop_token> prediction
        Args:
                time: scalar `int32` tensor.
                inputs: A (structure of) input tensors.
                state: A (structure of) state tensors and TensorArrays.
                name: Name scope for any created operations.
        Returns:
                `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
            #Call outputprojection wrapper cell
            (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

            #apply output_layer (if existant)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            sample_ids = self._helper.sample(
                            time=time, outputs=cell_outputs, state=cell_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                            time=time,
                            outputs=cell_outputs,
                            state=cell_state,
                            sample_ids=sample_ids,
                            stop_token_preds=stop_token)

        outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
        return (outputs, next_state, next_inputs, finished)



def decoder_test():
    x_data = np.array([[SOS_token, 3, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[1,2,0,3,2,EOS_token],[3,2,3,3,1,EOS_token],[3,1,1,2,0,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =6
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        cells = []
        for _ in range(num_layers):
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)    
        #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
    
        # tf.contrib.rnn.OutputProjectionWrapper  마지막에 FC layer를 하나 더 추가하는 효과. 아래에서 적용하는 Dense보다 앞에 적용된다. Dense가 있기 때문에 OutputProjectionWrapper 또는 Dense로 처리 가능함
        # FC layer를 multiple로 적용하려면 OutputProjectionWrapper을 사용해야 함.
        if True:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,13)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,17)
    

        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
    
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        # last_state는 num_layers 만큼 나온다.
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)   # logit: batch_major(N,T,output_dim) --> dynamic_decode의 output_time_major=True와 아귀가 맞지 않음.
    
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
    
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )


if __name__ == "__main__":
    decoder_test()

