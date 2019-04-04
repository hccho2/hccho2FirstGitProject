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

class MyRnnWrapper(RNNCell):
    # tf.contrib.seq2seq.InferenceHelper 테스트를 위한 Wrapper
    # input data를 모두 합해서 hidden state로 넘긴다.  h_t = sum(x_t) + h_{t-1}
    def __init__(self,name):
        super(MyRnnWrapper, self).__init__(name=name)
        self.sate_size = 1

    def build(self, inputs_shape):

        self.inputs_shape = inputs_shape.as_list()
        self.built = True

    @property
    def output_size(self):
        return 1

    @property
    def state_size(self):
        return self.sate_size  

    # 다음의 call은 내부적으로 __call__과 연결되어 있다.
    def call(self, inputs, state):
        # 이 call 함수를 통해 cell과 cell이 연결된다.
        # input에 필요에 따라, 원하는 작업을 하면 된다.
        
        
        next_state = state + tf.reduce_sum(inputs,axis=-1)
        cell_output = next_state
        return cell_output, next_state 


    # zero_state는 반드시 재정의해야 하는 것은 아니다. 필요에 따라...
    def zero_state(self,batch_size,dtype=tf.float32):
        return tf.ones([batch_size,self.sate_size],dtype)  # test 목적으로 1을 넣어 봄


def wapper_test():
    batch_size =1
    output_dim = 2
    input_dim = output_dim
    with tf.variable_scope('test') as scope:
        # Make rnn
        cell = MyRnnWrapper("xxx")
        
    
        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        
        
        def _sample_fn(decoder_outputs):
            # (batch_size,output_dim) shape을 만들어 return해야 한다.
            return tf.concat([decoder_outputs,decoder_outputs+1],axis=-1)  # decoder_outputs은 MyRnnWrapper의 cell_output  --> next step의 input을 만든다.
        def _end_fn(sample_ids):
            # infinite
            return tf.tile([False], [batch_size])
        helper = tf.contrib.seq2seq.InferenceHelper(
            sample_fn=_sample_fn,
            sample_shape=[output_dim],   # sample_fn의 output dimension
            sample_dtype=tf.float32,
            start_inputs=[[1.0,3.0]],
            end_fn=_end_fn,
        )
        
        #BasicDecoder는 clas
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state)    
        

        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=5)
    
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        print("\n",o) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
    



if __name__ == "__main__":
    wapper_test()


