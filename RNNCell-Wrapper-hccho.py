# coding: utf-8
# user defined Wrapper
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.layers.core import Dense

tf.reset_default_graph()
SOS_token = 0
EOS_token = 4
class MyRnnWrapper(RNNCell):
    # property(output_size, state_size) 2개와 call을 정의하면 된다.
    def __init__(self,name,hidden_dim):
        super(MyRnnWrapper, self).__init__(name=name)
        self.sate_size = hidden_dim

    @property
    def output_size(self):
        return 4  # embedding_dim *2

    @property
    def state_size(self):
        return self.sate_size  

    # 다음의 call은 내부적으로 __call__과 연결되어 있다.
    def call(self, inputs, state):
        # 이 call 함수를 통해 cell과 cell이 연결된다.
        # input에 필요에 따라, 원하는 작업을 하면 된다.
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
        return 2 + self.sate_size  # embedding_dim + 

    @property
    def state_size(self):
        return self.sate_size  

    # 다음의 call은 내부적으로 __call__과 연결되어 있다.
    def call(self, inputs, state):
        # 이 call 함수를 통해 cell과 cell이 연결된다.
        # input에 필요에 따라, 원하는 작업을 하면 된다.
        cell_out, cell_state = self.cell(inputs,state)
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

    def next_inputs(self, time, outputs, state,sample_ids, name=None):
        finished = (time + 1 >= 7)
        next_inputs = outputs[:, -self._output_dim:]*2
        return (finished, next_inputs, state)

    def initialize(self, name=None):
        # 시작하는 input을 정의한다.
        # return finished, first_inputs. finished는 시작이니까, 무조건 False
        # first_inputs는 예를 위해서, SOS_token으로 만들어 보았다.
        return (tf.tile([False], [self._batch_size]), tf.nn.embedding_lookup(self._embedding,tf.tile([SOS_token], [self._batch_size])))  

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them




def wapper_test():
    vocab_size = 5
    x_data = np.array([[SOS_token, 3, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
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
        cell = MyRnnWrapper("xxx",hidden_dim)
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
    
        #output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=None)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다.
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
    
    
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
    



if __name__ == "__main__":
    wapper_test()





