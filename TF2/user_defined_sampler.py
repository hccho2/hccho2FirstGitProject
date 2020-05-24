
'''


'''

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant

from tensorflow_addons.seq2seq import Sampler



class MySampler(Sampler):
    # GreedyEmbeddingSampler를 만들어 보자.
    # initializer로 넘겨받는 것들을, init에서 넘겨 받을 수도 있다.
    def __init__(self):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])   # sample_ids의 shape이 (batch_size,) 이므로, batch_size를 제외하면, "[]"이 된다.


    def next_inputs(self, time, outputs, state, sample_ids, name=None):   # time+1을 위한 input을 만든다., outputs,state,sample_ids는 time step에서의 결과이다.
        # 넘어오는 sample_ids는 sample 함수에어 계산된어 넘어온 값이다.   <----- 이런 계산은 BasicDecoder의 'step' 함수에서 이루어 진다.
        # next input을 계산하기 위해서 sample_ids를 이용하거나, outputs를 이용하거나 선택하면 된다.
        
        finished = tf.equal(sample_ids, self.end_token)
        next_inputs = tf.nn.embedding_lookup(self.embedding,sample_ids)
        return (finished, next_inputs, state)  #finished==True이면 next_inputs,state는 의미가 없다.

    def initialize(self, embedding, start_tokens=None, end_token=None):
        # 시작하는 input을 정의한다.
        # return (finished, first_inputs). finished는 시작이니까, 무조건 False
        # first_inputs는 예를 위해서, SOS_token으로 만들어 보았다.
        self.embedding = embedding
        self.end_token = end_token
        self._batch_size = tf.size(start_tokens)
        return (tf.tile([False], [self._batch_size]), tf.nn.embedding_lookup(self.embedding,start_tokens))  

    def sample(self, time, outputs, state, name=None):
        return tf.argmax(outputs, axis=-1,output_type=tf.int32)






def user_defined_sampler_decoder_test():

    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7

    seq_length = x_data.shape[1]
    embedding_dim = 8

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    ##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 
    
    inputs = embedding(x_data)


    # Decoder
    
    method = 2
    if method==1:
        # single layer RNN
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
        # decoder init state:
        
        #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
        init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    else:
        # multi layer RNN
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
        init_state = decoder_cell.get_initial_state(inputs=inputs)  #inputs=tf.zeros_like(x_data,dtype=tf.float32)로 해도 됨. inputs의 batch_size만 참조하기 때문에
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    

    
    sampler = MySampler()

    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)
    outputs, last_state, last_sequence_lengths = decoder(embedding.weights,initial_state=init_state,
                                                         start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token,training=False)  
    



    logits = outputs.rnn_output
    
    print(logits.shape)





if __name__ == '__main__':
    user_defined_sampler_decoder_test()
