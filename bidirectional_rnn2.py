#  coding: utf-8
""" 
Bidirectional RNN에서, forward와 backward가 독립적으로 계산되고,

이런 Bidirectional RNN의 Multi로 쌓으려면, forward output과 backward output을 concat해서 다음 층의 input이 된다.

"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

vocab_size = 6
SOS_token = 0
EOS_token = 5

x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
print("data shape: ", x_data.shape)
sess = tf.InteractiveSession()

output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =7
num_layers = 2
seq_length = x_data.shape[1]
embedding_dim = 8
state_tuple_mode = True
init_state_flag = 1
init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)

train_mode = False
with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
    # Make rnn
    

    fw_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    bw_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)

    embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim

    Y = tf.convert_to_tensor(y_data)


    fw_initial_state = fw_cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
    bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
    
    

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,inputs,sequence_length=np.array([seq_length]*batch_size),
                                                             initial_state_fw=fw_initial_state, initial_state_bw=bw_initial_state)
    
 




    sess.run(tf.global_variables_initializer())

    print("\n\noutputs: ",outputs)
    o,s = sess.run([outputs,output_states])  #batch_size, seq_length, outputs
    print("outputs: ",  o)
    print("output_states: ",  s)
    
  