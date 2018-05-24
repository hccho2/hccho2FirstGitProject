# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


from tensorflow.python.layers.core import Dense
tf.reset_default_graph()

vocab_size = 5
SOS_token = 0
EOS_token = 4

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
init_state_flag = 0
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

    if init_state_flag==0:
         initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
    else:
        if state_tuple_mode:
            h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
            initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)
            
        else:
            h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
            initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
    if train_mode:
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)

    output_layer = Dense(output_dim, name='output_projection')
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
    # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
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
    
    print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
    print("bias",sess.run(output_layer.trainable_weights[1]))  # bias

    if train_mode:
        p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
        print("loss: {:20.6f}".format(sess.run(loss)))
        print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )