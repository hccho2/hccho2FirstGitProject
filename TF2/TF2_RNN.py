# coding: utf-8

'''


'''


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.initializers import Constant
from tensorflow.python.ops.gen_batch_ops import batch

# Build RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]

vocab_size = 6
SOS_token = 0
EOS_token = 5

x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
print("data shape: ", x_data.shape)


output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =7
num_layers = 2
seq_length = x_data.shape[1]
embedding_dim = 8
state_tuple_mode = True
init_state_flag = 0
init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)

embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
input = embedding(x_data)


encoder = tf.keras.layers.LSTM(hidden_dim, return_state=True)
#encoder_outputs, state_h, state_c = encoder(input)
#encoder_state = [state_h, state_c]
init_state = [tf.zeros((batch_size,hidden_dim)), tf.zeros((batch_size,hidden_dim))]


# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
projection_layer = tf.keras.layers.Dense(output_dim)
decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)

outputs, _, _ = decoder(input,initial_state=init_state, sequence_length=[seq_length]*batch_size)
logits = outputs.rnn_output

print(logits.shape)

print('Done')