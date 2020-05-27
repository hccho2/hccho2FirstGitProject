# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.seq2seq import Sampler








vocab = 20
embedding_dim = 6
timestep = 20  #encoder time step
memory_size = 15

attention_cls = tfa.seq2seq.BahdanauAttention
units = 11
batch = 3



inputs = tf.keras.Input(shape=[timestep])
encoder_input = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)(inputs)


encoder_output = tf.keras.layers.LSTM(memory_size, return_sequences=True)( encoder_input )

attention = attention_cls(units, encoder_output)
query = tf.keras.Input(shape=[units])  # 이전 step attention
state = tf.keras.Input(shape=[timestep])   # 이전 alignment

score = attention([query, state])

x_test = np.random.randint(vocab, size=(batch, timestep))
model = tf.keras.Model([inputs, query, state], score)










