# -*- coding: utf-8 -*-
"""
Couresa Machine Traslation(Date conversion)구현에서 

attention이 post LSTM의 input이 되는 것을 구현하기위해
time에 대한 loop를 직접 돌리는 방식
decoder의  time length가 정해져 있어야 한다.

특수한 구조의 RNN 모델은 이렇게 구현해야 될 것 같다.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.layers.core import Dense

batch_size = 2
hidden_dim = 3
input_dims = 2
T = 5
init_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_dims))
h0 = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)

cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
output_layer = Dense(input_dims, name='output_projection')
outs = []
inputs = init_inputs
for _ in range(T):
    a = cell(inputs,h0)
    h0 = a[0]
    out = output_layer(a[1])
    inputs = out
    outs.append(out)

print(outs)
print(tf.trainable_variables())



print("Done")