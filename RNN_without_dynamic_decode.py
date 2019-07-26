# -*- coding: utf-8 -*-
"""
Couresa Machine Traslation(Date conversion)구현에서 

attention이 post LSTM의 input이 되는 것을 구현하기위해
time에 대한 loop를 직접 돌리는 방식
decoder의  time length가 정해져 있어야 한다.

특수한 구조의 RNN 모델은 이렇게 구현해야 될 것 같다.

tf.layers.Dense
tf.layers.Dropout

tf.layers.Conv1D
tf.layers.Conv2D

tf.layers.MaxPooling1D
tf.layers.MaxPooling2D

tf.layers.BatchNormalization
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test1():
    batch_size = 2
    hidden_dim = 3
    input_dims = 2
    T = 5
    init_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_dims))
    h0 = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)
    
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    output_layer = tf.layers.Dense(input_dims, name='output_projection')
    BN = tf.layers.BatchNormalization()
    outs = []
    inputs = init_inputs
    for _ in range(T):
        a = cell(inputs,h0)
        h0 = a[0]
        out = output_layer(a[1])
        out = BN(out)
        inputs = out
        outs.append(out)
    
    print(outs)
    print(tf.trainable_variables())

def test2():
    # tf.keras.layers.SimpleRNN 이용하기
    batch_size = 2
    hidden_dim = 3
    input_dims = 2
    T = 5
    init_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_dims))
    h0 = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)
    
    cell = tf.keras.layers.SimpleRNN(units=hidden_dim)
    output_layer = tf.layers.Dense(input_dims, name='output_projection')
    BN = tf.layers.BatchNormalization()
    outs = []
    inputs = init_inputs

    for _ in range(T):
        inputs = tf.expand_dims(inputs,axis=1)
        a = cell(inputs,initial_state=h0)
        h0 = a
        out = output_layer(a)
        out = BN(out)
        inputs = out
        outs.append(out)
    
    print(outs)
    print(tf.trainable_variables())


def test3():
    # tf.keras.layers.SimpleRNNCell  이용하기
    # 1 step을 반복적으로 사용하는 것이기 때문에, SimpleRNN보다 SimpleRNNCell이 더 자연스럽다
    batch_size = 2
    hidden_dim = 3
    input_dims = 2
    T = 5
    init_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_dims))
    h0 = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)
    
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_dim)
    output_layer = tf.layers.Dense(input_dims, name='output_projection')
    BN = tf.layers.BatchNormalization()
    outs = []
    inputs = init_inputs
    


    for _ in range(T):
        a = cell(inputs,states=[h0])
        h0 = a[0]
        out = output_layer(a[0])
        out = BN(out)
        inputs = out
        outs.append(out)
     
    print(outs)
    print(tf.trainable_variables())


test3()
    
print("Done")