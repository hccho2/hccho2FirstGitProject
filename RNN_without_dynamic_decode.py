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


1. 아래처럼 직접 loop 돌린다.
2. 0 한개로 만든 garbage input으르 만들어 넣는다.
3. AttentinWrapper를 customization한다.  AttentinWrapper에서 attention vector와 input이 concat되기 때문에, 이부분을 변경한다.


"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow에서는 5가지의 로깅 타입을 제공하고 있습니다. ( DEBUG, INFO, WARN, ERROR, FATAL ) INFO가 설정되면, 그 이하는 다 출력된다.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test1():
    # free forcing
    batch_size = 2
    hidden_dim = 3
    input_dims = 2
    T = 5
    init_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_dims))
    init_state = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)
    
    # RNNCell, FC, BN layer만 선언. 이후 data를 넣는 것은 다음 단계에서...
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    output_layer = tf.layers.Dense(input_dims, name='output_projection')
    BN = tf.layers.BatchNormalization()
    outs = []
    
    
    inputs = init_inputs
    new_state = init_state
    for _ in range(T):
        new_state,outputs = cell(inputs,new_state)

        outputs = output_layer(outputs)
        outputs = BN(outputs)
        inputs = outputs
        outs.append(outputs)
    
    outs = tf.stack(outs,axis=1)
    print(outs)
    print(tf.trainable_variables())
    

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    inp = np.random.randn(batch_size,input_dims)
    X = sess.run(outs,feed_dict={init_inputs: inp})
    print(X.shape,'\n', X)
    

    
    
    
def test11():
    # teacher forcing vs dynamic rnn
    batch_size = 2
    hidden_dim = 3
    input_dims = 2
    T = 5
    inputs = tf.placeholder(tf.float32, shape=(batch_size,T,input_dims))
    init_state = tf.zeros(shape=(batch_size,hidden_dim),dtype=tf.float32)
    
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    output_layer = tf.layers.Dense(input_dims, name='output_projection')
    BN = tf.layers.BatchNormalization()
    
    
    outs = []
    new_state = init_state
    for i in range(T):
        new_state,outputs = cell(inputs[:,i,:],new_state)

        outputs = output_layer(outputs)
        outputs = BN(outputs)
        outs.append(outputs)
    
    outs = tf.stack(outs,axis=1)
    print(outs)
    print(tf.trainable_variables())
    

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    inp = np.random.randn(batch_size,T,input_dims)
    X = sess.run(outs,feed_dict={inputs: inp})
    print("for loop: ", X.shape,'\n', X)


    outs2, last_state = tf.nn.dynamic_rnn(cell,inputs,sequence_length=[T]*batch_size,initial_state=init_state) 
    outs2 = output_layer(outs2)
    outs2 = BN(tf.reshape(outs2,[-1,input_dims])) # BN 적용을 위해 2-dim으로 
    outs2 = tf.reshape(outs2,(batch_size,-1,input_dims)) # 다시 3-dim으로

    Y = sess.run(outs2,feed_dict={inputs: inp})
    print("dynamic_rnn: ", Y.shape,'\n', Y)

    print(np.allclose(X,Y))




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


#test1()
test11()    
print("Done")
