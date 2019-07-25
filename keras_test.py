# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

def keras_layer_test():

    VOCAB_SIZE =20
    BATCH_SIZE = 3
    T = 10
    EMB_SIZE = 8
    
    x = np.random.randint(VOCAB_SIZE,size=(BATCH_SIZE,T))
    xx = tf.convert_to_tensor(x)
    embedding_layer = keras.layers.Embedding(VOCAB_SIZE,EMB_SIZE)(xx)
    
    conv1 = keras.layers.Conv1D(filters=128,kernel_size=3,padding='valid',activation=tf.nn.relu)(embedding_layer)
    
    pool1 = keras.layers.GlobalMaxPool1D()(conv1)# (3,128)
    
    pool2 = tf.keras.layers.MaxPool1D(conv1.shape.as_list()[1],1)(conv1) #(3,1,128)  
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    a1,a2 = sess.run([pool1,pool2])
    print(np.allclose(a1,np.squeeze(a2)))
    
  

def keras_test2():
    """
    keras는 class object를 먼저 만들고, 다음 단계로 input을 집어 넎는 방식
    
    """
    L1 = tf.keras.layers.Dense(units=39,activation=tf.nn.relu)
    
    
    
    init_inputs = tf.placeholder(tf.float32, shape=(3,4))
    y = L1(init_inputs)
    
    print(y)

