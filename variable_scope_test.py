# -*- coding: utf-8 -*-
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph()
def G(name,input):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(input,units=10)
    return out


x1 = tf.placeholder(tf.float32,[None,100])
x2 = tf.placeholder(tf.float32,[None,100])

y = G('a',x1)
z = G('a',x2)
w = G('b',x1)