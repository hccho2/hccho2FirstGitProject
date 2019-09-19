# coding: utf-8
import tensorflow as tf
import numpy as np
import librosa
from skimage import io
import time
from matplotlib import pyplot as plt
from tensorflow.python.layers.core import Dense
tf.reset_default_graph()

def f(a,b):
    return a*a+3*b

x = tf.Variable(4.0,dtype=tf.double)
y = tf.placeholder(tf.double,None)
L = tf.square(x - y*y-3*x)
#L2 = tf.square(x-tf.py_func(f,[y,x],Tout=tf.double))
L2 = tf.square(x-tf.py_func(f,[y,x],Tout=tf.double))  #py_func으로 들어간 부분은 constant 취급된다.


g = tf.gradients(L,x)
g2 = tf.gradients(L2,x)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([L,L2],feed_dict={y:np.array(2.0).astype(np.double)}))
print(sess.run([g,g2],feed_dict={y:np.array(2.0).astype(np.double)}))
print(sess.run([g,g2],feed_dict={y:np.array(2.0).astype(np.double)}))