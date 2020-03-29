# coding: utf-8
import tensorflow as tf
import numpy as np
import librosa
from skimage import io
import time
from matplotlib import pyplot as plt
from tensorflow.python.layers.core import Dense
tf.reset_default_graph()

def test1():
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
    print(sess.run([L,L2],feed_dict={y:np.array(2.0).astype(np.double)}))  # (4-2*2-3*4)^2 = (-12)^2 = 144
    print(sess.run([g,g2],feed_dict={y:np.array(2.0).astype(np.double)}))
    print(sess.run([g,g2],feed_dict={y:np.array(2.0).astype(np.double)}))

    
    
def test2():
    x_train = [1,2,3]
    y_train = [5,4,3]

    W = tf.Variable(tf.random_normal([1]),name='Weight')
    b = tf.Variable(tf.random_normal([1]),name='bias')

    y1 = x_train * W + b
    y2 = x_train * W + b

    hypothesis = 2*y1 + y2

    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(cost)


    grad = tf.gradients(cost,[W,b])
    grad1 = tf.gradients(cost,[y1,y2])  # ---> shape: (2,3) = (변수 갯수, batch_size)
    grad2 = tf.gradients([y1,y2],[W,b],grad1)  # grad와 같은 값.


    xxxx = tf.gradients([y1,y2],[W,b])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(4000):
        sess.run(train)

        if i%100 ==0:
            print(i,sess.run(cost), sess.run(W), sess.run(b))

            
def test3():
    batch_size = 2
    x_train = np.array([[1,2,3],[1,2,3]]).astype(np.float32)
    y_train = np.array([[5],[6]]).astype(np.float32)

    W = tf.Variable(tf.random_normal([3,4]),name='Weight')


    y = tf.matmul(x_train , W)
    hypothesis = tf.reduce_sum(y,axis=-1)

    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(cost)


    grad = tf.gradients(cost,[W])


    grad1 = tf.gradients(cost,[y])  # ---> list: 길이는 변수 갯수. [w에 관한 미분, b에 관한 미분]
    grad2 = tf.gradients(y,[W],grad1)  # grad와 같은 값.

    manual_grad = tf.matmul(x_train.T,grad1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2):
        sess.run(train)

        if i%100 ==0:
            print(i,sess.run(cost), sess.run(W))
    
