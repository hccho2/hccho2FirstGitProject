#  coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def my_func():
    x = np.random.randn(30)
    plt.plot(x)
    plt.savefig("xxx.jpg")
    return 0
def my_func2(filename):
    x = np.random.randn(30)
    plt.plot(x)
    plt.savefig(filename.decode("utf-8"))  # b'xx.jpg' ==> 'xx.jpg'
    return 0

filename = tf.placeholder(tf.string)
a = tf.py_func(my_func,inp=[],Tout=tf.int32)
b = tf.py_func(lambda f: my_func2(f),inp=[filename],Tout=tf.int32)
sess = tf.Session()
sess.run(b,feed_dict={filename: "xx.jpg"})

print('Done')