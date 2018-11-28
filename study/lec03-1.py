# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import skimage.io
import time
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(threshold=np.nan)



def MNIST_NN1():
    tf.set_random_seed(1234)
    batch_size = 125
    num_epoch= 10
    
    mnist = input_data.read_data_sets(".\\mnist", one_hot=True)   #mnist.train.images, mnist.test.labels
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])

    
    layer1 = tf.layers.dense(X,units=256,activation=tf.nn.relu)  
    layer2 = tf.layers.dense(layer1,units=125,activation=tf.nn.relu) 
    logits = tf.layers.dense(layer2,units=10,activation=None)
   
    predict = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    n_iter = mnist.train.num_examples // batch_size
    s=time.time()
    for i in range(num_epoch):
        for j in range(n_iter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})
        cost_,train_acc = sess.run([cost,accuracy],feed_dict={X:batch_xs,Y:batch_ys})
        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})  # test data 10000개
        print('epoch: {}, loss = {:.4f}, train acc = {:.4f}, test_acc = {:.4f}, elapsed = {:.2f}'.format(i+1,cost_,train_acc,test_acc,time.time()-s))
    
    
    batch_xs, batch_ys = mnist.test.next_batch(10)
    predict_ = sess.run(predict,feed_dict={X:batch_xs})

    print('label: ', np.argmax(batch_ys,axis=1))
    print('predict: ', np.argmax(predict_,axis=1))

    batch_xs = batch_xs.reshape(-1,28,28)
    skimage.io.imshow(np.concatenate(batch_xs,axis=1))
    plt.title('predict:' + str(np.argmax(predict_,axis=1)))
    plt.show()
    
def MNIST_NN2():
    tf.set_random_seed(1234)
    batch_size = 125
    num_epoch= 20
    
    mnist = input_data.read_data_sets(".\\mnist", one_hot=True)   #mnist.train.images, mnist.test.labels
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])
    training = tf.placeholder(tf.bool)
    
    
    init =  tf.contrib.layers.xavier_initializer(uniform=False)
    layer1 = tf.layers.dense(X,units=128,activation=tf.nn.relu,kernel_initializer=init)  
    layer2 = tf.layers.dense(layer1,units=64,activation=tf.nn.relu,kernel_initializer=init) 
    layer2 = tf.layers.dropout(layer2,rate=0.7,training=training)    # rate = drop rate
    logits = tf.layers.dense(layer2,units=10,activation=None,kernel_initializer=init)
   
    predict = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    n_iter = mnist.train.num_examples // batch_size
    s=time.time()
    for i in range(num_epoch):
        for j in range(n_iter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys,training: False})
        cost_,train_acc_ = sess.run([cost,accuracy],feed_dict={X:batch_xs,Y:batch_ys,training: False})
        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,training:False})  # test data 10000개
        print('epoch: {}, loss = {:.4f}, train acc = {:.4f}, test_acc = {:.4f}, elapsed = {:.2f}'.format(i+1,cost_,train_acc_,test_acc,time.time()-s))
    
    
    batch_xs, batch_ys = mnist.test.next_batch(10)
    predict_ = sess.run(predict,feed_dict={X:batch_xs,training: False})

    print('label: ', np.argmax(batch_ys,axis=1))
    print('predict: ', np.argmax(predict_,axis=1))

    batch_xs = batch_xs.reshape(-1,28,28)
    skimage.io.imshow(np.concatenate(batch_xs,axis=1))
    plt.title('predict:' + str(np.argmax(predict_,axis=1)))
    plt.show()
if __name__ == "__main__":    
    s=time.time()
    MNIST_NN1()
    e=time.time()
    
    print('done: {} sec'.format(e-s))