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


class MyMINIST():
    def __init__(self,sess,name=None):
        self.sess = sess
        self.name=name
        
        self.build()
    def build(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape=[None, 784])
            self.Y = tf.placeholder(tf.float32, shape=[None, 10])
        
            
            layer1 = tf.layers.dense(self.X,units=256,activation=tf.nn.relu)  
            layer2 = tf.layers.dense(layer1,units=125,activation=tf.nn.relu) 
            logits = tf.layers.dense(layer2,units=10,activation=None)
           
            self.predict_ = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(self.predict_, 1), tf.argmax(self.Y, 1))
            self.accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,logits=logits))
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.cost)
            
            
            
    def predict(self,x_test):
        return self.sess.run(self.predict_,feed_dict={self.X:x_test})
    def accuracy(self,x_test,y_test):
        return self.sess.run(self.accuracy_, feed_dict={self.X: x_test, self.Y: y_test})
    def train(self,x_data,y_data):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data})[0]


class MINIST_Ensemble():
    def __init__(self,sess,num_models=1,name=None):
        self.sess = sess
        self.name=name    
        self.num_models = num_models
        self.models = []
        for i in range(num_models):
            self.models.append(MyMINIST(sess,self.name+str(i)))

    def predict(self,x_test):
        pridictions = np.zeros([len(x_test),10])
        for i, m in enumerate(self.models):
            p = m.predict(x_test)
            pridictions += p
        return pridictions

    def accuracy(self,x_test,y_test):
        pridictions = self.predict(x_test)   
        correct_prediction = np.equal(np.argmax(pridictions,1), np.argmax(y_test, 1))
        return np.average(correct_prediction)        
    
    def train(self,x_data,y_data):
        total_cost=0.0
        for i, m in enumerate(self.models):
            total_cost += m.train(x_data,y_data)
        return total_cost / self.num_models
        




def myclass_test():
    tf.set_random_seed(1234)
    batch_size = 125
    num_epoch= 10
    
    mnist = input_data.read_data_sets(".\\mnist", one_hot=True)   #mnist.train.images, mnist.test.labels
    n_iter = mnist.train.num_examples // batch_size
    
    
    sess = tf.Session()
    
    #my_mnist = MyMINIST(sess,'mnist')
    my_mnist = MINIST_Ensemble(sess,5,'mnist')
    
    sess.run(tf.global_variables_initializer())
    s=time.time()
    for i in range(num_epoch):
        for j in range(n_iter): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost = my_mnist.train(batch_xs,batch_ys)
        
        train_acc = my_mnist.accuracy(batch_xs,batch_ys)
        test_acc = my_mnist.accuracy(mnist.test.images,mnist.test.labels)  # test data 10000개
        print('epoch: {}, loss = {:.4f}, train acc = {:.4f}, test_acc = {:.4f}, elapsed = {:.2f}'.format(i+1,cost,train_acc,test_acc,time.time()-s))




    batch_xs, batch_ys = mnist.test.next_batch(10)
    predict_ = my_mnist.predict(batch_xs)
 
    print('label: ', np.argmax(batch_ys,axis=1))
    print('predict: ', np.argmax(predict_,axis=1))
 
    batch_xs = batch_xs.reshape(-1,28,28)
    skimage.io.imshow(np.concatenate(batch_xs,axis=1))
    plt.title('predict:' + str(np.argmax(predict_,axis=1)))
    plt.show()










        
if __name__ == "__main__":    
    s=time.time()
    myclass_test()
    e=time.time()
    
    print('done: {} sec'.format(e-s))