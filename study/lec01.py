# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
"""
a = [[[  4,   3,   6,   7],
     [ -5,   5,   0,  -6],
     [ -4,   9,  -6,   1]],

     [[ -7,   1,  -3,  -1],
      [ -6,  16,  10,  12],
      [  6,   5,   8, -15]]]

A = np.array(a)

A.shape
np.argmax(A)
np.argmax(A,1)
A = np.concatenate((np.ones((A.shape[0],1)),A), axis=1)
A = np.insert(A,0,np.ones(A.shape[0]),axis=1)
"""



A = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])


B = np.array([[152.],[185.],[180.],[196.],[142.]])

def MultivariateRegression():
    learning_rate=1e-5
    N_Data = A.shape[0]
    
    W = np.ones((3,1))
    b = np.zeros(1)
    
    for step in range(2000):
        temp = 2*(np.dot(A,W)+b - B)/N_Data  # (5,1)
        W -= learning_rate * np.dot(A.T, temp)  # (3,5)x(5,1) ==> (3,1)
        b -= learning_rate * np.sum(temp,axis=0) # (5,1) ==> (1)
        
    print("W,b: ", W, b) 
    print("Cost: ", np.mean((np.dot(A,W)+b -B)**2))
    B_ = np.dot(A,W)+b
    print("Prediction: ", B_)
    bb, = plt.plot(B,label='raw')
    bb_, = plt.plot(B_,label='prediction')
    plt.legend(handles=[bb, bb_])
    plt.show()
    
def MultivariateRegressionTF():
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
#    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
#    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    W = tf.Variable(tf.ones([3, 1]), name='weight')
    b = tf.Variable(tf.zeros([1]), name='bias')  
    
#     W = tf.get_variable( name='weight',shape=[3,1],initializer=tf.initializers.random_normal())
#     b = tf.get_variable( name='bias',shape=[1],initializer=tf.initializers.zeros())
    
    
    # Hypothesis
    hypothesis = tf.matmul(X, W) + b
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(2000):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: A, Y: B})
        if step % 500 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    print("W, b ", sess.run([W,b]))
    print("Cost: ", sess.run(cost,feed_dict={X: A, Y: B}))
    print("Prediction: ", sess.run(hypothesis,feed_dict={X: A}))  # Y에 대한 feed가 필요없음
    
    #print(tf.global_variables())
    
    # 새로운 data에 대한 predict(inference)
    new_data = [[90,91,92]]
    print("Prediction for new data: {} --->".format(new_data), sess.run(hypothesis,feed_dict={X: new_data}).T)  # Y에 대한 feed가 필요없음
    
def MultivariateRegressionTF2():   
    # variable 선언없이 dense를 활용 
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    hypothesis = tf.layers.dense(X,units=1)
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(20000):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: A, Y: B})
        if step % 1000 == 0:
            print(step, "Cost: ", cost_val)


    print("Cost: ", sess.run(cost,feed_dict={X: A, Y: B}))
    print("Prediction: ", sess.run(hypothesis,feed_dict={X: A}).T)  # Y에 대한 feed가 필요없음
 

    
    
def MNIST(): 
    from tensorflow.examples.tutorials.mnist import input_data
    import skimage.io
    #tf.set_random_seed(1234)
    mnist = input_data.read_data_sets(".\\mnist", one_hot=False)   #mnist.train.images, mnist.test.labels
    print(mnist.train.labels.shape)
    batch_size = 5
    # mnist.validation.num_examples <---5000, mnist.train.num_examples <---- 55000, mnist.test.num_examples <---- 10000
    print("# of train data", mnist.train.num_examples)
    
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #(batch_size,784)   MNISTdata는 0~1사이값
    
    skimage.io.imshow(batch_xs[0].reshape(28,28))
    plt.show()
    batch_xs = batch_xs.reshape(-1,28,28)
    skimage.io.imshow(np.concatenate(batch_xs,axis=1))
    plt.show()
    print(batch_ys)
    print('Done')
    
if __name__ == "__main__":    
    #MultivariateRegression()
    #MultivariateRegressionTF()
    #MultivariateRegressionTF2()
    MNIST()