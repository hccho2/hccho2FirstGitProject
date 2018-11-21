# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(threshold=np.nan)



mydata = np.genfromtxt('mydata2.txt',delimiter=',',dtype=np.float32)
A = mydata[:,0:2]
B = mydata[:,-1].reshape(-1,1)  # mydata[:,2:3]
plt.subplot(131)
plt.scatter(A[:, 0], A[:, 1], c=B.flatten(),marker=">")


def LogisticRegression():   
    # Data 변환 없이 Logistic Regression 적용
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    hypothesis = tf.layers.dense(X,units=1,activation=tf.nn.sigmoid)
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *tf.log(1 - hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(cost)    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    total_cost = []
    for step in range(20000):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: A, Y: B})
        total_cost.append(cost_val)
        if step % 1000 == 0:
            print(step, "Cost: ", cost_val)
            


    print("Cost: ", sess.run(cost,feed_dict={X: A, Y: B}))
    prediction = sess.run(hypothesis,feed_dict={X: A}) >= 0.5
    acc = np.mean(1*(prediction==B))
    print("Accuracy: ", acc)
    
    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=prediction.flatten(),marker=">")
    plt.subplot(133)
    plt.plot(total_cost)

    plt.show()

def LogisticRegression2():   
    # data feature를 추가
    
    AA = [[x[0],x[1],x[0]**2,x[0]*x[1],x[1]**2] for x in A]
    #AA = [[x[0],x[1],x[0]**2,x[0]*x[1],x[1]**2,x[0]**3,x[0]**2*x[1],x[0]*x[1]**2,x[1]**3] for x in A]
    X = tf.placeholder(tf.float32, shape=[None, 5])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    logits = tf.layers.dense(X,units=1,activation=None)
    hypothesis = tf.nn.sigmoid(logits)
    
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *tf.log(1 - hypothesis))
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(cost)    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    total_cost = []
    for step in range(20000):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: AA, Y: B})
        total_cost.append(cost_val)
        if step % 1000 == 0:
            print(step, "Cost: ", cost_val)
            


    print("Cost: ", sess.run(cost,feed_dict={X: AA, Y: B}))
    prediction = sess.run(hypothesis,feed_dict={X: AA}) >= 0.5
    acc = np.mean(1*(prediction==B))
    print("Accuracy: ", acc)
    
    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=prediction.flatten(),marker=">")
    plt.subplot(133)
    plt.plot(total_cost)

    plt.show()
def LogisticRegression3():   
    # deep net
    
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    x1 = tf.layers.dense(X,units=10,activation=tf.nn.relu)  # units=200 으로 하면 acc = 1.0
    logits = tf.layers.dense(x1,units=1,activation=None)
   
    hypothesis = tf.nn.sigmoid(logits)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(cost)    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    total_cost = []
    for step in range(20000):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: A, Y: B})
        total_cost.append(cost_val)
        if step % 1000 == 0:
            print(step, "Cost: ", cost_val)
            


    print("Cost: ", sess.run(cost,feed_dict={X: A, Y: B}))
    prediction = sess.run(hypothesis,feed_dict={X: A}) >= 0.5
    acc = np.mean(1*(prediction==B))
    print("Accuracy: ", acc)
    
    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=prediction.flatten(),marker=">")
    plt.subplot(133)
    plt.plot(total_cost)

    plt.show()
    
    plot_decision_boundary(A,B,sess,hypothesis,X)
    
def plot_decision_boundary(A,B,sess,hypothesis,X):
    # Set min and max values and give it some padding
    x_min, x_max = A[:, 0].min() - .5, A[:, 0].max() + .5
    y_min, y_max = A[:, 1].min() - .5, A[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = sess.run(hypothesis,feed_dict={X: np.c_[xx.ravel(), yy.ravel()]}) >= 0.5
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(A[:, 0], A[:, 1], c=B.flatten(), cmap=plt.cm.Greens,s=4)    
    plt.show()
    
if __name__ == "__main__":    
    #LogisticRegression()
    #LogisticRegression2()
    LogisticRegression3()