
#  coding: utf-8
import tensorflow as tf
import numpy as np

"""

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,1]]).T
x = tf.placeholder(tf.float32, [None,3])
y = tf.placeholder(tf.float32, [None,1])

L1 = tf.layers.dense(x,units=4, activation = tf.sigmoid,name='L1')
L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
train = tf.train.AdamOptimizer(learning_rate=1).minimize( tf.reduce_mean( 0.5*tf.square(L2-y)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(60000):
        sess.run(train, feed_dict={x: X, y: Y})
"""

x = tf.placeholder(tf.float32, [None,284])
with tf.variable_scope("hccho"):
    fc1 = tf.layers.dense(x,units=1024, activation = tf.nn.relu,name='fc1')
    fc2 = tf.layers.dense(fc1,units=10, activation = None,name='fc2')    
    out = tf.nn.softmax(fc2)    

image = np.random.randn(5,284)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_ = sess.run(out,feed_dict={x: image})
    

var_list = tf.trainable_variables()   
var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(var_list)
print(var_list2)    

====================================================
def simple_net():
    tf.reset_default_graph()
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
    Y = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T
    
    W = tf.get_variable('weight', dtype=tf.float32,shape=[3,1], initializer=tf.initializers.constant(1))
    b = tf.get_variable('bias',dtype=tf.float32,shape=[1],initializer=tf.initializers.zeros())
    Z = tf.matmul(X,W)+b
    
    loss = tf.nn.l2_loss(Z-Y)
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        
        for step in range(100):
            _, loss_ = sess.run([opt,loss])
            print('{}:  loss = {}'.format(step,loss_))

====================================================

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
A = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])

# Feature Scaling을 하면, learning rate을   1.0e-5 ==> 1.0e-2로 조정해야 함
#A = (A-np.mean(A,0))/np.std(A,0)

B = np.array([[152.],[185.],[180.],[196.],[142.]])


def MultivariateRegressionTF():
    tf.reset_default_graph()

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W1 = tf.Variable(tf.random_normal([3, 5]), name='W1')
    b1 = tf.Variable(tf.random_normal([5]), name='b1')
    W2 = tf.Variable(tf.random_normal([5, 1]), name='W2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')    
    # Hypothesis
    L1 = tf.nn.relu( tf.matmul(X, W1) + b1)
    logit = tf.matmul(L1, W2) + b2
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(logit - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, hy_val, _ = sess.run(
            [cost, logit, train], feed_dict={X: A, Y: B})
        if step % 5000 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    print("W1, b1, W2, b2 ", sess.run([W1,b1,W2,b2]))

def MultivariateRegressionTF2():
    import matplotlib.pyplot as plt
    tf.reset_default_graph()
    
    mydata = np.genfromtxt('mydata2.txt',delimiter=',',dtype=np.float32)
    A = mydata[:,0:2]
    B = mydata[:,-1].reshape(-1,1)  # mydata[:,2:3]
    plt.subplot(131)
    plt.scatter(A[:, 0], A[:, 1], c=B.flatten(),marker=">")
    

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    L1 = tf.layers.dense(X,units=5, activation = tf.nn.relu,name='L1')
    logit = tf.layers.dense(L1,units=1, activation = None,name='L2')
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(logit - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, hy_val, _ = sess.run(
            [cost, logit, train], feed_dict={X: A, Y: B})
        if step % 5000 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    var_list = tf.trainable_variables()
    print(var_list)
    print(sess.run(var_list))
    graph = tf.get_default_graph()
    print(sess.run(graph.get_tensor_by_name(name='L1/kernel:0')))

def LogisticRegressionTF4():    
    # veiwing gradient values within tensorflow
    tf.reset_default_graph()
    
    #AA2 = AddFeatures(AA[:,1:],3)
    AA2 = A
    
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, AA2.shape[1]])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    
    mode = 1
    
    if mode == 0:
        logits = tf.layers.dense(X,units=1, activation = None,name='L1')
    else:
        L1 = tf.layers.dense(X,units=5, activation = tf.nn.relu,name='L1')
        logits = tf.layers.dense(L1,units=1, activation = None,name='L2')
    
    

    
    # Simplified cost/loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y ))
    
    # Minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    
    predicted = tf.cast(logits >= 0, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(10000):
            _, l, acc,p = sess.run([train,loss,accuracy,predicted], feed_dict={X: AA2, Y: B})
            if step % 1000 == 0:
                print(step, l)
                total_cost.append(l)
        # Accuracy report
        print("\nAccuracy: ", acc, "\nloss", l)    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=p.flatten(),marker=">")    
    plt.subplot(133)
    plt.plot(total_cost)
    sess.close()  
    
    
def sin_fitting():
    
    X = np.linspace(0, 15, 301)
    Y =  np.sin(2*X - 0.1)+ np.random.normal(size=len(X), scale=0.2)
    X = X.reshape(-1,1)
    Y= Y.reshape(-1,1)
    plt.plot(X,Y)
    
    x = tf.placeholder(tf.float32, [None,1])
    y = tf.placeholder(tf.float32, [None,1])
    
    L1 = tf.layers.dense(x,units=10, activation = tf.nn.sigmoid,name='L1')
    L2 = tf.layers.dense(L1,units=10, activation = tf.nn.sigmoid,name='L2')
    L3 = tf.layers.dense(L2,units=1, activation = None,name='L3')
    loss = tf.reduce_mean( 0.5*tf.square(L3-y))
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(30000):
            sess.run(train, feed_dict={x: X, y: Y})
            if j%4000 ==0:
                loss_ = sess.run(loss, feed_dict={x: X, y: Y})
                print("{}: loss = {}".format(j,loss_ ))
    
        Y_ = sess.run(L3,feed_dict={x:X})
        plt.plot(X,Y_)

if __name__ == "__main__":
    MultivariateRegressionTF()
    MultivariateRegressionTF2()





