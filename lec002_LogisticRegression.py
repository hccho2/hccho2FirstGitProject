import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
#tf.set_random_seed(777)  # for reproducibility
np.set_printoptions(threshold=np.nan)



mydata = np.genfromtxt('mydata.txt',delimiter=',',dtype=np.float32)
A = mydata[:,0:2]
B = mydata[:,-1].reshape(-1,1)  # mydata[:,2:3]
plt.subplot(131)
plt.scatter(A[:, 0], A[:, 1], c=B,marker=">")


#A = (A-np.mean(A,0))/np.std(A,0)
AA=np.insert(A,0,np.ones(A.shape[0]),axis=1)


def AddFeatures(A,n):
    N=A.shape[0]
    B = np.ones([N,1])
    for i in range(1,n+1):
        for j in range(i+1):
            B=np.append(B,(A[:,0]**(i-j) * A[:,1]**(j)).reshape(N,-1),axis=1)
    return B



def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


def LogisticRegressionFallacy():
    learning_rate=1e-5
    N_Data = A.shape[0]
    
    W = np.random.standard_normal(size=(3,1))
    
    for step in range(2001):
        temp = (np.dot(AA,W) - B)/N_Data
        W -= learning_rate * np.dot(AA.T, temp)
    
        
    print("W: ", W) 
    print("Cost: ", np.mean((np.dot(AA,W) - B)**2)/2 )
    prediction = 1*(np.dot(AA,W)>=0.5)
    acc = np.mean(1*(prediction==B))
    print("Accuracy: ", acc)
    
    plt.subplot(122)
    plt.scatter(A[:, 0], A[:, 1], c=prediction,marker=">")


def LogisticRegression():
    learning_rate=1e-3
    N_Data = A.shape[0]
    
    
    W = np.random.standard_normal(size=(3,1))
    W = np.array([[ 0.8330605 ],[-0.35352278],[-0.44010514]])
    #W = np.zeros([3,1])
    
    total_cost = []
    for step in range(400001):
        temp = (sigmoid(np.dot(AA,W)) - B) /N_Data
        W -= learning_rate * np.dot(AA.T, temp)
        
        if step % 1000 == 0:
            total_cost.append(np.mean( -B*np.log(sigmoid(np.dot(AA,W))+0.0000001) - (1-B)*np.log(1-sigmoid(np.dot(AA,W))+0.0000001)  ))
            print(step, total_cost[-1])

            
    print("W: ", W) 
    print("Cost: ", np.mean( -B*np.log(sigmoid(np.dot(AA,W))+0.0000001) - (1-B)*np.log(1-sigmoid(np.dot(AA,W))+0.0000001)  ) )
    prediction = 1*(sigmoid(np.dot(AA,W))>=0.5)
    acc = np.mean(1*(prediction==B))
    print("Accuracy: ", acc)
    
    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=prediction,marker=">")
    plt.subplot(133)
    plt.plot(total_cost)


def LogisticRegressionTF():    
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    #W = tf.Variable(tf.random_normal([3, 1],mean=0.0,stddev=1.0/np.sqrt(100),dtype=tf.float32), name='weight') 
    W = tf.Variable([[ 0.8330605 ],[-0.35352278],[-0.44010514]], name='weight')
    #W = tf.Variable(tf.zeros([3, 1]), name='weight')  
    hypothesis = tf.sigmoid(tf.matmul(X, W))    
    
    cost = -tf.reduce_mean(Y * tf.log(hypothesis+0.0000001) + (1 - Y) *tf.log(1 - hypothesis+0.0000001))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    #train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    
    predicted = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(400001):
            cost_val, _,ww = sess.run([cost, train,W], feed_dict={X: AA, Y: B})
            #print(step, "W: ",ww)
            if step % 1000 == 0:
                print(step, cost_val)
                total_cost.append(cost_val)
        # Accuracy report
        h, p, a, w,c = sess.run([hypothesis, predicted, accuracy, W,cost],feed_dict={X: AA, Y: B})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", p, "\nAccuracy: ", a, "\nW",w, "\ncost", c)    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=p,marker=">")    
    plt.subplot(133)
    plt.plot(total_cost)
    sess.close()
    
def LogisticRegressionTF2():
    # manually gradient updating within tensorflow    
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    #W = tf.Variable(tf.random_normal([3, 1],mean=0.0,stddev=1.0/np.sqrt(100),dtype=tf.float32), name='weight') 
    W = tf.Variable([[ 0.8330605 ],[-0.35352278],[-0.44010514]], name='weight')
    #W = tf.Variable(tf.zeros([3, 1]), name='weight')  
    hypothesis = tf.sigmoid(tf.matmul(X, W))    
    
    cost = -tf.reduce_mean(Y * tf.log(hypothesis+0.0000001) + (1 - Y) *tf.log(1 - hypothesis+0.0000001))

    
    N_Data = A.shape[0]
    gradient = tf.matmul(tf.transpose(X),(hypothesis-Y)/N_Data)
    descent = W - 0.001*gradient
    update = W.assign(descent)
    
    predicted = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(400001):
            cost_val, _,ww = sess.run([cost,update,W], feed_dict={X: AA, Y: B})
            if step % 1000 == 0:
                print(step, cost_val)
                total_cost.append(cost_val)
        # Accuracy report
        h, p, a, w,c = sess.run([hypothesis, predicted, accuracy, W,cost],feed_dict={X: AA, Y: B})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", p, "\nAccuracy: ", a, "\nW",w, "\ncost", c)    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=p,marker=">")    
    plt.subplot(133)
    plt.plot(total_cost)
    sess.close()    
def LogisticRegressionTF3():    
    # veiwing gradient values within tensorflow
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])    
    
    #W = tf.Variable(tf.random_normal([3, 1],mean=0.0,stddev=1.0/np.sqrt(100),dtype=tf.float32), name='weight') 
    W = tf.Variable([[ 0.8330605 ],[-0.35352278],[-0.44010514]], name='weight')
    #W = tf.Variable(tf.zeros([3, 1]), name='weight')  
    hypothesis = tf.sigmoid(tf.matmul(X, W))    
    
    cost = -tf.reduce_mean(Y * tf.log(hypothesis+0.0000001) + (1 - Y) *tf.log(1 - hypothesis+0.0000001))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    gradient = optimizer.compute_gradients(cost,[W])
    update = optimizer.apply_gradients(gradient)

    
    N_Data = A.shape[0]
    manual_gradient = tf.matmul(tf.transpose(X),(hypothesis-Y)/N_Data)

    
    predicted = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(10):
            g, _ ,mg = sess.run([gradient,update,manual_gradient], feed_dict={X: AA, Y: B})
            cost_val= sess.run(cost, feed_dict={X: AA, Y: B})
            if step % 1 == 0:
                print(step, g, mg)
                total_cost.append(cost_val)
        # Accuracy report
        h, p, a, w,c = sess.run([hypothesis, predicted, accuracy, W,cost],feed_dict={X: AA, Y: B})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", p, "\nAccuracy: ", a, "\nW",w, "\ncost", c)    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=p,marker=">")    
    plt.subplot(133)
    plt.plot(total_cost)
    sess.close()     
if __name__ == "__main__":
    start = time.time()
    print ((datetime.datetime.now()), " Start")
    #LogisticRegressionFallacy()
    #LogisticRegression()
    #LogisticRegressionTF()
    #LogisticRegressionTF2()
    LogisticRegressionTF3()
    
    print ((datetime.datetime.now()), " Finish")    
    finish = time.time()
    print (int((finish - start)/60.0), "Min", (finish - start)%60, "Sec elapsed")    
    
    
