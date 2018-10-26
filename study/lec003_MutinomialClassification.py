# coding: utf-8
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
np.set_printoptions(threshold=np.nan)

def change_ont_hot_label(X,n_dim=10):
    X = X.flatten()
    T = np.zeros((X.size, n_dim))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T 

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    # format of t.
    # 1. one_hot encoding format
    # 2. [3,1,3,0,5]
    # 3. [[3],[1],[3],[0],[5]]
    if y.ndim == 1:  # Data가 1개 들어오면, 강제로 차원을 높임.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
         
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    if t.ndim ==2:
        t = t.flatten()
              
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7)) / batch_size

########################################################################
########################################################################
########################################################################
mydata = np.genfromtxt('zoo.txt',delimiter=',',dtype=np.float32)
mydata=np.insert(mydata,0,np.ones(mydata.shape[0]),axis=1)

mydata_train = mydata[:90,:]
mydata_test = mydata[90:,:]

A = mydata_train[:, 0:-1]
B = mydata_train[:, [-1]].astype(int)   # mydata[:, [-1]]  == mydata[:, -1].reshape(-1,1)

N_Class = 7



def MultinomialClassification():
    learning_rate=1e-3
    N_Data = A.shape[0]
    N_Feature = A.shape[1]
    
    BB = change_ont_hot_label(B,N_Class)
    
    
    W = np.random.standard_normal(size=(N_Feature,N_Class))    


    total_cost = []
    for step in range(300000):
        temp = (softmax(np.dot(A,W)) - BB) /N_Data
        W -= learning_rate * np.dot(A.T, temp)
        
        if step % 10000 == 0:
            total_cost.append(cross_entropy_error(softmax(np.dot(A,W)),B))
            print(step, total_cost[-1])

    prediction = np.argmax(softmax(np.dot(A,W)),axis=1)
    acc = np.mean(1*(prediction==B.flatten()))
    print("Accuracy for Training Data: ", acc)
    
    prediction = np.argmax(softmax(np.dot(mydata_test[:,0:-1],W)),axis=1)
    acc = np.mean(1*(prediction==mydata_test[:,-1]))   
    print("Accuracy for Test Data: ", acc)
    
    print(mydata_test, prediction)
    plt.plot(total_cost)
    
    
def MultinomialClassificationTF():
    learning_rate=1e-3
    N_Data = A.shape[0]
    N_Feature = A.shape[1]
    
    X = tf.placeholder(tf.float32, shape=[None, N_Feature])
    Y = tf.placeholder(tf.int32, shape=[None, 1])  
    BB = tf.reshape(tf.one_hot(Y,N_Class),[-1,N_Class])
    
    W = tf.Variable(tf.random_normal([N_Feature, N_Class],mean=0.0,stddev=1.0/np.sqrt(100),dtype=tf.float32), name='weight')
    logits = tf.matmul(X,W)
    hypothesis = tf.nn.softmax(logits)  
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=BB))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    
    # For output    
    prediction = tf.argmax(hypothesis,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(BB,1)), dtype=tf.float32))      
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(300000):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: A, Y: B})

            if step % 10000 == 0:
                total_cost.append(cost_val)    
                print(step, total_cost[-1])

        acc = sess.run([accuracy],feed_dict={X: A, Y: B})     
        print("Accuracy for Training Data: ", acc) 
        
        acc,pred = sess.run([accuracy,prediction],feed_dict={X: mydata_test[:, 0:-1], Y: mydata_test[:, [-1]].astype(int)})
        print("Accuracy for Training Data: ", acc) 
        print(mydata_test, pred)
        
    plt.plot(total_cost)


if __name__ == "__main__":
    start = time.time()
    print ((datetime.datetime.now()), " Start")
    MultinomialClassification()
    #MultinomialClassificationTF()
    
    
    print ((datetime.datetime.now()), " Finish")    
    finish = time.time()
    print (int((finish - start)/60.0), "Min", (finish - start)%60, "Sec elapsed") 
