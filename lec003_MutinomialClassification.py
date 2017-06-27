# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime


def change_ont_hot_label(X,n_dim=10):
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

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:  # Data가 1개 들어오면, 강제로 차원을 높임.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
         
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
              
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7)) / batch_size

mydata = np.genfromtxt('zoo.txt',delimiter=',',dtype=np.float32)

A = mydata[:, 0:-1]
B = mydata[:, -1].astype(int)   # mydata[:, [-1]]  == mydata[:, -1].reshape(-1,1)

N_Class = 7



def MultinomialClassification():
    learning_rate=1e-3
    N_Data = A.shape[0]
    N_Feature = A.shape[1]
    
    BB = change_ont_hot_label(B,N_Class)
    
    
    W = np.random.standard_normal(size=(N_Feature,N_Class))    


    total_cost = []
    for step in range(1000000):
        temp = (softmax(np.dot(A,W)) - BB) /N_Data
        W -= learning_rate * np.dot(A.T, temp)
        
        if step % 10000 == 0:
            total_cost.append(cross_entropy_error(softmax(np.dot(A,W)),B))
            print(step, total_cost[-1])

    prediction = np.argmax(softmax(np.dot(A,W)),axis=1)
    acc = np.mean(1*(prediction==B))
    
    plt.plot(total_cost)

    print("Accuracy: ", acc)
    print("W: ", W)

def MultinomialClassificationTF():
    learning_rate=1e-3

if __name__ == "__main__":
    start = time.time()
    print ((datetime.datetime.now()), " Start")
    MultinomialClassification()
    #MultinomialClassificationTF()
    
    
    print ((datetime.datetime.now()), " Finish")    
    finish = time.time()
    print (int((finish - start)/60.0), "Min", (finish - start)%60, "Sec elapsed") 
    
