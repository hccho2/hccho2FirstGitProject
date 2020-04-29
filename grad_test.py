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

def test2():  # 선형. 행렬곱(중간변수 1개)
    X = np.array([[1,2,3],[2,0,3]]).astype(np.float32)
    Y = np.array([[5],[6]]).astype(np.float32)
    W = tf.Variable(tf.random_normal([3,4]),name='Weight')
    Z = tf.matmul(X , W)
    f = tf.reduce_sum(Z,axis=-1,keepdims=True) # keepdims가 없어도 되려면, Y의 shape이 바뀌어야 한다.
    L = tf.reduce_mean(tf.square(f - Y))
    grad = tf.gradients(L,W)[0]
    grad1 = tf.gradients(L,Z)[0] # ---> list: grad1이 list라서 [0]
    grad2 = tf.gradients(Z,W,grad1)[0] # grad와 같은 값.
    manual_grad = tf.matmul(X.T,grad1)



    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(L)
    ap = optimizer.apply_gradients(zip([grad],[W]))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # grad1은 dL/dy Z-Y을 나열한 것과 같다.
    print('grad1:\n', sess.run(grad1), '\nf-Y\n: ', sess.run(f)-Y)
    print('\n\n')

    # 전체 graient갑
    print('모두 동일:', sess.run([grad, grad2, manual_grad]))

    print('Before: ',sess.run(W))
    print('직접계산:', sess.run(W-grad[0]*0.01))
    sess.run(ap)
    print('After: ',sess.run(W))
    
    
    
    for i in range(2):
        sess.run(train)

        if i%100 ==0:
            print(i,sess.run(cost), sess.run(W))
    
    
def test3():  # 중간 변수가 2개(y1,y2)
    x_train = np.array([1,2,3]).reshape(3,-1).astype(np.float32)
    y_train = np.array([5,4,3]).reshape(3,-1).astype(np.float32)

    W = tf.Variable(tf.random_normal([1,1]),name='Weight')
    b = tf.Variable(tf.random_normal([1]),name='bias')

    y1 = tf.matmul(x_train,W) + b
    y2 = tf.matmul(x_train,W) + b

    hypothesis = 6*y1 + y2

    cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    train = optimizer.minimize(cost)


    grad = tf.gradients(cost,[W,b])
    grad1 = tf.gradients(cost,[y1,y2])  # ---> y1,y2 두개이므로, 길이 2짜리 list가 생성. [y1 shape, y2 shape]. y1 shape은 (N,1)
    grad2 = tf.gradients([y1,y2],[W,b],grad1)  # grad와 같은 값.

    manual_grad = tf.matmul(x_train.T, grad1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(400):
        sess.run(train)

        if i%100 ==0:
            print(i,sess.run(cost), sess.run(W), sess.run(b))


    print('-'*10)
    print('grad:', sess.run(grad))
    print('='*10)
    print('grad2:', sess.run(grad2))

    print('manual_grad for W:', np.sum(sess.run(manual_grad)))
    print('manual_grad for b:', np.sum(sess.run(grad1)))

            

