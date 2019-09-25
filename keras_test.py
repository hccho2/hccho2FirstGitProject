# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K

def simple_ex():
    # mnist.load_data(): path를 지정하지 않으면, C:\Users\Administrator\.keras\datasets 에 받는다.  상대 path말고, 절대 path로 지정해야 됨
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='D:/hccho/keras-test/mnist.npz')
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    # 2. 모델 구성하기
    model = Sequential()
    model.add(Dense(units=64, input_dim=28*28, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    
    # 3. 모델 학습과정 설정하기
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # 4. 모델 학습시키기
    hist = model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    # 5. 학습과정 살펴보기
    print('## training loss and acc ##')
    print(hist.history['loss'])
    print(hist.history['acc'])
    
    # 6. 모델 평가하기
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print('## evaluation loss and_metrics ##')
    print(loss_and_metrics)
    
    # 7. 모델 사용하기
    xhat = x_test[0:1]
    yhat = model.predict(xhat)
    print('## yhat ##')
    print(yhat)



def simple2():
    model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
    model.compile(loss='mean_squared_error', optimizer='sgd')
    
    a = np.random.randn(5,3)
    b = np.random.randn(5,1)
    
    
    result = model.fit(a, b, epochs=5, batch_size=1)


def simple2():
    # 이 방식은 tensorflow 방식과 유사하다.. keras.layers.Input이 placeholder와 유사하다.
    a = np.random.randn(5,3)
    b = np.random.randn(5,1)
    
    
    X = Input(shape=(3,))
    my_layers = Sequential([Dense(100, input_shape=(3,), activation='relu'), Dense(1,)])
    L1 = my_layers.layers[0](X)
    Y = my_layers.layers[1](L1)
    
    
    model = Model(X,Y)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    print(model.input, model.output)
    
    model.fit(a, b, epochs=50, batch_size=6,verbose=1)
    
    sess = K.get_session()
    print(sess.run(model.output,feed_dict= {model.input: a}))  # sess.run(Y,feed_dict= {model.input: a})
    print(sess.run(Y,feed_dict= {X: a}))

simple2()








