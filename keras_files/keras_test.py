# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Input, Lambda
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
def simple1():
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
    model.summary()
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
    model.summary()
    
    
    a = np.random.randn(5,3)
    b = np.random.randn(5,1)
    
    
    result = model.fit(a, b, epochs=5, batch_size=1)


def simple3():
    # 이 방식은 tensorflow 방식과 유사하다.. keras.layers.Input이 placeholder와 유사하다.
    a = np.random.randn(5,3)
    b = np.random.randn(5,1)
    
    
    X = Input(shape=(3,))
    my_layers = Sequential([Dense(100, input_shape=(3,), activation='relu'), Dense(1,)])
    L1 = my_layers.layers[0](X)
    Y = my_layers.layers[1](L1)
    
    
    model = Model(X,Y)
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.001))
    model.summary()
    print("model.input, model.output: ", model.input, model.output)
    
    model.fit(a, b, epochs=50, batch_size=6,verbose=1)
    
    sess = K.get_session()
    print(sess.run(model.output,feed_dict= {model.input: a}))  # sess.run(Y,feed_dict= {model.input: a})
    print(sess.run(Y,feed_dict= {X: a}))
def simple4():

    def K_loss(args, weight):
        
        y_true, y_pred = args
        loss = weight * K.mean(K.square(y_pred-y_true))
        return loss

    
    
    
    # 이 방식은 tensorflow 방식과 유사하다.. keras.layers.Input이 placeholder와 유사하다.
    a = np.random.randn(5,3)
    b = np.random.randn(5,1)
    
    
    X = Input(shape=(3,))
    Y_true = Input(shape=(1,))
    my_layers = Sequential([Dense(100, input_shape=(3,), activation='relu'), Dense(1,)])
    L1 = my_layers.layers[0](X)
    Y = my_layers.layers[1](L1)
    
    
    
    
    
    model_loss = Lambda(K_loss, output_shape=(1, ),name='hccho_loss', arguments={'weight': 1.0})([Y_true,Y])  # [tensor, tensor, ...]
    
    model = Model([X,Y_true],model_loss)
    
    #  lmabda y_true, y_pred: y_pred 이 형식이 중요함.
    model.compile(loss={'hccho_loss': lambda y_true, y_pred: y_pred}, optimizer=optimizers.Adam(lr=0.001))   # model_loss의 [Y,Y_true] ~ lambda y_true, y_pred: y_pred형식상이지만, 순서 중요. 미분이 되게...)
    model.summary()
    print("model.input, model.output: ", model.input, model.output)
    
    model.fit([a,b], np.zeros(len(a)), epochs=50, batch_size=6,verbose=1)
    
    
    ############# inference ###########################
    sess = K.get_session()
    # model.output은 model_loss 값이다.
    print(sess.run(model.output,feed_dict= {model.input[0]: a,model.input[1]: b}))  # sess.run(Y,feed_dict= {model.input: a})
    
    print('target: ', b)
    print('prediction: ',sess.run(Y,feed_dict= {X: a}))
if __name__ == '__main__':

    #simple1()
    #simple2()
    #simple3()
    simple4()








