'''
optimizer: SGD(lr=0.01), SGD(lr=0.01, momentum=0.9), RMSprop(lr=0.001), Adam(lr=0.001)



'''


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
    model.add(Dense(units=10, activation='softmax'))  # model.add(Dense(units=10,input_dim=28*28, activation='softmax')) <--- input_dim이 잘못 들어가면 무시함.
    
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

    # 8. weights 보기
    print(model.get_weights())  # model.set_weights( weights ) 로 weight update도 가능

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

    
 def K_function_test():
    x1 = K.placeholder(shape=[None, 3])
    x2 = K.placeholder(shape=[None, 3])
    y = x1+x2
    
    train = K.function([x1,x2], [y])  # tensor x1,x2,y의 관계를 만들어 놓으면, x1,x2 자리에 data를 넣으면, y값이 계산된다.
    
    data1 = np.random.randn(2,3)
    data2 = np.random.randn(2,3)
    zz = train([data1,data2])
    
    print('data1: ', data1 )
    print('data2: ', data2 )
    print('zz: ', zz )   
    
def K_function_train():
    # K.function을 이용해서, training을 할 수 있다.
    from keras import backend as K
    from keras.layers.core import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    y =  K.placeholder(shape=[None, 1])
    model = Sequential()
    model.add(Dense(24, input_dim=3, activation='relu'))
    model.add(Dense(1))
    loss = K.mean( K.square(model.output - y))
    
    optimizer = Adam(lr=0.001)
    updates = optimizer.get_updates(model.trainable_weights,[],loss)
    
    
    train = K.function([model.input,y], [model.output,loss],updates = updates)
    
    
    # train 해보기
    data = np.random.randn(2,3)
    target = np.random.randn(2,1)
    output = model.predict(data)
    
    for i in range(1000):
        temp = train([data,target])
        if i%100 == 0:
            print(i,temp)
    
    # 결과 확인
    print('data', data)
    print('target', target)
    print('predict', model.predict(data))
    
if __name__ == '__main__':

    #simple1()
    #simple2()
    #simple3()
    simple4()








