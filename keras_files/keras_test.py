'''
optimizer: SGD(lr=0.01), SGD(lr=0.01, momentum=0.9), RMSprop(lr=0.001), Adam(lr=0.001)



keras.layers.RepeatVector(n)    (N,D) ---> (N,n,D)


https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
keras.layers.Attention: Dot-product attention layer, a.k.a. Luong-style attention



https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
keras.layers.TimeDistributed  video같이 image가 series 있을 때, (N,T,H,W,C) ---> Convolution으로 처리하고자 할 때.

'''


# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Input, Lambda
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils,plot_model
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
    # 참고: loss가 2개의 합으로 되어 있을 경우도 처리하는 방법이 있다.  https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    
    # 모델을 그림으로 출력
    plot_model(model, to_file='keras_model2.png', show_shapes = True)  # show_shapes=False --> shape 없이...
    
    
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
    with K.tf.device('/gpu:0'):  # with K.tf.device('/cpu:0')
        model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.summary()


        a = np.random.randn(5,3)
        b = np.random.randn(5,1)


        result = model.fit(a, b, epochs=5, batch_size=1)  # result.history

        
def simple3():
    # user-defined loss function
    def my_loss(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true-y_pred),axis=-1)  # batch에 대한 평균까지 계산할 필요가 없다.return shape: (batch_size,)

    model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    model.compile(loss=my_loss, optimizer='sgd')
    model.summary()


    a = np.random.randn(5,3)
    b = np.random.randn(5,1)


    result = model.fit(a, b, epochs=10, batch_size=1,verbose=1)  # result.history
    print(np.sum(np.square(model.predict(a)-b))/5)
        
def model_save_load():
    # 모델 구조, weights를 각각 저장
    def train():
        model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.summary()


        a = np.random.randn(5,3)
        b = np.random.randn(5,1)


        result = model.fit(a, b, epochs=5, batch_size=1)  # result.history



        model.save_weights('my_keras_model.h5')

        model_json = model.to_json()
        with open("my_keras_model.json", "w") as json_file : 
            json.dump(model_json, json_file)



    def infer():
        method=1
        if method ==2:
            model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
            model.load_weights('my_keras_model.h5')
        else:
            with open('my_keras_model.json','r') as f:
                model_json = json.load(f)
            model = model_from_json(model_json)


        model.load_weights('my_keras_model.h5')


        a = np.array([[-0.24941969, -1.05335905, -1.84161028],
                       [-0.53400826, -0.07559009,  1.03925354],
                       [ 0.28233044, -0.53535825, -1.2506007 ],
                       [-0.96030663,  0.50624464, -0.086618  ],
                       [ 0.06110731,  0.99453469, -0.34139146]])
        print(model.predict(a))
    
    #train()
    infer()

def model_save_load_advanced():
    # 모델 구조까지 h5파일에 모두 저장
    def train():
        model = Sequential([Dense(1, input_shape=(3,), activation='relu')])
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.summary()


        a = np.random.randn(10,3)
        b = np.random.randn(10,1)

        method = 2
        if method==1:
            logging = TensorBoard()
            checkpoint = ModelCheckpoint("my_keras_model_{epoch:04d}.h5", monitor='loss', save_weights_only=False, save_best_only=True,period=10) #monitor = 'val_loss'
            early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')


            result = model.fit(a, b, epochs=100, batch_size=1,callbacks=[logging, checkpoint, early_stopping])  # result.history        
        else:
            logging = TensorBoard()
            checkpoint = ModelCheckpoint("my_keras_model.h5", monitor='val_loss', save_weights_only=False, save_best_only=False)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

            result = model.fit(a, b, epochs=100, batch_size=2,validation_split=0.1,callbacks=[logging, checkpoint, early_stopping])  # result.history


    def infer():

        model= load_model('my_keras_model.h5')


        a = np.array([[-0.24941969, -1.05335905, -1.84161028],
                       [-0.53400826, -0.07559009,  1.03925354],
                       [ 0.28233044, -0.53535825, -1.2506007 ],
                       [-0.96030663,  0.50624464, -0.086618  ],
                       [ 0.06110731,  0.99453469, -0.34139146]])
        print(model.predict(a))


    infer()
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

    
def keras_layer_test():

    VOCAB_SIZE =20
    BATCH_SIZE = 3
    T = 10
    EMB_SIZE = 8
    
    x = np.random.randint(VOCAB_SIZE,size=(BATCH_SIZE,T))
    xx = tf.convert_to_tensor(x)
    embedding_layer = keras.layers.Embedding(VOCAB_SIZE,EMB_SIZE)(xx)
    
    conv1 = keras.layers.Conv1D(filters=128,kernel_size=3,padding='valid',activation=tf.nn.relu)(embedding_layer)
    
    pool1 = keras.layers.GlobalMaxPool1D()(conv1)# (3,128)
    
    pool2 = tf.keras.layers.MaxPool1D(conv1.shape.as_list()[1],1)(conv1) #(3,1,128)  
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    a1,a2 = sess.run([pool1,pool2])
    print(np.allclose(a1,np.squeeze(a2)))
    
    
    
if __name__ == '__main__':

    simple1()
    #simple2()
    #simple3()
    #simple4()







