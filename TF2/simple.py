# coding: utf-8


'''
2020년5월19일 현재: tensorflow-gpu 2.2, 2.1 설치해도 error. 2.0.2는 error 안남.
버전 2.2 설치 해결책: https://github.com/tensorflow/tensorflow/issues/35618#issuecomment-596631286   <-- 여기 참고.
        latest microsoft visual c++ redistributable 설치하면, 해결된다.

1.x 에서의 contrib가 SIG Addons로 갔다. SIG(special Interest Group)   --> pip install tensorflow-addons
https://www.tensorflow.org/addons/api_docs/python/tfa


import tensorflow as tf
import tensorflow_addons as tfa     ---> tfa.seq2seq.BahdanauMonotonicAttention 이런 것이 있다.


https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko




모델 저장
https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
https://www.tensorflow.org/guide/saved_model?hl=ko

방법 1: tf.saved_model.save  ----> tf.saved_model.load
밥법2: checkpoint 파일로 저장 (2가지 방법)
     1. model.save_weights  ---> model.load_weights
     2. tf.train.Checkpoint를 이용하는 방법

'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import Constant
print(tf.__version__)
print('gpu available?', tf.test.is_gpu_available())



def embeddidng_test():
    embedding_dim =5
    vocab_size =3
    
    
    init = np.random.randn(vocab_size,embedding_dim)
    print('init: ',init)
    #embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True,name='my_embedding') 
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    print('embedding.trainable_variables', embedding.trainable_variables)
    
    
    input = np.array([[1,0,2,2,0,1],[1,1,1,2,2,0]])
    
    output = embedding(input)
    
    
    
    
    print('='*10)
    print(input,output)
    print('done')
    
    
    model = tf.keras.Sequential()
    model.add(embedding)
    print('trainable: ',model.trainable_variables)





def simple_model():


    X_train = np.arange(10).reshape((10, 1))
    y_train = np.array([1.0, 1.3, 3.1,2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

    class TfLinreg(object):
        
        def __init__(self, learning_rate=0.01):
            ## 가중치와 절편을 정의합니다
            self.w = tf.Variable(tf.zeros(shape=(1)))
            self.b = tf.Variable(tf.zeros(shape=(1)))
            ## 경사 하강법 옵티마이저를 설정합니다.
            self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
            
        def fit(self, X, y, num_epochs=10):
            ## 비용 함수의 값을 저장하기 위한 리스트를 정의합니다.
            training_costs = []
            for step in range(num_epochs):
                ## 자동 미분을 위해 연산 과정을 기록합니다.
                with tf.GradientTape() as tape:
                    z_net = self.w * X + self.b
                    z_net = tf.reshape(z_net, [-1])
                    sqr_errors = tf.square(y - z_net)
                    mean_cost = tf.reduce_mean(sqr_errors)
                    
                    
                ## 비용 함수에 대한 가중치의 그래디언트를 계산합니다.
                grads = tape.gradient(mean_cost, [self.w, self.b])
                ## 옵티마이저에 그래디언트를 반영합니다.
                self.optimizer.apply_gradients(zip(grads, [self.w, self.b]))
                ## 비용 함수의 값을 저장합니다.
                training_costs.append(mean_cost.numpy())
            return training_costs
        
        def predict(self, X):
            return self.w * X + self.b
    
    
    
    model = TfLinreg()
    training_costs = model.fit(X_train, y_train)
    print("w: ", model.w, "b: ", model.b)
    
    plt.plot(range(1,len(training_costs) + 1), training_costs)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.tight_layout()
    plt.show()
    
    plt.scatter(X_train, y_train, marker='s', s=50,label='Training Data')
    plt.plot(range(X_train.shape[0]),  model.predict(X_train),color='gray', marker='o', markersize=6, linewidth=3,label='LinReg Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()



def keras_standard_model():
    batch_size = 2
    input_dim = 3
    model = tf.keras.models.Sequential()
    
    
    model.add(tf.keras.layers.Dense(units=10,input_dim=3,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1,activation=None))
    
    print(model.summary())
    
    
    
    #X = tf.random.normal(shape=(batch_size, input_dim))
    #Y = tf.random.normal(shape=(batch_size, 1))
    
    X = tf.convert_to_tensor(np.array([[1.4358643,  1.275539,  -1.8608146 ], [-0.3436857, -0.7065693, -1.1548917]]),dtype=tf.float32)
    Y = tf.convert_to_tensor(np.array([[-1.4839303 ], [0.88788706]]),dtype=tf.float32)
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse')
    
    model.fit(X,Y,epochs=100,verbose=1)
    
    
    print(X,Y)
    print(model.predict(X))
    
    #tf.saved_model.save(model,'./saved_model')   # ----> model_load_test()
    
    save_method=2
    if save_method==1:
        model.save_weights('./saved_model/model_ckpt')   # train하지 않은 모델을 restore하기 때문에  몇가지 WARNING이 나온다. WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer
    else:
        checkpoint = tf.train.Checkpoint(model=model)   # tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint.save('./saved_model/model_ckpt')   # model_ckpt-1로 저장된다.
    
    print(model.weights)


def model_load_test():
    # tf.saved_model.save() 로 저장딘 것 복원.
    # https://www.tensorflow.org/api_docs/python/tf/saved_model/load
    '''
    .fit, .predict는 없다.
    .variables, .trainable_variables    .__call__    가능함.
    '''
    model = tf.saved_model.load('./saved_model')
    
    print(model)
    batch_size = 2
    input_dim = 3
    #X = tf.random.normal(shape=(batch_size, input_dim))
    X = tf.convert_to_tensor(np.array([[-0.03935467, -1.461705,   -1.4099646 ], [-0.20841599,  0.47920665, -0.44796956]]),dtype=tf.float32)
    print(model(X))
    
def model_load_checkpoint():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10,input_dim=3,activation='relu'))  # input_dim을 넣어주면, weight를 미리 생성한다.
    model.add(tf.keras.layers.Dense(units=1,activation=None))
    
    print(model.summary())
    
    #print('before:', model.weights)
    
    save_method=2
    if save_method==1:
        model.load_weights('./saved_model/model_ckpt')
    else:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore('./saved_model/model_ckpt-1')
    #print('after: ', model.weights)
    
    
    X = tf.convert_to_tensor(np.array([[1.4358643,  1.275539,  -1.8608146 ], [-0.3436857, -0.7065693, -1.1548917]]),dtype=tf.float32)
    Y = tf.convert_to_tensor(np.array([[-1.4839303 ], [0.88788706]]),dtype=tf.float32)
    
    print('target: ', Y.numpy())
    print('predict: ', model.predict(X))
    
def keras_standard_model2():
    batch_size = 2
    input_dim = 3
    
    inputs = tf.keras.Input(shape=(input_dim,))  # 구제적인 입력 data없이 ---> placeholder같은 ...
    
    L1 = tf.keras.layers.Dense(units=10,input_dim=3,activation='relu')
    L2 = tf.keras.layers.Dense(units=1,activation=None)
    
    output = L2(L1(inputs))
    
    model = tf.keras.Model(inputs,output)
    print(model.summary())
    
    
    X = tf.random.normal(shape=(batch_size, input_dim))
    
    Y = tf.random.normal(shape=(batch_size, 1))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse')
    
    model.fit(X,Y,epochs=100,verbose=1)
    
    
    print(X,Y)
    print(model.predict(X))


if __name__ == "__main__":    
    #embeddidng_test()
    #simple_model()
    #keras_standard_model()   # ---> model_load_test
    #model_load_test()
    model_load_checkpoint()
    #keras_standard_model2()




