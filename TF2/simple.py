
'''
2020년5월19일 현재: tensorflow-gpu 2.2, 2.1 설치해도 error. 2.0.2는 error 안남.
버전 2.2 설치 해결책: https://github.com/tensorflow/tensorflow/issues/35618#issuecomment-596631286   <-- 여기 참고.
        latest microsoft visual c++ redistributable 설치하면, 해결된다.

1.x 에서의 contrib가 SIG Addons로 갔다. SIG(special Interest Group)   --> pip install tensorflow-addons
https://www.tensorflow.org/addons/api_docs/python/tfa


import tensorflow as tf
import tensorflow_addons as tfa     ---> tfa.seq2seq.BahdanauMonotonicAttention 이런 것이 있다.


https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko


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

if __name__ == "__main__":    
    #embeddidng_test()
    simple_model()






