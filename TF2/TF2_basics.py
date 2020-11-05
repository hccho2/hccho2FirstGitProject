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


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)



'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
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
    plt.tight_layout()  # 지동으로 layout 조정.
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
    
    
    data_mode = 2
    if data_mode == 1:
        X = tf.random.normal(shape=(10, input_dim))
        Y = tf.random.normal(shape=(10, 1))
        
        #X = tf.convert_to_tensor(np.array([[1.4358643,  1.275539,  -1.8608146 ], [-0.3436857, -0.7065693, -1.1548917]]),dtype=tf.float32)
        #Y = tf.convert_to_tensor(np.array([[-1.4839303 ], [0.88788706]]),dtype=tf.float32)
        
        #X = np.array([[1.4358643,  1.275539,  -1.8608146 ], [-0.3436857, -0.7065693, -1.1548917]])
        #Y = np.array([[-1.4839303 ], [0.88788706]])
    else:
        X = tf.random.normal(shape=(10, input_dim))
        Y = tf.random.normal(shape=(10, 1))        
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))  # 여기의 argument가 mapping_fn의 argument가 된다.
        dataset = dataset.shuffle(buffer_size=batch_size*10).repeat() # 반복회수를 지정하지 않으면 무한반복
        dataset = dataset.batch(batch_size,drop_remainder=False)
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        validation_dataset = validation_dataset.batch(len(X),drop_remainder=False)
        
        
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse')
    
    if data_mode == 1:
        # data를 X,Y를 batch_size로 나누어서 epochs만큼 train
        #model.fit(X,Y,batch_size=batch_size, epochs=5,verbose=1) # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
        
        model.fit(X,Y,batch_size=batch_size, epochs=5,verbose=1,validation_split=0.1)
    else:
        # dataset 자체에 batch_size가 정해져 있기 때문에, 몇 step을 1epoch으로 볼 것인가 지정(steps_per_epoch)
        # data가 부족하면, 지정한 epoch을 다 채우지 못한채 끝낸다.
        # tf.data.Dataset오로 data를 주면, validation_split이 작동하지 않는다.
        #model.fit(dataset, epochs=5,verbose=1, steps_per_epoch = 10)
        
        
        history = model.fit(dataset, epochs=5,verbose=1, steps_per_epoch = 10,validation_data=validation_dataset,validation_freq=1)  # validation_freq는 epoch단위
        plt.plot(history.history['loss'],label="train loss")
        plt.plot(history.history['val_loss'],label="val loss")
        plt.legend()  # plt.legend(loc="upper right")
        plt.show()
    
    
    print(X,Y)
    print(model.predict(X))
    
    #tf.saved_model.save(model,'./saved_model')   # ----> model_load_test()
    
    model_dir = './saved_model'
    model_dir_preface = './saved_model/model_ckpt'   # 저장할 때는 디렉토리 + preface
    save_method=3
    if save_method==1:
        # model로 부터 바로 저장
        model.save_weights(model_dir_preface)   # train하지 않은 모델을 restore하기 때문에  몇가지 WARNING이 나온다. WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer
    elif save_method==2:
        # tf.train.Checkpoint를 사용하여 저장
        # tf.train.Checkpoint(model=model), tf.train.Checkpoint(net=model), tf.train.Checkpoint(mymodel=model) ... argument이름은 임의로 설정가능.
        checkpoint = tf.train.Checkpoint(model=model)   # tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint.save(model_dir_preface)   # model_ckpt-1로 저장된다.    
    else:
        # tf.train.Checkpoint --> tf.train.CheckpointManager를 이용하여 저장.
        checkpoint = tf.train.Checkpoint(model=model)   # tf.train.Checkpoint(optimizer=optimizer, model=model)
        chekpoint_manager = tf.train.CheckpointManager(checkpoint, model_dir,checkpoint_name = 'model_ckpt', max_to_keep=5)   # preface없이 모델 dir만 넣어준다.
        ckpt_save_path = chekpoint_manager.save()
        print('model saved: ', ckpt_save_path)
    
    
    print(model.weights)


def model_load_test():
    # tf.saved_model.save() 로 저장된 것 복원.
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
    
    
    model_dir = './saved_model'
    
    save_method=1
    if save_method==1:
        model.load_weights(model_dir+'/model_ckpt')
    elif save_method==2:
        checkpoint = tf.train.Checkpoint(model=model)
        chekpoint_manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=5)   # preface없이 모델 dir만 넣어준다.
        checkpoint.restore(chekpoint_manager.latest_checkpoint)  # checkpoint.restore('./saved_model/model_ckpt-1')
        
        
    #print('after: ', model.weights)
    
    
    X = tf.convert_to_tensor(np.array([[1.4358643,  1.275539,  -1.8608146 ], [-0.3436857, -0.7065693, -1.1548917]]),dtype=tf.float32)
    Y = tf.convert_to_tensor(np.array([[-1.4839303 ], [0.88788706]]),dtype=tf.float32)
    
    print('target: ', Y.numpy())
    print('predict: ', model.predict(X))
    
def keras_standard_model2():  # tf.keras.Input 사용
    # functional API라 부른다. <----> sequential API(tf.keras.Sequential or tf.keras.models.Sequential)
    batch_size = 2
    input_dim = 3
    
    inputs = tf.keras.Input(shape=(input_dim,))  # 구제적인 입력 data없이 ---> placeholder같은 ...
    
    L1 = tf.keras.layers.Dense(units=10,input_dim=3,activation='relu')
    L2 = tf.keras.layers.Dense(units=1,activation=None)
    
    outputs = L2(L1(inputs))
    
    model = tf.keras.Model(inputs = inputs,outputs = outputs)
    print(model.summary())
    
    
    X = tf.random.normal(shape=(batch_size, input_dim))
    
    Y = tf.random.normal(shape=(batch_size, 1))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse')
    
    history = model.fit(X,Y,epochs=100,verbose=1)
    
    
    print(X,Y)
    print(model.predict(X))

def keras_standard_model3():
    # https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
    # Loss Function customization  ----> Loss function을 만들어 넘기는 과정이 좀 거지 같다....
    import tensorflow.keras.backend as K

    def loss_fn(y_true,y_pred):
        return  K.mean(K.square(y_pred-y_true))


    batch_size = 2
    input_dim = 3
    
    inputs1 = tf.keras.Input(shape=(input_dim,))  # shape: batch_size제외 ---> (None,input_dim)
    inputs2 = tf.keras.Input(shape=(input_dim,))
    Y_true = tf.keras.Input(shape=(1,))
    
    
    inputs = tf.concat([inputs1,inputs2],axis=-1)
    
    L1 = tf.keras.layers.Dense(units=10,input_dim=3,activation='relu')
    L2 = tf.keras.layers.Dense(units=1,activation=None)
    
    outputs = L2(L1(inputs))
    
    model = tf.keras.Model([inputs1,inputs2],outputs)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer = optimizer,loss=loss_fn)
    
    print(model.summary())
    
    
    X1 = tf.random.normal(shape=(batch_size, input_dim))
    X2 = tf.random.normal(shape=(batch_size, input_dim))
      
    Y = tf.random.normal(shape=(batch_size, 1))

    model.fit([X1,X2],Y,epochs=100,verbose=1)
      
      
    print(X1,X2,Y)
    print(model.predict([X1,X2]))



def mode_test():
    # 
    mode = 2
    if mode == 1:
        class MyModel(tf.keras.Model):
            def __init__(self):
                super(MyModel, self).__init__()
                
                self.dropout = tf.keras.layers.Dropout(0.5)
                self.dense = tf.keras.layers.Dense(1,name='hccho')
            def call(self,x,training=True):
                # tf.keras.Model을 통해서 간접적으로 call 될 때는, training=None이 들어온다. ===> training= K.learning_phase()가 내부적으로 적용된다.
                return self.dense(self.dropout(x, training = training))


        model = MyModel()
    else:
        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=1))
    
    
    batch_size=2
    input_dim=3
    x = tf.random.normal(shape=(batch_size,input_dim))
    y = tf.random.normal(shape=(batch_size,1))
    
    y1 = model(x,True)
    y2 = model(x,False)
    
    print("dropout on: ", y1,'\ndropout off: ', y2)
    print('manual cal: ', np.matmul(x,model.get_weights()[0]) + model.get_weights()[1])
    print('loss: ', np.square(np.subtract(y, y1)).mean(), np.square(np.subtract(y, y2)).mean())
    
    
    print('='*20)
    X = tf.keras.Input(shape=(input_dim,),dtype=tf.float32)
    Y1 = model(X,True)  # train mode용
    Y2 = model(X) # eval mode용 ----> call의 training에는 default값이 들어간다.  ----> 아래에 있는 Model을 통한 구조에서는 None이 들어간다.
    
    
    
    
    model1 = tf.keras.Model(X,Y1)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model1.compile(optimizer,loss='mse')
    
    model2 =  tf.keras.Model(X,Y2)
    model2.compile(optimizer,loss='mse')

    
    
    
    print('train mode: ', model1(x))  
    print('eval mode: ',model1.evaluate(x,y))  # training=True로 설정되어 있기 때문에, evaluate에서도 training=True가 적용된다. dropout이 random하기 때문에 위에서 계산한 loss와 다르다.
    
    print('='*40)
    
    print('train mode: ', model2(x)) # training=None이 들어간다.  ---> K.learning_phase() = 0이다  ----> dropout=off
    print('eval mode: ',model2.evaluate(x,y))  # training=False가 적용되어 있기 때문에 위에서 계산한 loss값과 일치한다.


def load_data():
    # tf.keras.mnist, cifar10, cifar100, imdb, reuters, boston_housing
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # C:\Users\MarketPoint\.keras\datasets\mnist.npz  (11M)
    
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)   # numpy array - uint8, (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255    

if __name__ == "__main__":    
    #embeddidng_test()
    #simple_model()
    #keras_standard_model()   # ---> model_load_test
    
    #model_load_test()
    #model_load_checkpoint()
    #keras_standard_model2()
    #keras_standard_model3()
    #mode_test()

    load_data()
