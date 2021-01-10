# coding: utf-8

'''

https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator?hl=ko
https://stackoverflow.com/questions/49140164/tensorflow-error-unsupported-callable


# metric은 train중에 관측하는 개념이 아니고, evaluate에서 사용하는 개념. tf.estimator.EstimatorSpec에서 eval_metric_ops를 넣어 주는 것과 동일.
# evaluate에서는 tf.keras.backend.learning_phase() = False이다.
# input_fn은 argument가 없는 함수로 만들어야 한다. argument가 있는 경우는, lambda function으로 변환해 주어야 한다.


('C:\\Anaconda3\\envs\\tf2\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv',
 array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
 ['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)'])


'''


import numpy as np
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf_compat
print(f'tensorflow version: {tf.__version__}')
import matplotlib.pyplot as plt
from sklearn import datasets
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)   # hooks에서 정보가 출력되기 위해서 필요.

def test1():
    batch_size = 2
    input_dim = 3
    
    
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=10,input_dim=3,activation='relu',name='L1'),
                                        tf.keras.layers.Dense(units=1,activation=None,name='L2')])
    
    
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.InputLayer(input_shape=(3,), name='dense_input'))   # name과 dataset의 dict key가 일치해야 한다.
    # model.add(tf.keras.layers.Dense(units=10,activation='relu',name='L1'))
    # model.add(tf.keras.layers.Dense(units=1,activation=None,name='L2'))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse',metrics=['mse'])
    print(model.summary())
    
    
    model_dir= './estimator_model'   # 없으면 만들어 준다   ==> 디렉토리이름/keras/ 아래에, checkpoint, keras_model.ckpt.data-00000-of-00001, keras_model.ckpt.index
    

    keras_estimator = tf.keras.estimator.model_to_estimator( keras_model=model, model_dir=model_dir)
    
    print(keras_estimator)
    
    def input_fn():
        X = tf.random.normal(shape=(10, input_dim))
        Y = tf.random.normal(shape=(10, 1))        
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))  # 여기의 argument가 mapping_fn의 argument가 된다.
        #dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))    
        dataset = dataset.shuffle(buffer_size=batch_size*10).repeat() # 반복회수를 지정하지 않으면 무한반복
        dataset = dataset.batch(batch_size,drop_remainder=False)    
        return dataset
    
    
    # for features_batch, labels_batch in input_fn().take(2):
    #     print(features_batch)
    #     print(labels_batch)
    
    keras_estimator.train(input_fn=input_fn, steps=1000)


def test2():

    batch_size = 2
    input_dim = 3
    
    
    def model_fn(features, labels, mode):
    
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=10,input_dim=3,activation='relu',name='L1'),
                                            tf.keras.layers.Dense(units=1,activation=None,name='L2')])
    
    
    
        opt = tf.keras.optimizers.Adam(0.01)
        ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(), optimizer=opt, net=model)
        with tf.GradientTape() as tape:
            output = model(features)
            loss = tf.reduce_mean(tf.square(output - labels))
            
        gradients = tape.gradient(loss, model.trainable_variables)
        
        train_op = tf.group(opt.apply_gradients(zip(gradients, model.trainable_variables)), ckpt.step.assign_add(1))
        
        logging_hook = tf.estimator.LoggingTensorHook({"loss----" : loss }, every_n_iter=2)  # dict key값을 정렬 순으로 출력한다.
        
        return tf.estimator.EstimatorSpec(mode,loss=loss, train_op=train_op,scaffold=tf_compat.train.Scaffold(saver=ckpt),training_hooks = [logging_hook])
    
    
    def input_fn():
        X = tf.random.normal(shape=(10, input_dim))
        Y = tf.random.normal(shape=(10, 1))        
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))  # 여기의 argument가 mapping_fn의 argument가 된다.
        dataset = dataset.shuffle(buffer_size=batch_size*10).repeat() # 반복회수를 지정하지 않으면 무한반복
        dataset = dataset.batch(batch_size,drop_remainder=False)
        
        return dataset
        
          
    model_dir= './estimator_model'   # 없으면 만들어 준다   ==> 디렉토리이름/keras/ 아래에, checkpoint, keras_model.ckpt.data-00000-of-00001, keras_model.ckpt.index
    
    
    tf.keras.backend.clear_session()
    est = tf.estimator.Estimator(model_fn, model_dir)
    est.train(input_fn, steps=10)   # 저장된 steps에 이어서 추가적으로 더 train


def test3():
    # random data를 fitting하는 것이므로, batch_size가 커야된다.
    batch_size = 10
    input_dim = 3
    output_dim = 3   # classification 문제.
    
    
    def model_fn(features, labels, mode):
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=50,input_dim=3,activation='relu',name='L1'),
                                            tf.keras.layers.Dense(units=output_dim,activation=None,name='L2')])    
        
        #model = tf.keras.models.Sequential([tf.keras.layers.Dropout(0.5)])
        logits = model(features)  # dropout의 training=True/False는 default값으로 결정된다.
        predictions = {'logits': logits}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
        # cross entropy loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)


        if mode == tf.estimator.ModeKeys.EVAL:   # dropout 적용 안됨.  K.learning_phase() = False로 적용됨.
            # metric
            # accuracy <---- metric으로 정의되어야 한다. 단순 op로 넘기려면, hook으로 넘겨야 한다.
            acc1 = tf.compat.v1.metrics.accuracy(labels,tf.argmax(logits,axis=-1))
            
            auc_metric = tf.keras.metrics.Accuracy()
            auc_metric.update_state(y_true=labels, y_pred=tf.argmax(predictions['logits'],axis=-1)) 
            
            metrics= {'my acc': acc1, 'my acc2': auc_metric}  # metric은 누적으로 값이 계산된다.
            
            # hook <--- op를 넘긴다.
            eval_logging_hook = tf.estimator.LoggingTensorHook({"eval-----my logits" : logits, "eval2----my labels": labels,
                                                                "training phase": tf.keras.backend.learning_phase()}, every_n_iter=2)
            
            # loss의 평균값, metric의 평균값을 계산해 준다.  predictions를 넘겨 주어야, add_metric에서 사용할 수 있다.
            return tf.estimator.EstimatorSpec(mode, loss=loss,predictions=predictions,eval_metric_ops=metrics,evaluation_hooks=[eval_logging_hook])

        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdadeltaOptimizer (learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step)  # var_list=model.trainable_variables
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels,tf.int64),tf.argmax(logits,axis=-1)),tf.float32))
        logging_hook = tf.estimator.LoggingTensorHook({"step": global_step, "loss----" : loss,'my acc': acc}, every_n_iter=50)  # dict key값을 정렬 순으로 출력한다.
        

        return tf.estimator.EstimatorSpec(mode,loss=loss, train_op=train_op,training_hooks = [logging_hook])
    
    
    def input_fn(X=None,Y=None,training=True,repeat=None):
        N = 10 # data size
        if X is None:
            X = tf.random.normal(shape=(N, input_dim))
        
        if training and Y is None:
            #Y = tf.random.normal(shape=(N, 1))
            Y = tf.random.uniform(shape=(N,),maxval=output_dim,dtype=tf.int32)      
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))  # 여기의 argument가 mapping_fn의 argument가 된다.
        if repeat is None:
            dataset = dataset.shuffle(buffer_size=batch_size*10).repeat() # 반복회수를 지정하지 않으면 무한반복
        else: 
            dataset = dataset.repeat(repeat)

        dataset = dataset.batch(batch_size,drop_remainder=False)
        
        return dataset

    myX = np.array([[ 0.66848207, -0.4954776,  -0.56775373],[-0.9689017,   2.7078648,  -1.4647455 ],
                    [-1.026334,    1.1309948,   1.6681341 ], [-0.18581304,  1.2994535,  -0.33380032]])
    myY = np.array([1,2,1,0])   # np.array([2,1,1,0])   np.array([0,1,1,0])
    
    
#     for x,y in input_fn().take(5):   # input_fn(myX,training=False,repeat=1).take(5)
#     #for x,y in input_fn(myX,training=False,repeat=1).take(5):
#         print(x,y)
#         print('='*5)


          
    model_dir= './estimator_model'   # 없으면 만들어 준다   ==> 디렉토리이름/keras/ 아래에, checkpoint, keras_model.ckpt.data-00000-of-00001, keras_model.ckpt.index

    tf.keras.backend.clear_session()
    my_config =tf.estimator.RunConfig(log_step_count_steps=500,save_summary_steps=500,save_checkpoints_steps=1000)
    est = tf.estimator.Estimator(model_fn, model_dir,config = my_config)
    
    
    # add_metric은 evaluaiton에서만 작동. 
    # tf.estimator.EstimatorSpec에서 eval_metric_ops를 넣어 주는 것과 동일.
    # 아래에서 정의하는 metric_fn의 predictions는 tf.estimator.EstimatorSpec에서 predictions를 넘겨주어야 된다.
#     def metric_fn(labels,predictions):
#         acc = tf.compat.v1.metrics.mean(tf.cast(tf.equal(tf.cast(labels,tf.int64),tf.argmax(predictions['logits'],axis=-1)),tf.float32))
#         return {'myyyyy acc': acc}

    def metric_fn(labels, predictions):
        auc_metric = tf.keras.metrics.Accuracy(name="my_auc")
        auc_metric.update_state(y_true=labels, y_pred=tf.argmax(predictions['logits'],axis=-1))
        return {'myyyyy acc': auc_metric}



    est = tf.estimator.add_metrics(est, metric_fn)
      
    
    if True:
        est.train(input_fn, steps=5000)   # 저장된 steps에 이어서 추가적으로 더 train

    if False:
        print('2: --------- evaluation ---------')
        eval_result = est.evaluate(input_fn = lambda : input_fn(myX,myY,training=False,repeat=2),steps=2)
        print(eval_result)


    if False:
  
        print('3: --------- prediction ---------')
        ####  predictions = est.predict(input_fn(myX,myY,training=False,repeat=1))  # <========== 이렇게 하면 안된다. 아래와 같이 lambda function으로 
        predictions = est.predict(lambda : input_fn(myX,myY,training=False,repeat=2))
        for p in predictions:
            print(p['logits'])
      
        print('*'*10)
      
        input_fn_predict = tf.compat.v1.estimator.inputs.numpy_input_fn(x = myX,batch_size=2,shuffle=False)
        predictions = est.predict(input_fn_predict)
        for p in predictions:
            print(f"logit: {p['logits']} ----> {np.argmax(p['logits'])}   ")  
  
  
    print('done')


def test4():
    '''
        1. iris dataset으로 분류
        2. init_from_checkpoint 사용해 보기.
    
    ''' 
    batch_size = 64
    input_dim=4
    output_dim = 3
    
    def model_fn(features, labels, mode):
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=10,input_dim=input_dim,activation='relu',name='L1'),
                                            tf.keras.layers.Dense(units=10,activation='relu',name='L2'),
                                            tf.keras.layers.Dense(units=output_dim,activation=None,name='L3')])    

        logits = model(features)  # dropout의 training=True/False는 default값으로 결정된다.
        predictions = {'logits': logits, 'prob': tf.nn.softmax(logits,axis=-1)}
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
        # cross entropy loss
        #  labels: (N,)  ===> onehot으로 변환해야 됨.
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(labels,output_dim), logits, from_logits=True, label_smoothing=0))


        if mode == tf.estimator.ModeKeys.EVAL:   # dropout 적용 안됨.  K.learning_phase() = False로 적용됨.
            auc_metric = tf.keras.metrics.Accuracy()
            auc_metric.update_state(y_true=labels, y_pred=tf.argmax(predictions['logits'],axis=-1)) 
            
            metrics= {'accuracy': auc_metric}  # metric은 누적으로 값이 계산된다.
            
            # loss의 평균값, metric의 평균값을 계산해 준다.  predictions를 넘겨 주어야, add_metric에서 사용할 수 있다.
            return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=metrics,predictions=predictions,)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step)  # var_list=model.trainable_variables
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels,tf.int64),tf.argmax(logits,axis=-1)),tf.float32))
        logging_hook = tf.estimator.LoggingTensorHook({"step": global_step, "loss----" : loss,'my acc': acc}, every_n_iter=10)

        init_checkpoint=False
        if init_checkpoint:
            # 여기서 넣어주는 checkpoint가 먼저 load되고, 이후에 model_dir에 있는 checkpoint가 load된다.
            # 따라서 model_dir에는 checkpoint가 없어야 의미가 있다.
            # 사실 이 기능은 Fine-Tuning 모델을 train하기 전에, pre-trained model의 weight를 load할 때 적합한 기능이다.
            
            assignment_map = {v.name.split(':')[0]: v for v in model.trainable_weights}
            tf.compat.v1.train.init_from_checkpoint('./estimator_model2/pre-trained/model.ckpt-1000',assignment_map)        
        
        
        return tf.estimator.EstimatorSpec(mode,loss=loss, train_op=train_op,training_hooks = [logging_hook])
    
    
    def input_fn(repeat=None):
        iris = datasets.load_iris()
        dataset = tf.data.Dataset.from_tensor_slices((iris['data'], iris['target']))
        if repeat is None:
            dataset = dataset.shuffle(buffer_size=batch_size*10).repeat()
        else:
            dataset = dataset.repeat(repeat)
        
        dataset = dataset.batch(batch_size,drop_remainder=False)
        return dataset


    # Data Load Test
    if False:
        for x,y in input_fn().take(1):
            print(x,y)



    model_dir= './estimator_model2'
 
    tf.keras.backend.clear_session()
    my_config =tf.estimator.RunConfig(log_step_count_steps=500,save_summary_steps=500,save_checkpoints_steps=1000)
    est = tf.estimator.Estimator(model_fn, model_dir,config = my_config)
 
 
    if False:
        est.train(input_fn, steps=1)

    if False:
        est.evaluate(lambda: input_fn(repeat=1),100)

    if True:
        x = np.array([[5.0,3.3,1.4,0.2],[7.0,3.2,4.7,1.4],[5.1,2.5,3.0,1.1],[6.4,3.1,5.5,1.8]])  # numpy array어야 한다. list면 안됨
        y = [0,1,1,2]
        input_fn_predict = tf.compat.v1.estimator.inputs.numpy_input_fn(x = x,batch_size=len(x),shuffle=False)
        predictions = est.predict(input_fn_predict)
         
        #predictions =  [[p['logits'],p['prob']] for p in predictions]
        
        
        for p,t in zip(predictions,y):
            #print(f"label: {t}, logit: {p['logits']} ----> {p['prob']} ---> {np.argmax(p['prob'])}   ")          
            print(f"label: {t}, logit: {p['logits']} ----> {', '.join('{:.3f}'.format(x) for x in p['prob'])} ---> {np.argmax(p['prob'])}   ")          


if __name__ == "__main__": 
    #test1()
    
    #test2()   # train mode만
    
    #test3()
    
    test4()
