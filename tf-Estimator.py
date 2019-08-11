# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import iris_data
import timeline
from tensorflow.python.keras import backend as K

"""
model_dir에 checkpoint 파일이 저장되어 있으면 load하여 train을 이어간다.


# TensorFlow에서는 5가지의 로깅 타입을 제공하고 있습니다. ( DEBUG, INFO, WARN, ERROR, FATAL ) INFO가 설정되면, 그 이하는 다 출력된다.
tf.logging.set_verbosity(tf.logging.INFO) # 이게 있어야 train log가 출력된다.
"""


def Run1():

    num_points = 300
    
    vectors_set = []
    
    for i in range(num_points):
        x = np.random.normal(5,5)+15
        y =  x*2+ (np.random.normal(0,3))*2
        vectors_set.append([x,y])
    
      
    
    x_data = [v[0] for v in vectors_set ]
    y_data = [v[1] for v in vectors_set ]
    A1 = plt.plot(x_data,y_data,'ro')
    plt.ylim([0,100])
    plt.xlim([5,35])
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.legend([A1[0]], ['Hello'])
    
    plt.show()
    
    
    #tf.logging._logger.setLevel(logging.INFO)   # 이게 있어야 출력이됨.(없으면 spyder에서만 출력이됨)
    tf.logging.set_verbosity(tf.logging.INFO)
    
    
    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array(x_data[:200],dtype=np.float32)},
        y = np.array(y_data[:200],dtype=np.float32),
        num_epochs=100000,
        batch_size=50,
        shuffle=True
    )
    
    """
    ###  usage: numpy_io.numpy_input_fn
    
    age = np.arange(4) * 1.0
    height = np.arange(32, 36)
    x = {'age': age, 'height': height}
    y = np.arange(-32, -28)
    
    with tf.Session() as session:
        input_fn = numpy_io.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)
    """    
    
    
    input_fn_eval = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array(x_data[200:300],dtype=np.float32)},
        y = np.array(y_data[200:300],dtype=np.float32),
        num_epochs=100000,
        batch_size=50,
        shuffle=True
    )
    
    input_fn_predict = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array([15,20,25,30],dtype=np.float32)},
        num_epochs=1,
        shuffle=False
    )
    column_x = tf.feature_column.numeric_column("x",dtype=tf.float32)
    columns = [column_x]
    
    
    estimator = tf.estimator.LinearRegressor(feature_columns=columns,optimizer="Adam")
    estimator.train(input_fn = input_fn_train,steps=5000)  # iteration회수 = steps. 단 input_fn_eval에서 명시한  num_ephochs/batch_size를 넘지는 않는다.
    estimator.evaluate(input_fn = input_fn_eval,steps=10)
    result = list(estimator.predict(input_fn = input_fn_predict))
    
    print("result", result)

def Run2():

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


    
def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        # K.learning_phase(): False
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)   # loss, eval_metric_ops른 넣었기 때문에 2개가 return

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')
    args = parser.parse_args(argv[1:])
    


    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()   #pandas (120, 4), (120,) , (30, 4), (30,)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir = 'D:\\hccho\\RNN\\seq2seq\\Estimator-ckpt',
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))    


def my_format(values):
    res = []
    for v in values:
        res.append('{}={}'.format(v,values[v]))
    return '\n'.join(res)
def Run3():
    # train, eval, predict mode에서의 다양한 출력
    tf.logging.set_verbosity(tf.logging.INFO)
    A = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
    B = np.array([[152.],[185.],[180.],[196.],[142.]])



    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array(A,dtype=np.float32)},
        y = np.array(B,dtype=np.float32),
        num_epochs=10000,
        batch_size=5,
        shuffle=True
    )


    input_fn_eval = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array(A,dtype=np.float32)},
        y = np.array(B,dtype=np.float32),
        num_epochs=1,
        batch_size=2,
        shuffle=False
    )
    column_x = tf.feature_column.numeric_column("x",shape=(3,),dtype=tf.float32)
    my_feature_columns = [column_x]
    
    
    
    params={'feature_columns': my_feature_columns,'hidden_units': [10, 10],'n_classes': 3, 'model_dir': 'D:\\hccho\\RNN\\seq2seq\\Estimator-ckpt' }
    
    
     
    
    def hccho_model(features, labels, mode, params):
    
        inputs = tf.feature_column.input_layer(features, params['feature_columns'])
    
        # model
        L1 = tf.layers.dense(inputs,units=5, activation = tf.nn.relu,name='L1')
        logits = tf.layers.dense(L1,units=1, activation = None,name='L2')

        # predicction
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'logits': logits, 'L1': L1,
            }
            prediction_hooks = tf.train.LoggingTensorHook({"prediction hook-----my logits" : -logits, "prediction hook2----L1": L1}, every_n_iter=1)
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,prediction_hooks=[prediction_hooks])   # predictions를 train mode에 넣어도 계산해주지 않는다.


        # Compute loss.     loss를 tf.estimator.ModeKeys.PREDICT 보다 앞쪽에 정의하면 predict 모드에서 error발생
        loss = tf.reduce_mean(tf.square(logits - labels))

        # Compute evaluation metrics.
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.mean_absolute_error(labels=labels,predictions=logits)
            metrics = {'xxxx': accuracy}    
            metrics = {'xxxx': accuracy}    
            eval_logging_hook = tf.train.LoggingTensorHook({"eval-----my logits" : -logits, "eval2----my labels": labels}, every_n_iter=1)  # 각 iteration에서 계산되는 값
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics,evaluation_hooks=[eval_logging_hook]) # 전체 iteration 평균값   

        #optimizer = tf.train.GradientDescentOptimizer(0.00001)   
        optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)  # AdagradOptimizer는 lr이 좀 높아야 되네...
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())   
   
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : tf.reduce_mean(tf.abs(logits-labels))}, every_n_iter=200)
        logging_hook2 = tf.train.LoggingTensorHook({"my logits" : -logits, "my labels": labels}, every_n_iter=200)
        
        # dict형으로 첫번째 argument를 만들어 넘기면, fotmatter 함수로 넘겨진 my_format 함수의 argument로 dict형이 넘어간다.
        logging_hook3 = tf.train.LoggingTensorHook({"xxxx" : -logits, "yyy": labels}, every_n_iter=200,formatter=my_format)
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks = [logging_hook,logging_hook2,logging_hook3])

    
    
    classifier = tf.estimator.Estimator(model_fn=hccho_model,model_dir = params['model_dir'] ,params=params)
    
    
    
    
    classifier.train( input_fn=input_fn_train,steps=1000)  # steps는 최대 실행 횟수이다. data가 다 소진되면 steps를 다 채우지 못할 수도 있다.
    
    

    # evaluation
    print("Evaluation ================================")
    eval_result = classifier.evaluate(input_fn = input_fn_eval,steps=2)  # steps는 최대 실행 횟수이다. data가 다 소진되면 steps를 다 채우지 못할 수도 있다.
    print('\nTest set loss: {loss:0.3f}, xxxx: {xxxx:0.3f}\n'.format(**eval_result))
    
    
    # predict
    print("Prediction ================================")
    A2 = np.array([[73., 80., 75.],[58., 66., 70.]])
    input_fn_predict = tf.estimator.inputs.numpy_input_fn(x = {"x":A2},batch_size=len(A2),shuffle=False)
    predictions = classifier.predict(input_fn=input_fn_predict)   # user defined function이 아니면, lambda function으로 넘기면 안됨
    print(list(predictions))

    

def Run4():
    # input data를 placeholder로...
    tf.logging.set_verbosity(tf.logging.INFO)
    mnist = input_data.read_data_sets("D:\\hccho\\CommonDataset\\mnist", one_hot=True)
    batch_size = 128

    def input_fn_train():
        inp = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
        output = tf.placeholder(tf.int64, shape=[None, 10], name='y')
        #return {'x': inp,'y': output}, None    # 두번째 return targets은 여기서는 사용하지 않으므로 None
        return {'my_x': inp}, output    # targets를 사용하려면...
    
    
    
    def feed_fn():
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        return {'x:0': batch_x, 'y:0': batch_y }
    
    
    def hccho_model2(features, labels, mode, params):
        # 이 model 함수는 아래의 classifier.train, classifier.evaluate 실행될 때, 각각 불려진다.
        x = features['my_x']
        
    
        # model
        init = tf.contrib.layers.xavier_initializer()
        L1 = tf.layers.dense(x,units=256, kernel_initializer=init,activation = tf.nn.relu,name='L1')
        L2 = tf.layers.dense(L1,units=128, kernel_initializer=init, activation = tf.nn.relu,name='L2')
        logits = tf.layers.dense(L2,units=10, kernel_initializer=init,activation = None,name='L3')

        predicted_classes = tf.argmax(logits,axis=1)
        # predicction
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'predict': predicted_classes }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)   

        
        #y = features['y']
        y = labels
        
        # Compute loss.     loss를 tf.estimator.ModeKeys.PREDICT 보다 앞쪽에 정의하면 predict 모드에서 error발생
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits = logits))
        accuracy = tf.metrics.accuracy(labels=tf.argmax(y,axis=1), predictions=predicted_classes, name='acc_op')
        # Compute evaluation metrics.
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {'acc=====': accuracy,'acc---': accuracy} # 이곳에 들어가는 op는 tf.metrics.accuracy로 만들어 진 것이어야 한다.
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
   

       
        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())   
        logging_hook = tf.train.LoggingTensorHook({"loss----" : loss, "accuracy" : accuracy[1] }, every_n_iter=200)
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks = [logging_hook])
    
    params={'hidden_units': [10, 10],'n_classes': 10, 'model_dir': 'D:\\hccho\\RNN\\seq2seq\\Estimator-ckpt' }
    
    
    """
    log_step_count_steps: log 출력 주기, global_step 기준이 아니고, 이어서 시작한 step으로 부터
    save_checkpoints_steps: checkpoint save 주기. 이 주기마다 저장 + 제일 마지막에도 저장.
    
    """
    
    my_config =tf.estimator.RunConfig(log_step_count_steps=500,save_summary_steps=500,save_checkpoints_steps=1000) # INFO:tensorflow:global_step/sec: 317.864  <--- 출력회수 제어
    classifier = tf.estimator.Estimator(model_fn=hccho_model2,model_dir = params['model_dir'] ,params=params,config = my_config)
    
    
    # hooks를 통해. feed_fn이 input_fn_train에 있는 palceholder를 채워준다.
    classifier.train( input_fn=input_fn_train,hooks=[tf.train.FeedFnHook(feed_fn)],steps=1000)  # steps=train 회수
    print("---Evaluation---")
    classifier.evaluate( input_fn=input_fn_train,hooks=[tf.train.FeedFnHook(feed_fn)],steps=1)  

def Run5():
    # input function을 tf.data.Dataset를 이용하여 구현
    #
    
    DATA_SIZE = 1000
    
    BATCH_SIZE = 3
    NUM_EPOCHS = 1
    
    train_input = np.random.randn(DATA_SIZE,3)
    train_label = np.random.randint(2,size=DATA_SIZE).reshape(-1,1)
    
    def mapping_fn(X, Y):
        # def model_fn(features, labels, mode):  
        #     features['x']로 접근
        inputs, labels = {'x': X}, Y
        return inputs, labels
    
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label))
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.batch(BATCH_SIZE,drop_remainder=True) # BATCH_SIZE단위로 자르고, 남는 자투리 처리 여부
        dataset = dataset.repeat(count=NUM_EPOCHS)
        
        ### tf.data.Dataset.from_tensor_slices((A,B,C,D)) 와 같이 4개의 argument를 받는 다면,
        ### 아래에 넣어주는 mapping_fn은 4개의 argument를 받는 함수여야 한다. return 값은 model_fn(features, labels, mode, params) 이 받는 
        ### 4개 중에 처음 2개. 
        ### 첫번째 features는 dict형. 두번째는 labels는 tensor 이다.
        
        dataset = dataset.map(mapping_fn)
        iterator = dataset.make_one_shot_iterator()
        
        return iterator.get_next()
    
    
    a1 = train_input_fn()
    a2 = train_input_fn()
    
    sess = tf.Session()
    b1,b2 = sess.run([a1,a2])
    
    
    train_input2 = np.random.randn(DATA_SIZE,3)
    
    def mapping_fn2(base, hypothesis, label):
        # def model_fn(features, labels, mode):  
        #     features['x1'], features['x2']로 접근
        features = {"x1": base, "x2": hypothesis}
        return features, label
    
    def train_input_fn2():
        dataset = tf.data.Dataset.from_tensor_slices((train_input, train_input2, train_label))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(mapping_fn2)
        dataset = dataset.repeat(NUM_EPOCHS)
        iterator = dataset.make_one_shot_iterator()
        
        return iterator.get_next()
    
    c1 = train_input_fn2()
    c2 = train_input_fn2()
    
    d1,d2 = sess.run([c1,c2])
    print(d1,d2)


def argparse_test():
    # python tf-Estimator.py --batch_size 124  <----add_argument를 통해 추가한 것만 가능
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')
    args = parser.parse_args()
    print(args.batch_size)  
if __name__ == '__main__':
    
    #argparse_test()
    
    #Run1()  # tf.estimator.LinearRegressor
    #Run2()
    #Run3()

    Run4()
    
    #Run5()

    print('Done')
