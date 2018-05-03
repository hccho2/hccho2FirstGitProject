# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import iris_data
import timeline
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
    
    
    tf.logging._logger.setLevel(logging.INFO)   # 이게 있어야 출력이됨.(없으면 spyder에서만 출력이됨)
    
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
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
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
        batch_size=2,
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
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)   


        # Compute loss.     loss를 tf.estimator.ModeKeys.PREDICT 보다 앞쪽에 정의하면 predict 모드에서 error발생
        loss = tf.reduce_mean(tf.square(logits - labels))

        # Compute evaluation metrics.
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.mean_absolute_error(labels=labels,predictions=logits)
            metrics = {'xxxx': accuracy}    
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
   

       
        optimizer = tf.train.AdagradOptimizer(learning_rate=10)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())   
   
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : tf.reduce_mean(tf.abs(logits-labels))}, every_n_iter=200)
        logging_hook2 = tf.train.LoggingTensorHook({"my logits" : -logits, "my labels": labels}, every_n_iter=200)
        
        # dict형으로 첫번째 argument를 만들어 넘기면, fotmatter 함수로 넘겨진 my_format 함수의 argument로 dict형이 넘어간다.
        logging_hook3 = tf.train.LoggingTensorHook({"xxxx" : -logits, "yyy": labels}, every_n_iter=200,formatter=my_format)
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,training_hooks = [logging_hook,logging_hook2,logging_hook3])

    
    
    classifier = tf.estimator.Estimator(model_fn=hccho_model,model_dir = params['model_dir'] ,params=params)
    
    
    
    
    classifier.train( input_fn=input_fn_train,steps=1000)  
    
    

    # evaluation
    eval_result = classifier.evaluate(input_fn = input_fn_eval,steps=10)
    print('\nTest set loss: {loss:0.3f}, xxxx: {xxxx:0.3f}\n'.format(**eval_result))
    
    
    # predict
    A2 = np.array([[73., 80., 75.],[73., 66., 70.]])
    input_fn_predict = tf.estimator.inputs.numpy_input_fn(x = {"x":A2},batch_size=len(A2),shuffle=False)
    predictions = classifier.predict(input_fn=input_fn_predict)   # user defined function이 아니면, lambda function으로 넘기면 안됨
    print(list(predictions))

    

    
    
    
if __name__ == '__main__':
    #Run1()
    
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', default=100, type=int, help='batch size')
#     parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')
#     Run2()




    Run3()



    print('Done')

