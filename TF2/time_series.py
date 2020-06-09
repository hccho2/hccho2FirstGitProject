# coding: utf-8

'''
https://www.tensorflow.org/tutorials/structured_data/time_series

data file: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip(13M) --> 압축풀면  42M

- 10분단위 data
- 5일 data = 6*24*5=720 개 data로 부터 ===> 6시간 후 온도 예측.  ----> 아래 예에서는 12시간 후의 온도 예측

Data를 만들 때, padas로부터 만들면 1시간 소요됨. numpy array로 하면 1초~7초

3개의 model
1. 온도 feature 1개만 사용하여 온도 예측
2. 온도를 포함한 3개 feature를 사용하여 12시간 후 온도 예측
3. 온도를 포함한 3개 feature를 사용하여 12시간 후까지의 온도 예측  ---> 10분 단위로 12시간  예측. 72개 온도 예측


'''

import tensorflow as tf
import numpy as np
import matplotlib as mpl    # pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os,time
import pandas as pd  # pip install pandas
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False



def univariate_data(dataset, start_index, end_index, history_size, target_size):
    # start_index ~ end_index 사이의 data를 사용
    # i번째 data를 중심으로 보면, data: [i-history_size, i)  , labels: i+target_size 
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        #indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.array(dataset[i-history_size:i]).reshape(history_size, 1))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        #indices = range(i-history_size, i, step)
        data.append(dataset[i-history_size:i:step])
        
        if single_step:  # 예측값을 1개만 할지, 아니면 여러개 할지
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)






def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']  # rx: red X, go: green O
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def plot_train_history(history, title, validation_freq=None):
    # keras fit함수가 return하는 history를 plot
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    if validation_freq is not None:
        plt.plot(validation_freq, val_loss, 'r', label='Validation loss')
    else:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()

def multi_step_plot(history, true_future, prediction,STEP = 6):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',label='True Future')
    if prediction.any():  #  예측 결과도 주어져 있는 경우
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
def baseline(history):
    # 평균값으로 예측.
    return np.mean(history)


    
def test1():
    # feature로 온도 data 1개만 가지고, 온도를 예측한다.
    csv_path = 'jena_climate_2009_2016_simple.csv'    # 'jena_climate_2009_2016.csv' ,    'jena_climate_2009_2016_simple.csv'
    df = pd.read_csv(csv_path)
    print(df.head())
    
    TRAIN_SPLIT = 400  #300000
    tf.random.set_seed(13)
    
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    print('data 길이: ', len(uni_data))
    print(uni_data.head())
    
    uni_data.plot(subplots=True)   # 전체 data 그력보기.   ----> subplots=True ---> 각 column feature 별로 그래프가 그려진다.  False ---> 하나의 chart에 같이 그린다.
    plt.show()
    
    uni_data = uni_data.values    #    ---> 이 부분이 없으면, padas를 넘기게 되는데, padas로 data를 만들면 1시간 소요. numpy array로 넘기면  1초.
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    print('평군/분산: ',uni_train_mean,  uni_train_std)
    
    
    uni_data = (uni_data-uni_train_mean)/uni_train_std   # mean-variance normalization
    
    
    univariate_past_history = 20
    univariate_future_target = 0
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, univariate_past_history, univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
    
    
    
    print ('Single window of past history')
    print (x_train_uni[0])
    print ('\n Target temperature to predict')
    print (y_train_uni[0])
    
    
    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    plt.show()
    
    
    show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,'Baseline Prediction Example')
    plt.show()

def univariate_model():
    s_time = time.time()
    csv_path = 'jena_climate_2009_2016.csv'    # 'jena_climate_2009_2016.csv' ,    'jena_climate_2009_2016_simple.csv'
    df = pd.read_csv(csv_path)
    print(df.head())
    
    TRAIN_SPLIT = 300000  #300000
    tf.random.set_seed(13)
    
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    print('data 길이: ', len(uni_data))
    print(uni_data.head())
    
    
    uni_data = uni_data.values    #    ---> 이 부분이 없으면, padas를 넘기게 되는데, padas로 data를 만들면 1시간 소요. numpy array로 넘기면  1초.
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    print('평군/분산: ',uni_train_mean,  uni_train_std)
    
    
    uni_data = (uni_data-uni_train_mean)/uni_train_std   # mean-variance normalization
    
    
    univariate_past_history = 20
    univariate_future_target = 0
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, univariate_past_history, univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
    print('data loading: ', int(time.time()-s_time))
    print('x_train_uni shape: {}, y_train_uni shape: {}, x_val_uni shape: {}, y_val_uni shape: {}'.format(x_train_uni.shape,y_train_uni.shape,x_val_uni.shape,y_val_uni.shape ))
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    
    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    
    simple_lstm_model.compile(optimizer='adam', loss='mae')

    # val_univariate.take(10)   ----> 크기 10짜리 Dataset을 만든다.

    for x, y in val_univariate.take(3):
        print(x.shape, simple_lstm_model.predict(x).shape)


    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    
    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)


    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                        simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()




def multivariate_single_step_model():
    s_time = time.time()
    csv_path = 'jena_climate_2009_2016.csv'    # 'jena_climate_2009_2016.csv' ,    'jena_climate_2009_2016_simple.csv'
    df = pd.read_csv(csv_path)    

    features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
    
    features = df[features_considered]
    features.index = df['Date Time']
    print(features.head())
    
    features.plot(subplots=True)
    plt.show()



    TRAIN_SPLIT = 300000  #300000
    dataset = features.values   # ---> numpy array (N,3)
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std


    past_history = 720   # 720이면 5일치   
    future_target = 72   # 6*12  ---> 12시간 후의 온도 예측
    STEP = 6  # data는 10분 단위로 관측되어 있다. ----> 1시간 단위의 data를 사용
    # 종합하면   120개 data를 사용 (N,120,3)
    
    
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0, TRAIN_SPLIT, past_history,
                                                       future_target, STEP, single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1], TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP, single_step=True)

    print('data loading: ', int(time.time()-s_time))
    print ('Single window of past history : {}'.format(x_train_single[0].shape))   # (120,3)
    print('x_train_single shape: {}, y_train_single shape: {}, x_val_single shape: {}, y_val_single shape: {}'.format(x_train_single.shape,y_train_single.shape,x_val_single.shape,y_val_single.shape ))

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    
    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')


    for x, y in train_data_single.take(2):
        print(x.shape, single_step_model.predict(x).shape)

    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)




    plot_train_history(single_step_history,'Single Step Training and validation loss')

    for x, y in val_data_single.take(3):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                        single_step_model.predict(x)[0]], 12,
                       'Single Step Prediction')
        plot.show()



def multivariate_multi_step_model():
    s_time = time.time()
    csv_path = 'jena_climate_2009_2016.csv'    # 'jena_climate_2009_2016.csv' ,    'jena_climate_2009_2016_simple.csv'
    df = pd.read_csv(csv_path)    

    features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
    
    features = df[features_considered]
    features.index = df['Date Time']
    print(features.head())
    
    features.plot(subplots=True)
    plt.show()



    TRAIN_SPLIT = 300000  # 전체 data는 약 42만 lines
    dataset = features.values   # ---> numpy array (N,3)
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    
    dataset = (dataset-data_mean)/data_std

    past_history = 720   # 720이면 5일치   ----> STEP간격으로 추출
    future_target = 72   # 6*12  ---> 12시간 후의 온도 예측
    STEP = 6  # data는 10분 단위로 관측되어 있다. ----> 1시간 단위의 data를 사용
    # 종합하면   120개 data를 사용 (N,120,3)
    
    # 
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,TRAIN_SPLIT, past_history, future_target, STEP,single_step=False)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],TRAIN_SPLIT, None, past_history, future_target, STEP,single_step=False)

    print('data loading: ', int(time.time()-s_time))
    print ('Single window of past history : {}'.format(x_train_multi[0].shape))
    print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

    print('x_train_multi shape: {}, y_train_multi shape: {}, x_val_multi shape: {}, y_val_multi shape: {}'.format(x_train_multi.shape,y_train_multi.shape,x_val_multi.shape,y_val_multi.shape ))
  
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()




    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]),STEP)
        plt.show()

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(72))
    
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    


    for x, y in train_data_multi.take(2):
        print(x.shape, multi_step_model.predict(x).shape)

    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    
    
    # steps_per_epoch=EVALUATION_INTERVAL  -----> Dataset으로 data가 주어져 있기 때문에 batch_size가 Datset속에 정해져 있다. 그래서 몇 step을 1 ephoch로 볼 것인가?
    # validation_steps=50  ----> validation data로 Dataset으로 주어져 있기 때문에, validation loss를 몇 변 계산할 것인가?
    validation_freq=[1, 4, 6,9,10]
    # validation_freq= 2 ---> 매 2 epoch마다
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_multi,
                                                validation_steps=50,validation_freq=validation_freq)



    # validation_freq=[1, 4, 6,9,10] ---> 이렇게 주면 train, validation loss갯수가 맞지 않아 error
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss',np.array(validation_freq)-1)

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


if __name__ == '__main__':
    #test1()
    #data_load_test()
    univariate_model()
    #multivariate_single_step_model()
    #multivariate_multi_step_model()

    print('Done')