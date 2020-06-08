# coding: utf-8

'''
https://www.tensorflow.org/tutorials/structured_data/time_series

data file: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip(13M) --> 압축풀면  42M

- 10분단위 data
- 5일 data = 6*24*5=720 개 data로 부터 ===> 6시간 후 온도 예측.

'''

import tensorflow as tf

import matplotlib as mpl    # pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
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

def baseline(history):
    # 평균값으로 예측.
    return np.mean(history)









csv_path = 'jena_climate_2009_2016_simple.csv'    # 'jena_climate_2009_2016.csv' ,    'jena_climate_2009_2016_simple.csv'
df = pd.read_csv(csv_path)
print(df.head())

TRAIN_SPLIT = 400  #300000
tf.random.set_seed(13)

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
print('data 길이: ', len(uni_data))
print(uni_data.head())

uni_data.plot(subplots=True)   # 전체 data 그력보기.
plt.show()


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







print('Done')