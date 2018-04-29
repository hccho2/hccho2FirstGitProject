# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
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


estimator = tf.contrib.learn.LinearRegressor(feature_columns=columns,optimizer="Adam")
estimator.fit(input_fn = input_fn_train,steps=5000)
estimator.evaluate(input_fn = input_fn_eval,steps=10)
result = list(estimator.predict_scores(input_fn = input_fn_predict))

print("result", result)






