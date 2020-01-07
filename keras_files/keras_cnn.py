import numpy as np
from keras import layers
from keras.models import Model,Sequential
import tensorflow as tf
tf.reset_default_graph()


vocab_size = 6
SOS_token = 0
EOS_token = 5

x_data = np.random.randn(2,20,20,3)

model = Sequential()
model.add(layers.Conv2D(32,(3,3), strides=(2,2),activation='relu',input_shape=(20,20,3)))  # default padding='valid'


y = model.predict(x_data)  # (20-3+1)/2 = 9 -->(N,9,9,32)
print(y.shape)




#if __name__ == '__main__':
#    test1()