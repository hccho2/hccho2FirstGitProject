import numpy as np
from keras.layers import LSTM, Embedding
from keras.models import Model,Sequential
import tensorflow as tf
tf.reset_default_graph()

def test1():
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    T = x_data.shape[1]
    
    embedding_dim = 7
    model = Sequential()
    model.add(Embedding(vocab_size,embedding_dim))  # (N,T,embedding_dim)
    model.add(LSTM(32,input_shape=(T,embedding_dim)))  # sequence의 마지막 만 return. input_shape는 아무 역할이 없음.
    #model.add(LSTM(32,return_sequences=True))
    
    
    y = model.predict(x_data)
    print(y.shape)




if __name__ == '__main__':
    test1()