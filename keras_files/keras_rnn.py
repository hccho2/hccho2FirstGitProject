import numpy as np
from keras.layers import LSTM, Embedding
from keras.models import Model,Sequential
import tensorflow as tf
tf.reset_default_graph()

def basic()
    from tensorflow.keras import layers
    encoder_vocab = 1000
    decoder_vocab = 2000

    encoder_input = layers.Input(shape=(None, ))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM( 64, return_state=True, name='encoder')(encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None, ))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(64, name='decoder')(decoder_embedded, initial_state=encoder_state)
    output = layers.Dense(10)(decoder_output)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model.summary()


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
