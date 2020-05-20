# coding: utf-8

'''

https://www.tensorflow.org/tutorials/text/text_generation  ---> RNN 기초

https://github.com/tensorflow/addons/issues/1856   ---> 아직 bug가 있다. AttentionWraper의 state를 list로 할 것인가? tuple로 할 것인가? 정리가 되어 있지 않다.

'''


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.initializers import Constant



def simple_rnn():
    # https://www.tensorflow.org/guide/keras/rnn
    
    inputs = tf.random.normal([3, 5, 7])
    rnn1 = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(4),tf.keras.layers.LSTMCell(11)])  # RNN(LSTMCell(units)) will run on non-CuDNN kernel
    output = rnn1(inputs)
    print('output shape:', output.shape)



    rnn2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4), return_sequences=True, return_state=True)
    whole_seq_output, final_memory_state, final_carry_state = rnn2(inputs)
    print('output shape: {}, hidden_state_shape: {}, cell_state_shape: {}, '.format(whole_seq_output.shape,final_memory_state.shape,final_carry_state.shape  ))
    
    # tf.keras.layers.LSTM은 CuDNN Kernel 사용.
    rnn3 = tf.keras.layers.LSTM(4,return_sequences=True, return_state=True, name='encoder')  
    whole_seq_output, final_memory_state, final_carry_state = rnn3(inputs)
    print('output shape: {}, hidden_state_shape: {}, cell_state_shape: {}, '.format(whole_seq_output.shape,final_memory_state.shape,final_carry_state.shape  ))


def simple_rnn2():
    
    batch_size = 3
    input_dim = 5
    
    units = 7
    output_size = 6  
    
    
    def build_model(allow_cudnn_kernel=True):
        # CuDNN is only available at the layer level, and not at the cell level.
        # This means `LSTM(units)` will use the CuDNN kernel,
        # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
        if allow_cudnn_kernel:
            # The LSTM layer with default options uses CuDNN.
            lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
        else:
            # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
            lstm_layer = tf.keras.layers.RNN(
                tf.keras.layers.LSTMCell(units),
                input_shape=(None, input_dim))
        model = tf.keras.models.Sequential([
            lstm_layer,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_size)]
        )
        return model
    
    model = build_model(allow_cudnn_kernel=True)
    inputs = tf.random.normal([batch_size, 5, input_dim])
    outputs = model(inputs)




def decoder_test():

    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 0
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    ##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 
    
    input = embedding(x_data)
    

    
    # Sampler
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    
    # Decoder
    
    method = 2
    if method==1:
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
        # decoder init state:
        
        #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
        init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    else:
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
        init_state = decoder_cell.get_initial_state(inputs=input)
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    outputs, last_state, last_sequence_lengths = decoder(input,initial_state=init_state, sequence_length=[seq_length]*batch_size)
    logits = outputs.rnn_output
    
    print(logits.shape)


def attention_test():
    
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 0
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    ##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 
    
    decoder_input = embedding(x_data)


    encoder_outputs = tf.random.normal(shape=(batch_size, 20, 30))  # encoder length=20, encoder_dim= 30
    encoder_sequence_length = [10,20,15]  # batch에 대한, encoder의 길이. padding이 있을 수 있기 때문. [20]*batch_size

    # units = Na = 11 <---- score 계산하기 전에, 몇차 vector를 만들 것인지 결정.
    attention_mechanism = tfa.seq2seq.BahdanauAttention(units=11, memory=encoder_outputs, memory_sequence_length=encoder_sequence_length)
    #attention_mechanism = tfa.seq2seq.LuongAttention(units=hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_sequence_length)




    # decoder init state:
    init_state = (tf.ones((batch_size,hidden_dim)), tf.zeros((batch_size,hidden_dim)))   # tuple(h,c) --> [h,c] ---> error남.
    
    
    # Sampler
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    
    # Decoder
    decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
    
    # tfa.seq2seq.AttentionWrapper의 initial_cell_state로 tuple을 넣어야 되는데... 이건 버그임. 
    decoder_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=init_state,alignment_history=True)
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    attention_init_state = decoder_cell.get_initial_state(inputs = None, batch_size = batch_size, dtype=tf.float32)  # inputs의 역할은 없느데.. .source보면.
    
    attention_init_state2 = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                              attention_init_state.alignment_history,attention_init_state.attention_state)
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    

    outputs, last_state, last_sequence_lengths = decoder(decoder_input,initial_state=attention_init_state2, sequence_length=[seq_length]*batch_size)
    logits = outputs.rnn_output
    
    print(logits.shape)


if __name__ == '__main__':
    #simple_rnn()
    simple_rnn2()
    #decoder_test()
    #attention_test()
    print('Done')









