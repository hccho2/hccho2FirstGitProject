# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.seq2seq import Sampler


def try1():
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    
    
    output_dim = vocab_size
    decoder_hidden_dim =7
    decoder_embedding_dim = 8
    encoder_hidden_dim = 30
    
    batch_size = 3
    
    
    
    decoder_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)  # (None,None)
    encoder_outputs = tf.keras.Input(shape=(None,encoder_hidden_dim),batch_size=batch_size,dtype=tf.float32)  #(None,None,encoder_hidden_dim)
    encoder_sequence_length = tf.keras.Input(shape=(),dtype=tf.int32)
    
    
    
    decoder_sequence_length = 10
    
    
    
    embedding = tf.keras.layers.Embedding(vocab_size, decoder_embedding_dim,trainable=True) 
    
    
    
    
    decoder_embedded_inputs = embedding(decoder_inputs)
    attention_mechanism = tfa.seq2seq.BahdanauAttention(units=11, memory=encoder_outputs, memory_sequence_length=encoder_sequence_length)
    
    init_state = (tf.zeros((batch_size,decoder_hidden_dim)), tf.zeros((batch_size,decoder_hidden_dim)))   # tuple(h,c) --> [h,c] ---> error남.
    
    
    # Sampler
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    
    # Decoder
    decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
    decoder_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=init_state,output_attention=True,alignment_history=True)
    
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    attention_init_state = decoder_wrapper_cell.get_initial_state(inputs = None, batch_size = batch_size, dtype=tf.float32)  # inputs의 역할은 없느데.. .source보면.
    
    attention_init_state2 = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                              attention_init_state.alignment_history,attention_init_state.attention_state)
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_wrapper_cell, sampler, output_layer=projection_layer)
    
    
    
    outputs, last_state, last_sequence_lengths = decoder(decoder_embedded_inputs,initial_state=attention_init_state2, sequence_length=[decoder_sequence_length]*batch_size,training=True)
    
    
    #model = tf.keras.Model([decoder_inputs, encoder_outputs, encoder_sequence_length],[outputs, last_state, last_sequence_lengths])


vocab_size = 6
SOS_token = 0
EOS_token = 5

x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
print("data shape: ", x_data.shape)


output_dim = vocab_size
decoder_hidden_dim =7
decoder_embedding_dim = 8
encoder_hidden_dim = 30

batch_size = 3
decoder_sequence_length = 6


decoder_inputs = tf.keras.Input(shape=(decoder_sequence_length,), dtype=tf.int32)  # (None,None)
encoder_outputs = tf.keras.Input(shape=(None,encoder_hidden_dim),batch_size=batch_size,dtype=tf.float32)  #(None,None,encoder_hidden_dim)
encoder_sequence_length = tf.keras.Input(shape=(),dtype=tf.int32)







embedding = tf.keras.layers.Embedding(vocab_size, decoder_embedding_dim,trainable=True) 




decoder_embedded_inputs = embedding(decoder_inputs)
attention_mechanism = tfa.seq2seq.BahdanauAttention(units=11, memory=encoder_outputs, memory_sequence_length=encoder_sequence_length)

init_state = (tf.zeros((batch_size,decoder_hidden_dim)), tf.zeros((batch_size,decoder_hidden_dim)))   # tuple(h,c) --> [h,c] ---> error남.


# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
decoder_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=init_state,output_attention=True,alignment_history=True)



projection_layer = tf.keras.layers.Dense(output_dim)

attention_init_state = decoder_wrapper_cell.get_initial_state(inputs = None, batch_size = batch_size, dtype=tf.float32)  # inputs의 역할은 없느데.. .source보면.

attention_init_state2 = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                          attention_init_state.alignment_history,attention_init_state.attention_state)

decoder = tfa.seq2seq.BasicDecoder(decoder_wrapper_cell, sampler, output_layer=projection_layer)
kwargs={'initial_state': init_state}


outputs, last_state, last_sequence_lengths = tfa.seq2seq.dynamic_decode(decoder = decoder,maximum_iterations = decoder_sequence_length,
                                            impute_finished=True, output_time_major=False,decoder_init_input=decoder_embedded_inputs,decoder_init_kwargs=kwargs,training=True)



#model = tf.keras.Model([decoder_inputs, encoder_outputs, encoder_sequence_length],[outputs, last_state, last_sequence_lengths])
