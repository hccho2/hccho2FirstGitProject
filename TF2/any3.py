
# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


#tf.compat.v1.disable_eager_execution()


batch_size = 2
encoder_timestep = 10

encoder_hidden_dim = 6
encoder_output = tf.keras.Input(shape=[None,encoder_hidden_dim],batch_size=batch_size) # Input
encoder_seq_length = tf.keras.Input(shape=[],batch_size=batch_size, dtype=tf.int32) # Input
#encoder_seq_length = tf.convert_to_tensor([encoder_timestep]*batch_size,dtype=tf.int32)

decoder_vocab_size = 10
decoder_embedding_dim = 8
decoder_hidden_dim = 5
attention_units = 11
output_dim = 5

decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
decoder_init_state = tuple(decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32))
attention_mechanism = tfa.seq2seq.BahdanauAttention(attention_units, encoder_output,memory_sequence_length=encoder_seq_length)
attention_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=decoder_init_state,output_attention=True,alignment_history=False)
projection_layer = tf.keras.layers.Dense(output_dim)


attention_init_state = attention_wrapper_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)


attention_init_state = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                          attention_init_state.alignment_history,attention_init_state.attention_state)


sampler = tfa.seq2seq.sampler.TrainingSampler()
decoder = tfa.seq2seq.BasicDecoder(attention_wrapper_cell, sampler, output_layer=projection_layer)

decoder_inputs = tf.keras.Input(shape=[None],batch_size=batch_size, dtype=tf.int32)   # Input
decoder_seq_length = tf.keras.Input(shape=[],batch_size=batch_size, dtype=tf.int32)  # Input

decoder_embedding = tf.keras.layers.Embedding(decoder_vocab_size, decoder_embedding_dim,trainable=True) 
decoder_embedded = decoder_embedding(decoder_inputs)

outputs, last_state, last_sequence_lengths = decoder(decoder_embedded,initial_state=attention_init_state, sequence_length=decoder_seq_length,training=True)


my_model = tf.keras.Model([decoder_inputs,decoder_seq_length,encoder_output,encoder_seq_length],[outputs, last_state, last_sequence_lengths])



### Test

decoder_timestep = 12

encoder_output_data = tf.random.normal(shape=(batch_size, encoder_timestep, encoder_hidden_dim))
encoder_seq_length_data = tf.convert_to_tensor([encoder_timestep]*batch_size,dtype=tf.int32)


decoder_inputs_data = tf.random.uniform([batch_size,decoder_timestep], 0,decoder_vocab_size,tf.int32)
decoder_seq_length_data = tf.convert_to_tensor([decoder_timestep]*batch_size,dtype=tf.int32)


a,b,c = my_model([decoder_inputs_data,decoder_seq_length_data,encoder_output_data,encoder_seq_length_data])

print(a)
print('done')

# 
# out, state = attention_wrapper_cell(decoder_inputs1,attention_init_state)
# print(out, state)
# 
# out, state = attention_wrapper_cell(decoder_inputs2,attention_init_state)  # error!!!!
# print(out, state)
# 
# 
# 
# my_model = tf.keras.Model([decoder_inputs2, encoder_output],[out, state])



















