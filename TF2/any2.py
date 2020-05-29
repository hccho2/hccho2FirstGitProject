# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.seq2seq import Sampler



def alignment_test():

    vocab = 20
    embedding_dim = 6
    timestep = 20  #encoder time step
    memory_size = 15
    
    attention_cls = tfa.seq2seq.BahdanauAttention
    units = 11
    batch = 3
    
    
    
    encoder_inputs = tf.keras.Input(shape=[timestep])
    encoder_embedded = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)(encoder_inputs)
    
    
    encoder_output = tf.keras.layers.LSTM(memory_size, return_sequences=True)( encoder_embedded )   # memory
    
    attention_mechanism = attention_cls(units, encoder_output)
    processed_query = tf.keras.Input(shape=[units])  # decoder의 hidden state에 Wq가 곱해진 것
    prev_alignment = tf.keras.Input(shape=[timestep])   # 이전 alignment
    
    score = attention_mechanism([processed_query, prev_alignment])  # _calculate_attention 에서 alignment 계산 ---> alignments, next_state  ---> return 되는 2개는 같은 것
    
    model = tf.keras.Model([encoder_inputs, processed_query, prev_alignment], score)
    
    print(model.summary())
    
    
    
    
    encoder_input_data = tf.random.uniform([batch,timestep], 0,vocab,tf.int32)  # np.random.randint(vocab, size=(batch, timestep))
    processed_query_data = tf.random.normal([batch,units])
    prev_alignment_data1 = tf.nn.softmax(tf.random.normal([batch,timestep]),axis=-1)  # BahdanauAttention는 미전 alignment의 영향을 받지 않는다.
    prev_alignment_data2 = tf.nn.softmax(tf.random.normal([batch,timestep]),axis=-1)
    
    score_result = model([encoder_input_data,processed_query_data,prev_alignment_data1])
    print(score_result)
    
    score_result = model([encoder_input_data,processed_query_data,prev_alignment_data2])
    print(score_result)




encoder_vocab = 20
encoder_embedding_dim = 6

encoder_hidden_dim = 5

decoder_vocab = 23
decoder_embedding_dim = 7

decoder_hidden_dim = 8


attention_units = 11
batch_size = 3





encoder_embedded = tf.keras.layers.Embedding(encoder_vocab, encoder_embedding_dim, mask_zero=False)
encoder_output = tf.keras.layers.LSTM(encoder_hidden_dim, return_sequences=True)( encoder_embedded )   # memory   ---> (N,,encoder_timestep, encoder_embedding_dim)







decoder_embedding = tf.keras.layers.Embedding(decoder_vocab, decoder_embedding_dim, mask_zero=False)


decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
decoder_init_state = tuple(decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32))
attention_mechanism = tfa.seq2seq.BahdanauAttention(attention_units, encoder_output)
attention_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=decoder_init_state,output_attention=True,alignment_history=True)





##### encoder_test
encoder_timestep = 20  #encoder time step
encoder_inputs_data = tf.random.uniform([batch_size,encoder_timestep], 0,encoder_vocab,tf.int32)
 
memory = my_encoder(encoder_inputs_data)  # encoder_timestep은 fix되지 않아도 된다.


decoder_timestep = 10  #encoder time step
attention_init_state = attention_wrapper_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)


state = attention_init_state
output_all = []

decoder_input_data = tf.random.uniform([batch_size,decoder_timestep], 0,decoder_vocab,tf.int32)
for i in range(decoder_timestep):
    #decoder_embedded = decoder_embedding(decoder_inputs[:,i])  # 값이 없는 tensor가 들어가면 error!!!   decoder_inputs = tf.keras.Input(shape=[None],batch_size=batch_size)
    decoder_embedded = decoder_embedding(decoder_input_data[:,i])   
    out, state = attention_wrapper_cell(decoder_embedded,state)
    output_all.append(out)

output_all = tf.stack(output_all,axis=1)



####################
# memory test











print(output_all)

print('Done')