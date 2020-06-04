
'''

tfa.seq2seq.BahdanauAttention ---> 이 놈이 encoder_output(즉, memory)를 넘겨 받아 생성되는 구조 
---> encoder_output이  계속 바뀌어야 하기 때문에,  numeric tensor가 들어갈 수가 없다. 
---> eager mode와 맞지가 않다. 그래서 encoder_output이 tf.keras.Input이 되거나, symbolic tensor여야 될 수 밖에 없다.
---> tf.keras.Input를 넘기는데 bug가 있다.


'''

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


def seq2seq_attention():
    encoder_vocab = 20
    encoder_embedding_dim = 6
    
    encoder_hidden_dim = 5
    
    decoder_vocab = 23
    decoder_embedding_dim = 7
    
    decoder_hidden_dim = 8
    
    
    attention_units = 11
    batch_size = 3
    
    
    
    
    encoder_inputs = tf.keras.Input(shape=[None],batch_size=batch_size)
    encoder_embedded = tf.keras.layers.Embedding(encoder_vocab, encoder_embedding_dim, mask_zero=False)(encoder_inputs)
    encoder_output = tf.keras.layers.LSTM(encoder_hidden_dim, return_sequences=True)( encoder_embedded )   # memory   ---> (N,,encoder_timestep, encoder_embedding_dim)
    
    my_encoder = tf.keras.Model(encoder_inputs,encoder_output)
    
    
    decoder_embedding = tf.keras.layers.Embedding(decoder_vocab, decoder_embedding_dim, mask_zero=False)
    
    
    decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
    decoder_init_state = tuple(decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32))
    attention_mechanism = tfa.seq2seq.BahdanauAttention(attention_units, encoder_output)
    attention_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=decoder_init_state,output_attention=True,alignment_history=True)
    
    
    
    
    
    ##### encoder_test
    encoder_timestep = 20  #encoder time step
    encoder_inputs_data = tf.random.uniform([batch_size,encoder_timestep], 0,encoder_vocab,tf.int32)
     
    memory = my_encoder(encoder_inputs_data)  # encoder_timestep은 fix되지 않아도 된다.
    
    
    decoder_timestep = 10  #decoder time step
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


def seq2seq_attention2():
    '''
    https://github.com/tensorflow/addons/issues/1898
    
    
    tfa.seq2seq.BahdanauAttention  ---> memory_sequence_length 에 tf.keras.Input가 들어가면 Error
    
    my_model  ---> return 하는 값이 왜 tensor일까? 수치적인 값이 나와야 되는데.....
    
    '''
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
    attention_mechanism = tfa.seq2seq.BahdanauAttention(attention_units, encoder_output,memory_sequence_length=None)
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
    
    
    my_model = tf.keras.Model([decoder_inputs,decoder_seq_length,encoder_output],[outputs, last_state, last_sequence_lengths])
    
    
    
    ### Test
    
    decoder_timestep = 12
    
    encoder_output_data = tf.random.normal(shape=(batch_size, encoder_timestep, encoder_hidden_dim))
    encoder_seq_length_data = tf.convert_to_tensor([encoder_timestep]*batch_size,dtype=tf.int32)
    
    
    decoder_inputs_data = tf.random.uniform([batch_size,decoder_timestep], 0,decoder_vocab_size,tf.int32)
    decoder_seq_length_data = tf.convert_to_tensor([decoder_timestep]*batch_size,dtype=tf.int32)
    
    
    a,b,c = my_model([decoder_inputs_data,decoder_seq_length_data,encoder_output_data])
    
    print(a)
    print('done')


def seq2seq_attention_graph_mode():
    # alignment_history=True ---> Error
    tf.compat.v1.disable_eager_execution()

    batch_size = 2
    
    
    encoder_hidden_dim = 6
    
    
    
    encoder_output = tf.compat.v1.placeholder(tf.float32, shape=[batch_size,None,encoder_hidden_dim])
    encoder_seq_length = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    
    decoder_vocab_size = 10
    decoder_embedding_dim = 8
    decoder_hidden_dim = 5
    attention_units = 11
    output_dim = 5
    
    decoder_cell = tf.keras.layers.LSTMCell(decoder_hidden_dim)
    decoder_init_state = tuple(decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32))
    attention_mechanism = tfa.seq2seq.BahdanauAttention(attention_units, encoder_output,memory_sequence_length=encoder_seq_length)
    attention_wrapper_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=decoder_init_state,
                                                          output_attention=True,alignment_history=False)
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    attention_init_state = attention_wrapper_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
    
    
    attention_init_state = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                              attention_init_state.alignment_history,attention_init_state.attention_state)
    
    
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    decoder = tfa.seq2seq.BasicDecoder(attention_wrapper_cell, sampler, output_layer=projection_layer)
    
    
    
    decoder_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size,None])
    decoder_seq_length = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    
    
    decoder_embedding = tf.keras.layers.Embedding(decoder_vocab_size, decoder_embedding_dim,trainable=True) 
    decoder_embedded = decoder_embedding(decoder_inputs)
    
    outputs, last_state, last_sequence_lengths = decoder(decoder_embedded,initial_state=attention_init_state, sequence_length=decoder_seq_length,training=True)
    
    
    
    
    print(outputs)
    
    
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    
    ### Test
    encoder_timestep = 10
    decoder_timestep = 12
    
    
    encoder_output_data = np.random.normal(0,1, [batch_size, encoder_timestep, encoder_hidden_dim])
    encoder_seq_length_data = np.array([encoder_timestep]*batch_size,dtype=np.int32)
    decoder_inputs_data = np.random.randint(decoder_vocab_size, size=[batch_size,decoder_timestep])
    decoder_seq_length_data = np.array([decoder_timestep]*batch_size,dtype=np.int32)
    
    
    a,b,c = sess.run([outputs,last_state, last_sequence_lengths], feed_dict={encoder_output: encoder_output_data,encoder_seq_length: encoder_seq_length_data, decoder_inputs: decoder_inputs_data,decoder_seq_length: decoder_seq_length_data })
     
     
    print(a)
    print(b)
    print(c)


if __name__ == '__main__':

    seq2seq_attention_graph_mode()



















