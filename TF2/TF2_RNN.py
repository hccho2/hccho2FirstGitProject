# coding: utf-8

'''

https://www.tensorflow.org/tutorials/text/text_generation  ---> RNN 기초

https://github.com/tensorflow/addons/issues/1856   ---> 아직 bug가 있다. AttentionWraper의 state를 list로 할 것인가? tuple로 할 것인가? 정리가 되어 있지 않다.


tensorflow nmt + attenstion manual
https://www.tensorflow.org/tutorials/text/nmt_with_attention


get_initial_state()함수가 class type에 따라, 일관성이 없다....
'''


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.initializers import Constant



def simple_rnn():
    # https://www.tensorflow.org/guide/keras/rnn
    
    batch_size = 3
    seq_length = 5
    inputs = tf.random.normal([batch_size, seq_length, 7])
    rnn1 = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(4),tf.keras.layers.LSTMCell(11)],return_sequences=True)  # RNN(LSTMCell(units)) will run on non-CuDNN kernel
    
    state =  rnn1.get_initial_state(inputs)
    output = rnn1(inputs,state)
    print('output shape:', output.shape)  # return_sequences=False: (batch_size,11)         return_sequences = True --> batch_size,seq_length,11)



    rnn2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4), return_sequences=True, return_state=True)
    whole_seq_output, final_memory_state, final_carry_state = rnn2(inputs)
    print('output shape: {}, hidden_state_shape: {}, cell_state_shape: {}, '.format(whole_seq_output.shape,final_memory_state.shape,final_carry_state.shape  ))
    
    # tf.keras.layers.LSTM은 CuDNN Kernel 사용.
    rnn3 = tf.keras.layers.LSTM(4,return_sequences=True, return_state=True, name='encoder')  
    whole_seq_output, final_memory_state, final_carry_state = rnn3(inputs)
    print('output shape: {}, hidden_state_shape: {}, cell_state_shape: {}, '.format(whole_seq_output.shape,final_memory_state.shape,final_carry_state.shape  ))


def simple_rnn2():
    # RNN + BN + FC
    batch_size = 3
    seq_length = 5
    input_dim = 7
    
    hidden_dim = 9
    output_size = 11  
    
    
    def build_model(allow_cudnn_kernel=True):
        # CuDNN is only available at the layer level, and not at the cell level.
        # This means `LSTM(hidden_dim)` will use the CuDNN kernel,
        # while RNN(LSTMCell(hidden_dim)) will run on non-CuDNN kernel.
        if allow_cudnn_kernel:
            # The LSTM layer with default options uses CuDNN.
            lstm_layer = tf.keras.layers.LSTM(hidden_dim, input_shape=(None, input_dim),return_sequences=True)
        else:
            # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
            lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(hidden_dim),input_shape=(None, input_dim),return_sequences=True)
        
        
        model = tf.keras.models.Sequential([
                                            lstm_layer,
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.Dense(output_size)])
        return model
    
    model = build_model(allow_cudnn_kernel=False)
    inputs = tf.random.normal([batch_size, seq_length, input_dim])
    outputs = model(inputs)
    print(outputs.shape) # return_sequences=False: (batch_size,11)         return_sequences = True --> batch_size,seq_length,11)

def seq_loss_test():
    batch_size = 2
    seq_length = 5
    vocab_size= 6
    
    x = tf.random.normal((batch_size,seq_length,vocab_size))
    logit = tf.nn.log_softmax(x)
    
    target = tf.random.uniform([batch_size,seq_length], 0,vocab_size,tf.int32)
    
    predict = tf.argmax(logit,axis=-1)
    
    
    print(x.shape)
    print('target: ',target)
    #print('logit: ', logit)
    print('predict: ', predict)
    
    
    loss = -tf.gather_nd(tf.reshape(logit,[-1,vocab_size]), tf.stack([tf.range(batch_size*seq_length),tf.reshape(target,[-1])],axis=-1))
    loss = tf.reduce_mean(loss)
    
    weights = tf.ones(shape=[batch_size,seq_length])
    loss2 = tfa.seq2seq.sequence_loss(logit,target,weights)  # logit: (batch_size, seq_length, vocab_size),        target, weights: (batch_size, seq_length)
    loss3 = tfa.seq2seq.SequenceLoss()(target, logit,weights)   # target, logit 순으로 
    print(loss,loss2,loss3)


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
    target = tf.convert_to_tensor(y_data)

    # Decoder
    
    method = 2
    if method==1:
        # single layer RNN
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
        # decoder init state:
        
        #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
        init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    else:
        # multi layer RNN
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
        init_state = decoder_cell.get_initial_state(inputs=input)  #inputs=tf.zeros_like(x_data,dtype=tf.float32)로 해도 됨. inputs의 batch_size만 참조하기 때문에
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    
    decoder_method = 1
    if decoder_method==1:
        # tensorflow 1.x에서 tf.contrib.seq2seq.TrainingHelper
        sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
    
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
        outputs, last_state, last_sequence_lengths = decoder(input,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
    
    elif decoder_method==2:
        # tensorflow 1.x에서 tf.contrib.seq2seq.GreedyEmbeddingHelper
        sampler = tfa.seq2seq.GreedyEmbeddingSampler()  # alias ---> sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler
        
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)
        outputs, last_state, last_sequence_lengths = decoder(embedding.weights,initial_state=init_state,
                                                             start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token,training=False)    

    elif decoder_method==3:
        sampler = tfa.seq2seq.sampler.TrainingSampler()
        
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
        kwargs={'initial_state': init_state}
        outputs, last_state, last_sequence_lengths = tfa.seq2seq.dynamic_decode(decoder = decoder,maximum_iterations = seq_length,
                                                    impute_finished=True, output_time_major=False,decoder_init_input=input,decoder_init_kwargs=kwargs,training=True)

    elif decoder_method==4:
        sampler = tfa.seq2seq.GreedyEmbeddingSampler()
        
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
        kwargs={'initial_state': init_state, 'start_tokens': tf.tile([SOS_token], [batch_size]), 'end_token': EOS_token}
        outputs, last_state, last_sequence_lengths = tfa.seq2seq.dynamic_decode(decoder = decoder,maximum_iterations = seq_length,
                                                    impute_finished=True, output_time_major=False,decoder_init_input=embedding.weights, decoder_init_kwargs=kwargs,training=False)    





    logits = outputs.rnn_output
    
    print(logits.shape)

    if decoder_method==1 or decoder_method==3:
        weights = tf.ones(shape=[batch_size,seq_length])
        loss = tfa.seq2seq.sequence_loss(logits,target,weights)  # logit: (batch_size, seq_length, vocab_size),        target, weights: (batch_size, seq_length)
    
        print('loss: ', loss)
    
    
def decoder_train_test():
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    # x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    # y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    # print("data shape: ", x_data.shape)
    
    
    index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}
    x_data = np.array([[SOS_token, 1, 2, 3, 3, 4]], dtype=np.int32)
    y_data = np.array([[1, 2, 3, 3, 4,EOS_token]],dtype=np.int32)
    
    
    
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
    
    
    target = tf.convert_to_tensor(y_data)
    
    # Decoder
    method = 1
    if method==1:
        # single layer RNN
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
        # decoder init state:
        
        #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
        init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    else:
        # multi layer RNN
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
        init_state = decoder_cell.get_initial_state(inputs=tf.zeros_like(x_data,dtype=tf.float32))  # inputs의 batch_size만 참조하기 때문에
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    

    sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    for step in range(50):
        with tf.GradientTape() as tape:
            input = embedding(x_data)
            outputs, last_state, last_sequence_lengths = decoder(input,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
            logits = outputs.rnn_output
            
            weights = tf.ones(shape=[batch_size,seq_length])
            loss = tfa.seq2seq.sequence_loss(logits,target,weights)
        grads = tape.gradient(loss,embedding.weights+decoder.weights)
        optimizer.apply_gradients(zip(grads,embedding.weights+decoder.weights))
        
        if step%10==0:
            print(step, loss.numpy())
    
    
    
    sampler = tfa.seq2seq.GreedyEmbeddingSampler()  # alias ---> sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler
    sample_batch_size = 4
    
    decoder_type = 1
    if decoder_type==1:
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)
        #init_state = [tf.ones_like(init_state[0]),tf.ones_like(init_state[1])]
        init_state = [tf.random.normal(shape=(4,hidden_dim),mean=0.5),tf.random.normal(shape=(4,hidden_dim),mean=0.5)]
    else:
        beam_width=2
        decoder = tfa.seq2seq.BeamSearchDecoder(decoder_cell,beam_width,output_layer=projection_layer,maximum_iterations=seq_length)
        init_state = decoder_cell.get_initial_state(inputs=tf.zeros_like(x_data,dtype=tf.float32))
        
    outputs, last_state, last_sequence_lengths = decoder(embedding.weights,initial_state=init_state,
                                                         start_tokens=tf.tile([SOS_token], [sample_batch_size]), end_token=EOS_token,training=False) 
    
    result = tf.argmax(outputs.rnn_output,axis=-1).numpy()
    
    print(result)
    for i in range(sample_batch_size):
        print(''.join( index_to_char[a] for a in result[i] if a != EOS_token))

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
    #simple_rnn2()
    #seq_loss_test()
    #decoder_test()
    decoder_train_test()
    #attention_test()
    print('Done')






