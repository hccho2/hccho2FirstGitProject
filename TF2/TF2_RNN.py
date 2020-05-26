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
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import Constant

class MyProjection(tf.keras.layers.Layer):  # tf.keras.layers.Layer    tf.keras.Model
    def __init__(self,output_dim):
        super(MyProjection, self).__init__()   # 이게 있어야 한다.
        self.output_dim = output_dim
        self.L1 = tf.keras.layers.Dense(20, activation = tf.nn.relu)
        self.L2 = tf.keras.layers.Dense(self.output_dim) 
    
    def build(self, input_shape):
        # tf.keras.layers.Layer를 상속받았을 때에는 .필요한 weight들을 여기서 만들어 주어야 한다.
        # input_shape을 참고해서, 필요한 크기의 weight를 생성한다.
        #self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        pass
        
    def call(self, inputs,training=None):
        y = self.L1(inputs)
        z = self.L2(y)
        return z



class MyProjection2(tf.keras.Model):  # tf.keras.layers.Layer    tf.keras.Model
    # tf.keras.Model은 새로운 weight 없이, 기존의 layer들의 조합으로 새로운 layer를 만들 때 사용하면 좋다.
    def __init__(self,output_dim):
        super(MyProjection2, self).__init__()   # 이게 있어야 한다.
        self.output_dim = output_dim
        self.L1 = tf.keras.layers.Dense(20, activation = tf.nn.relu)
        self.L2 = tf.keras.layers.Dense(self.output_dim) 
    
        
    def call(self, inputs,training=None):
        y = self.L1(inputs)
        z = self.L2(y)
        return z


def simple_rnn():
    # https://www.tensorflow.org/guide/keras/rnn
    
    batch_size = 3
    seq_length = 5
    input_dim = 7
    hidden_dim = 4
    inputs = tf.random.normal([batch_size, seq_length, input_dim])
    cells = [tf.keras.layers.SimpleRNNCell(hidden_dim),tf.keras.layers.LSTMCell(hidden_dim*2)] # 또는  cells = tf.keras.layers.StackedRNNCells(cells)
    rnn1 = tf.keras.layers.RNN(cells,return_sequences=True)  # RNN(LSTMCell(units)) will run on non-CuDNN kernel
    
    initial_state =  rnn1.get_initial_state(inputs)
    output = rnn1(inputs,initial_state)
    print('output shape:', output.shape)  # return_sequences=False: (batch_size,x)         return_sequences = True --> batch_size,seq_length,x)



    rnn2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4), return_sequences=True, return_state=True)
    initial_state =  rnn2.get_initial_state(inputs)
    whole_seq_output, final_memory_state, final_carry_state = rnn2(inputs,initial_state)
    print('output shape: {}, hidden_state_shape: {}, cell_state_shape: {}, '.format(whole_seq_output.shape,final_memory_state.shape,final_carry_state.shape  ))
    
    # tf.keras.layers.LSTM은 CuDNN Kernel 사용.
    rnn3 = tf.keras.layers.LSTM(4,return_sequences=True, return_state=True, name='encoder')
    initial_state =  rnn3.get_initial_state(inputs)
    whole_seq_output, final_memory_state, final_carry_state = rnn3(inputs,initial_state)
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
    outputs = model(inputs,training=True)
    print(outputs.shape) # return_sequences=False: (batch_size,11)         return_sequences = True --> batch_size,seq_length,11)


def bidirectional_rnn_test():
    batch_size = 3
    seq_length = 5
    input_dim = 7
    hidden_dim = 2
    
    forward_layer = tf.keras.layers.LSTM(hidden_dim, return_sequences=True,return_state=True)
    
    backward_layer = tf.keras.layers.LSTM(hidden_dim, activation='relu', return_sequences=True, return_state=True, go_backwards=True)
    
    
    rnn = tf.keras.layers.Bidirectional(layer=forward_layer,backward_layer=backward_layer)
    
    inputs = tf.random.normal([batch_size, seq_length, input_dim])
    outputs, f_h, f_c, b_h, b_c = rnn(inputs)  # SimpleRNN: output, f_h, b_h           LSTM: output, f_h,f_c, b_h, b_c
    
    print(outputs)

def simple_seq2seq():
    # encoder, decoder 모두를 tf.keras.layers.RNN으로 구현.
    batch_size = 3
    encoder_length = 5
    encoder_input_dim = 7
    hidden_dim = 4
    
    encoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
    encoder = tf.keras.layers.RNN(encoder_cell,return_sequences=False) # RNN
    
    
    
    
    encoder_inputs = tf.random.normal([batch_size, encoder_length, encoder_input_dim])  # Embedding을 거친 data라 가정.
    
    
    encoder_outputs = encoder(encoder_inputs) # encoder의 init_state을 명시적으로 전달하지 않으면, zero값이 들어간다.  ===> (batch_size, hidden_dim)
    
    
    decoder_length = 10
    decoder_input_dim = 7
    decoder_inputs = tf.random.normal([batch_size, decoder_length, decoder_input_dim])  # Embedding을 거친 data라 가정.
    
    
    decoder_multi_layer_flag = True
    
    if decoder_multi_layer_flag:
        # 단순히 list로 쌓아도 되고, tf.keras.layers.StackedRNNCells을 이용해도 된다.
        #decoder_cell = [tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)]
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
        
        decoder = tf.keras.layers.RNN(decoder_cell,return_sequences=True) # RNN
        initial_state = decoder.get_initial_state(decoder_inputs)
        initial_state[0] =  [encoder_outputs,encoder_outputs]  # decoder의 첫번째 layer의 init state를에 encoder output을 넣어준다. 
    else:
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
        decoder = tf.keras.layers.RNN(decoder_cell,return_sequences=True) # RNN
        initial_state =  [encoder_outputs,encoder_outputs]  # (h,c)모두에 encoder_outputs을 넣었다.
    
    decoder_outputs = decoder(decoder_inputs, initial_state)
    print(decoder_outputs)


def simple_seq2seq2():
    # decoder를 tfa.seq2seq.BasicDecoder로 구현.
    # decoder_multi_layer_flag = True or False
    batch_size = 3
    encoder_length = 5
    encoder_input_dim = 7
    hidden_dim = 4
    
    encoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
    encoder = tf.keras.layers.RNN(encoder_cell,return_sequences=False) # RNN
    
    
    
    
    encoder_inputs = tf.random.normal([batch_size, encoder_length, encoder_input_dim])  # Embedding을 거친 data라 가정.
    
    
    encoder_outputs = encoder(encoder_inputs) # encoder의 init_state을 명시적으로 전달하지 않으면, zero값이 들어간다.  ===> (batch_size, hidden_dim)
    
    
    decoder_length = 10
    decoder_input_dim = 11
    decoder_output_dim = 8
    
    decoder_multi_layer_flag = True
    if decoder_multi_layer_flag:
        # cell을 list로 쌓으면 안되고, tf.keras.layers.StackedRNNCells을 사용해야 된다.
        decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
    else:
        decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
    
    
    
    #projection_layer = tf.keras.layers.Dense(decoder_output_dim)
    #projection_layer = MyProjection(decoder_output_dim)
    projection_layer = MyProjection2(decoder_output_dim)
    
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    
    
    decoder_inputs = tf.random.normal([batch_size, decoder_length, decoder_input_dim])  # Embedding을 거친 data라 가정.
    
    if decoder_multi_layer_flag:
        initial_state = decoder_cell.get_initial_state(inputs=decoder_inputs)
        initial_state = ([encoder_outputs,encoder_outputs], initial_state[1])  # decoder의 첫번째 layer에 encoder hidden을 넣어준다.
        
    else:
        initial_state =  [encoder_outputs,encoder_outputs]  # (h,c)모두에 encoder_outputs을 넣었다.
    
    decoder_outputs = decoder(decoder_inputs, initial_state=initial_state,sequence_length=[decoder_length]*batch_size,training=True)
    print(decoder_outputs)



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

    seq_length = x_data.shape[1]
    embedding_dim = 8

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    ##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 
    
    inputs = embedding(x_data)
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
        init_state = decoder_cell.get_initial_state(inputs=inputs)  #inputs=tf.zeros_like(x_data,dtype=tf.float32)로 해도 됨. inputs의 batch_size만 참조하기 때문에
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    
    decoder_method = 1
    if decoder_method==1:
        # tensorflow 1.x에서 tf.contrib.seq2seq.TrainingHelper
        sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
    
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
        outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
    
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
    # BeamSearchDecoder
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

    seq_length = x_data.shape[1]
    embedding_dim = 8
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True) 
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
    
    
    # train용 Sampler로 TrainingSampler 또는 ScheduledEmbeddingTrainingSampler 선택.
    sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
    #sampler = tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler(sampling_probability=0.2)
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    
    for step in range(200):
        with tf.GradientTape() as tape:
            inputs = embedding(x_data)
            if isinstance(sampler, tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler):
                outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True,embedding=embedding.weights)
            else: outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
           
            outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
            logits = outputs.rnn_output
            
            weights = tf.ones(shape=[batch_size,seq_length])
            loss = tfa.seq2seq.sequence_loss(logits,target,weights)
        
        trainable_variables = embedding.trainable_variables + decoder.trainable_variables   # 매번 update되어야 한다.
        grads = tape.gradient(loss,trainable_variables)
        optimizer.apply_gradients(zip(grads,trainable_variables))
        
        if step%10==0:
            print(step, loss.numpy())
    
    
    
    sample_batch_size = 5
    
    decoder_type = 1
    if decoder_type==1:
        # GreedyEmbeddingSampler or SampleEmbeddingSampler()
        
        # sampler 선택 가능.
        sampler = tfa.seq2seq.GreedyEmbeddingSampler()  # alias ---> sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler
        #sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_fn=lambda ids: tf.nn.embedding_lookup(embedding.weights, ids)) # embedding_fn을 넘겨줄 수도 ㅣ있다.
        #sampler = tfa.seq2seq.SampleEmbeddingSampler()
        
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)
        if method==1:
            # single layer
            init_state = decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size, dtype=tf.float32)
        else:
            # multi layer
            init_state = decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size,hidden_dim],dtype=tf.float32))
        
    else:
        # Beam Search
        beam_width=2
        decoder = tfa.seq2seq.BeamSearchDecoder(decoder_cell,beam_width,output_layer=projection_layer,maximum_iterations=seq_length)
        
        # 2가지 방법은 같은 결과를 준다.
        if method==1:
            #init_state = decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size*beam_width, dtype=tf.float32)
            init_state = tfa.seq2seq.tile_batch(decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size, dtype=tf.float32),multiplier=beam_width)
        else:
            #init_state = decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size*beam_width,hidden_dim],dtype=tf.float32))
            init_state = tfa.seq2seq.tile_batch(decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size,hidden_dim],dtype=tf.float32)),multiplier=beam_width)
        
    outputs, last_state, last_sequence_lengths = decoder(embedding.weights,initial_state=init_state,
                                                         start_tokens=tf.tile([SOS_token], [sample_batch_size]), end_token=EOS_token,training=False) 
    
    
    if decoder_type==1:
        result = tf.argmax(outputs.rnn_output,axis=-1).numpy()
        
        print(result)
        for i in range(sample_batch_size):
            print(''.join( index_to_char[a] for a in result[i] if a != EOS_token))

    else:
        result = outputs.predicted_ids.numpy()
        print(result.shape)
        for i in range(sample_batch_size):
            print(i,)
            for j in range(beam_width):
                print(''.join( index_to_char[a] for a in result[i,:,j] if a != EOS_token))



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

    seq_length = x_data.shape[1]
    embedding_dim = 8

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
    decoder_cell = tfa.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=13,initial_cell_state=init_state,output_attention=True,alignment_history=True)
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    attention_init_state = decoder_cell.get_initial_state(inputs = None, batch_size = batch_size, dtype=tf.float32)  # inputs의 역할은 없느데.. .source보면.
    
    attention_init_state2 = tfa.seq2seq.AttentionWrapperState(list(attention_init_state.cell_state),attention_init_state.attention,attention_init_state.alignments,
                                                              attention_init_state.alignment_history,attention_init_state.attention_state)
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    

    outputs, last_state, last_sequence_lengths = decoder(decoder_input,initial_state=attention_init_state2, sequence_length=[seq_length]*batch_size)
    logits = outputs.rnn_output
    
    print(logits.shape)
    

    alignment_stack = last_state.alignment_history.stack().numpy()
    print("alignment_history: ", alignment_stack.shape)  # (seq_length, batch_size, encoder_length)
    
    plt.imshow(alignment_stack[:,0,:], cmap='hot',interpolation='nearest')
    plt.show()


def InferenceSampler_test():
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
    
    seq_length = x_data.shape[1]
    embedding_dim = 8
    
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
    ##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 
    
    
    target = tf.convert_to_tensor(y_data)
    
    # Decoder

    # single layer RNN
    decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
    # decoder init state:
    
    #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
    init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    
    sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    
    for step in range(500):
        with tf.GradientTape() as tape:
            inputs = embedding(x_data)
            if isinstance(sampler, tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler):
                outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True,embedding=embedding.weights)
            else: outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)
            
            logits = outputs.rnn_output
            
            weights = tf.ones(shape=[batch_size,seq_length])
            loss = tfa.seq2seq.sequence_loss(logits,target,weights)
        
        trainable_variables = embedding.trainable_variables + decoder.trainable_variables   # 매번 update되어야 한다.
        grads = tape.gradient(loss,trainable_variables)
        optimizer.apply_gradients(zip(grads,trainable_variables))
        
        if step%10==0:
            print(step, loss.numpy())
    
    
    
    
    sample_batch_size = 5
    
    # InferenceSampler를 사용해 보자.
    # GreedyEmbedding Sampler를 구현했다. 
    sampler = tfa.seq2seq.InferenceSampler(sample_fn = lambda outputs: tf.argmax(outputs, axis=-1, output_type=tf.int32), 
                                           sample_shape=[], sample_dtype=tf.int32,
                                           end_fn = lambda sample_ids: tf.equal(sample_ids, EOS_token),
                                           next_inputs_fn = lambda ids: tf.nn.embedding_lookup(embedding.weights, ids))
    
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)

    init_state = decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size, dtype=tf.float32)

    
    

    start_inputs = tf.nn.embedding_lookup(embedding.weights, tf.tile([SOS_token], [sample_batch_size]))  # embedding된 것을 넘겨주어야 한다.
    outputs, last_state, last_sequence_lengths = decoder(start_inputs,initial_state=init_state,training=False) 
    
    result = tf.argmax(outputs.rnn_output,axis=-1).numpy()
    
    print(result)
    for i in range(sample_batch_size):
        print(''.join( index_to_char[a] for a in result[i] if a != EOS_token))




if __name__ == '__main__':
    #simple_rnn()
    #simple_rnn2()
    #bidirectional_rnn_test()
    #simple_seq2seq()
    #simple_seq2seq2()
    #seq_loss_test()
    #decoder_test()
    decoder_train_test()
    #attention_test()
    #InferenceSampler_test()
    print('Done')

