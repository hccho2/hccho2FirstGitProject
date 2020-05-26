# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

from tensorflow.python.layers.core import Dense
tf.reset_default_graph()
class DynamicDecode():
    def __init__(self,batch_size,hidden_dim,output_dim,embedding_dim,seq_length=None,is_training=True):
        """
        batch_size를 입력받는 것이 좋아 보이지 않는다.
        inputs(tensor)가 있다면, batch_size = tf.shape(inputs)[0] <---- 이것도 tensor
        Helper에 들어가는 이것도 np.array([seq_length]*batch_size)   --> tf.tile([seq_length],[batch_size]) 이런 식으로 대체할 수 있다.
        """
        with tf.variable_scope('DynamicDecoder',reuse = tf.AUTO_REUSE) as scope:
            if not is_training:
                seq_length = 1
            self.X = tf.placeholder(tf.int32,shape=[None,None])   # batch_size, seq_length
            self.Y = tf.placeholder(tf.int32,shape=[None,None])
            
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
        
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            embedding = tf.get_variable("embedding", shape=[output_dim,embedding_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X) # batch_size  x seq_length x embedding_dim
        
            initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) 

            if is_training:
                helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
            else:
                SOS_token=0
                EOS_token = output_dim-1
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state)
            self.outputs, self.last_state, self.last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=20)

  
            weights = tf.ones(shape=[batch_size,seq_length])
            self.loss =   tf.contrib.seq2seq.sequence_loss(logits=self.outputs.rnn_output, targets=self.Y, weights=weights)
            self.opt = tf.train.AdamOptimizer(0.1).minimize(self.loss)
            
            
def dynamic_decode_test():

    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 0
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = False
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        
        method = 1
        if method == 0:
            cells = []
            for _ in range(num_layers):
                cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
                #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
                #cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)  # init_state_flag==0 으로 해야 됨.
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)    
        else:
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim,num_proj=7)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
    
        # tf.contrib.rnn.OutputProjectionWrapper  마지막에 FC layer를 하나 더 추가하는 효과. 아래에서 적용하는 Dense보다 앞에 적용된다. Dense가 있기 때문에 OutputProjectionWrapper 또는 Dense로 처리 가능함
        # FC layer를 multiple로 적용하려면 OutputProjectionWrapper을 사용해야 함.
        if False:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,13,activation=tf.nn.relu)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,17)
    
        if init_state_flag==0:
             initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        else:
            if state_tuple_mode:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                # 첫번째 layer의 c=0, h=h0, 두번째 layer의 c=0, h=0, ....
                initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)
                
            else:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
            #helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([[2],[4],[6]]).reshape(-1))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
    
        output_layer = Dense(output_dim, name='output_projection')
        #output_layer = None
        
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        # last_state는 num_layers 만큼 나온다.
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
    
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        if output_layer is not None:
            print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
            print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
    
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)   #(18,5) = (batch_size x seq_length, vocab_size)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )

def dynamic_decode_helpertest():

    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 0
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        
        method = 1
        if method == 0:
            cells = []
            for _ in range(num_layers):
                cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
                #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
                #cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)    
        else:
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim,num_proj=7)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
    
        # tf.contrib.rnn.OutputProjectionWrapper  마지막에 FC layer를 하나 더 추가하는 효과. 아래에서 적용하는 Dense보다 앞에 적용된다. Dense가 있기 때문에 OutputProjectionWrapper 또는 Dense로 처리 가능함
        # FC layer를 multiple로 적용하려면 OutputProjectionWrapper을 사용해야 함.
        if False:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,13)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,17)
    
        if init_state_flag==0:
             initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        else:
            if state_tuple_mode:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)
                
            else:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
        if train_mode:
            #helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs, np.array([seq_length]*batch_size),embedding,0.3)
            #helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs, np.array([seq_length]*batch_size),0.3)   # output dim(embedding 전),input dim이 잘 맞아야 한다. 예에서 embedding_dim=vocab_size
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
            #helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
    
        output_layer = Dense(output_dim, name='output_projection')
        #output_layer = None
        
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        # last_state는 num_layers 만큼 나온다.
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
    
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        
        # SampleEmbeddingHelper: randomness가 있다.
        o,o2 = sess.run([outputs.rnn_output,tf.argmax(outputs.rnn_output,axis=-1)])  #batch_size, seq_length, outputs
        print("\n outputs---",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        if output_layer is not None:
            print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
            print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
    
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)   #(18,5) = (batch_size x seq_length, vocab_size)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )

def attention_test():
    # BasicRNNCell을 single로 쌓아 attention 적용
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    alignment_history_flag = True   # True이면 initial_state나 last state를 sess.run 하면 안됨. alignment_history가 function이기 때문에...
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn cell
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        # encoder_outputs = tf.ones([batch_size,20,30])
        # encoder_outputs의 (N,T,xx). 여기서 하나의 미니 배치에서는 길이 T로 만들어져 있다(padding). 그러나 미니 배치마다 길이가 달라진다.
        encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32)) # 20: encoder sequence length, 30: encoder hidden dim
        
        #input_lengths = [20]*batch_size
        input_lengths = [5,10,20]  # encoder에 padding 같은 것이 있을 경우, attention을 주지 않기 위해
        
        # attention mechanism  # num_units = Na = 11
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=False)
        #attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        # LuongAttention에서는 num_units이 임의로 들어가면 안되고, decoder의 hidden_dim과 일치해야 한다
        #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hidden_dim, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        
        # output_attention = True(default) ==> 이면 output으로 attention이 나가고, False이면 cell의 output이 나간다
        # attention_layer_size = N_l
        
        attention_initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=13,initial_cell_state=attention_initial_state,
                                                   alignment_history=alignment_history_flag,output_attention=True)

        # 여기서 zero_state를 부르면, 위의 attentionwrapper에서 넝허준 attention_initial_state를 가져온다. 즉, AttentionWrapperState.cell_state에는 넣어준 값이 들어있다.
        initial_state = cell.zero_state(batch_size, tf.float32) # AttentionWrapperState
 
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
     
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
     
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
     
     
        
        
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)
        
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            loss_,_ =sess.run([loss,opt])
            print("{} loss: = {}".format(i,loss_))
        
        if alignment_history_flag ==False:
            print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
     
        print("\n\nlast_state: ",last_state)
        if alignment_history_flag == False:
            print(sess.run(last_state)) # batch_size, hidden_dim
        else:
            print("alignment_history: ", last_state.alignment_history.stack())
            alignment_history_ = sess.run(last_state.alignment_history.stack())
            print(alignment_history_)
            print("alignment_history sum: ",np.sum(alignment_history_,axis=-1))
            
            print("cell_state: ", sess.run(last_state.cell_state))
            print("attention: ", sess.run(last_state.attention))
            print("time: ", sess.run(last_state.time))
            
            alignments_ = sess.run(last_state.alignments)
            print("alignments: ", alignments_)
            print('alignments sum: ', np.sum(alignments_,axis=1))   # alignments의 합이 1인지 확인
            print("attention_state: ", sess.run(last_state.attention_state))

     
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
         
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
     
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )            

    """
    <tf.Variable 'test/embedding:0' shape=(6, 8) dtype=float32_ref>, 
    <tf.Variable 'test/memory_layer/kernel:0' shape=(30, 11) dtype=float32_ref>,                                                 Wm: (encoder_hidden_dim = 30, num_units=11)
    <tf.Variable 'test/decoder/attention_wrapper/basic_rnn_cell/kernel:0' shape=(28, 7) dtype=float32_ref>,                      28 = embedding_dim(=input dim = 8) + attention_layer_size(N_l=13) + hidden_dim(7)
    <tf.Variable 'test/decoder/attention_wrapper/basic_rnn_cell/bias:0' shape=(7,) dtype=float32_ref>, 
    <tf.Variable 'test/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel:0' shape=(7, 11) dtype=float32_ref>,      Wq: (hidden_dim, num_units)
    <tf.Variable 'test/decoder/attention_wrapper/bahdanau_attention/attention_v:0' shape=(11,) dtype=float32_ref>,               va
    <tf.Variable 'test/decoder/attention_wrapper/attention_layer/kernel:0' shape=(37, 13) dtype=float32_ref>,                    Wa: 37 = encoder hidden_dim(30)+ decoder_hidden_dim(7), attention_layer_size(13)
    
    <tf.Variable 'test/decoder/output_projection/kernel:0' shape=(13, 6) dtype=float32_ref>, 
    <tf.Variable 'test/decoder/output_projection/bias:0' shape=(6,) dtype=float32_ref>

    
    """



def attention_multicell_test():
    # BasicRNNCell을 multi로 쌓아 attention 적용. multi에서는 제일 아래 layer에 attention을 적용한다
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make multi-rnn cell
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        encoder_outputs = tf.ones([batch_size,20,30])
        input_lengths = [20]*batch_size
        # attention mechanism
        attention_initial_state = cell.zero_state(batch_size, tf.float32)  # 다른 값을 줄수도 있다.
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,initial_cell_state=attention_initial_state, attention_layer_size=13)  # AttentionWrapperState를 return한다.


        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수   ==> AttentionWrapperState class object를 return한다.
  
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
      
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
      
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
      
      
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
      
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
      
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
          
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
      
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) ) 



def dynamic_decode_class_test():
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    #x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    #y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    
    index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}
    x_data = np.array([[SOS_token, 1, 2, 3, 3, 4]], dtype=np.int32)
    y_data = np.array([[1, 2, 3, 3, 4,EOS_token]],dtype=np.int32)
    
    Y = tf.convert_to_tensor(y_data)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =6
    seq_length = x_data.shape[1]
    embedding_dim = 8
    
    model = DynamicDecode(batch_size=batch_size,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,seq_length=seq_length,is_training=True)
    test_model = DynamicDecode(batch_size=1,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,is_training=False)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    for i in range(2000):
        loss , _ = sess.run([model.loss,model.opt],feed_dict={model.X: x_data,model.Y: y_data})
        if i % 100 == 0:
            print(i, 'loss: {}'.format(loss))
    
    
    

    result = sess.run(test_model.outputs.rnn_output)
    result = np.argmax(result,axis=-1)
    result_all = [index_to_char[x] for x in result[0]]
        
    
    print(result_all)





def attention_keras_test():
    # tf.keras.layers.SimpleRNNCell를 이용하기
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    alignment_history_flag = True   # True이면 initial_state나 last state를 sess.run 하면 안됨. alignment_history가 function이기 때문에...
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn cell
        cell = tf.keras.layers.SimpleRNNCell(units=hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        #encoder_outputs = tf.ones([batch_size,20,30])
        encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32)) # 20: encoder sequence length, 30: encoder hidden dim
        
        #input_lengths = [20]*batch_size
        input_lengths = [5,10,20]  # encoder에 padding 같은 것이 있을 경우, attention을 주지 않기 위해
        
        # attention mechanism  # num_units = Na = 11
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=False)
        #attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        # LuongAttention에서는 num_units이 임의로 들어가면 안되고, decoder의 hidden_dim과 일치해야 한다
        #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hidden_dim, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        
        # output_attention = True(default) ==> 이면 output으로 attention이 나가고, False이면 cell의 output이 나간다
        # attention_layer_size = N_l
        
        attention_initial_state = [cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)]
        
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=13,initial_cell_state=attention_initial_state,
                                                   alignment_history=alignment_history_flag,output_attention=True)

        # 여기서 zero_state를 부르면, 위의 attentionwrapper에서 넝허준 attention_initial_state를 가져온다. 즉, AttentionWrapperState.cell_state에는 넣어준 값이 들어있다.
        initial_state = cell.zero_state(batch_size, tf.float32) # AttentionWrapperState
 
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size,dtype=np.int32))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
     
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
     
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
     
     
        
        
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)
        
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            loss_,_ =sess.run([loss,opt])
            print("{} loss: = {}".format(i,loss_))
        
        if alignment_history_flag ==False:
            print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
     
        print("\n\nlast_state: ",last_state)
        if alignment_history_flag == False:
            print(sess.run(last_state)) # batch_size, hidden_dim
        else:
            print("alignment_history: ", last_state.alignment_history.stack())
            alignment_history_ = sess.run(last_state.alignment_history.stack())
            print(alignment_history_)
            print("alignment_history sum: ",np.sum(alignment_history_,axis=-1))
            
            print("cell_state: ", sess.run(last_state.cell_state))
            print("attention: ", sess.run(last_state.attention))
            print("time: ", sess.run(last_state.time))
            
            alignments_ = sess.run(last_state.alignments)
            print("alignments: ", alignments_)
            print('alignments sum: ', np.sum(alignments_,axis=1))   # alignments의 합이 1인지 확인
            print("attention_state: ", sess.run(last_state.attention_state))

     
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
         
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
     
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )    




          
if __name__ == '__main__':
    #dynamic_decode_test()
    dynamic_decode_helpertest()
    #dynamic_decode_class_test()
    #attention_test()
    #attention_multicell_test()
    
    print('Done')















