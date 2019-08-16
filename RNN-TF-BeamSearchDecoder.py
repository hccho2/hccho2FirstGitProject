# coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
tf.reset_default_graph()

#########################################
class BeamDecode():
    def __init__(self,batch_size,hidden_dim,output_dim,embedding_dim,seq_length=None,is_training=True):
        
        with tf.variable_scope('DynamicDecoder',reuse = tf.AUTO_REUSE) as scope:
            if not is_training:
                seq_length = 1
            self.X = tf.placeholder(tf.int32,shape=[None,None])   # batch_size, seq_length
            self.Y = tf.placeholder(tf.int32,shape=[None,None])
            
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
        
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            embedding = tf.get_variable("embedding", shape=[output_dim,embedding_dim],initializer=init,dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X) # batch_size  x seq_length x embedding_dim
        
            

            if is_training:
                initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) 
                helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state)
                
                self.outputs, self.last_state, self.last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=20)
    
      
                weights = tf.ones(shape=[batch_size,seq_length])
                self.loss =   tf.contrib.seq2seq.sequence_loss(logits=self.outputs.rnn_output, targets=self.Y, weights=weights)
                self.opt = tf.train.AdamOptimizer(0.1).minimize(self.loss)                
                
            else:
                beam_width = 3
                SOS_token=0
                EOS_token = output_dim-1
                initial_state = tf.contrib.seq2seq.tile_batch(tf.zeros([batch_size,hidden_dim],tf.float32), multiplier=beam_width)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell,embedding=embedding,start_tokens=tf.tile([SOS_token], [batch_size]),
                                                               end_token=EOS_token,initial_state=initial_state,beam_width=beam_width)
                self.outputs, self.last_state, self.last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,maximum_iterations=20)  # impute_finished=True ==> error occurs
            
                print('aa')

        
#######################################

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
sess = tf.Session()

output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =6
seq_length = x_data.shape[1]
embedding_dim = 8

model = BeamDecode(batch_size=batch_size,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,seq_length=seq_length,is_training=True)
test_model = BeamDecode(batch_size=1,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,is_training=False)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(2000):
    loss , _ = sess.run([model.loss,model.opt],feed_dict={model.X: x_data,model.Y: y_data})
    if i % 100 == 0:
        print(i, 'loss: {}'.format(loss))




result = sess.run(test_model.outputs.predicted_ids)

# result_all = [[index_to_char[x[0]],index_to_char[x[1]]] for x in result[0]]
# result_all= list(map(list, zip(*result_all)))

result_all = [list(map(lambda a: index_to_char[a] ,x)) for x in result[0].T]

print(result_all)


#########################################
#########################################
#########################################
#########################################
AttentionWrapper를 사용하는 경우는 많이 복잡해진다.

class BeamDecode():
    def __init__(self,batch_size,hidden_dim,output_dim,embedding_dim,h0,seq_length=None,is_training=True):
        
        with tf.variable_scope('DynamicDecoder',reuse = tf.AUTO_REUSE) as scope:
            if not is_training:
                seq_length = 1
            self.X = tf.placeholder(tf.int32,shape=[None,None])   # batch_size, seq_length
            self.Y = tf.placeholder(tf.int32,shape=[None,None])
            
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
        
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            embedding = tf.get_variable("embedding", shape=[output_dim,embedding_dim],initializer=init,dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X) # batch_size  x seq_length x embedding_dim
            
            
            encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32)) # 20: encoder sequence length, 30: encoder hidden dim
            encoder_state = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,30]).astype(np.float32))
            input_lengths = tf.convert_to_tensor([20] * batch_size)
            
            # attention_initial_state를 일률적으로 0으로 주던지. encoder의 결과를 받아, batch data마다 다른 값을 같든지....여기서는 모든 batch에 대하여 실험적으로 0이아닌 같은 값을 갖게 해본다.
            #attention_initial_state = cell.zero_state(batch_size, tf.float32)
            attention_initial_state =  h0 # tf.tile(h0,[batch_size,1]) #   tf.convert_to_tensor(np.random.normal(0,1,size=(batch_size,output_dim)).astype(np.float32)) # 
            if not is_training:
                beam_width = 3
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                input_lengths = tf.contrib.seq2seq.tile_batch(input_lengths, multiplier=beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)      
                attention_initial_state = tf.contrib.seq2seq.tile_batch(attention_initial_state, multiplier=beam_width)  # <tf.Tensor 'DynamicDecoder_1/Tile:0' shape=(2, 6) dtype=float32> ===> <tf.Tensor 'DynamicDecoder_1/tile_batch_3/Reshape:0' shape=(6, 6) dtype=float32>
            
                
            
            
            
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=False)
            
            cell= tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=13,initial_cell_state=attention_initial_state,
                                                   alignment_history=True,output_attention=True)
            
            output_layer = Dense(output_dim, name='output_projection')
            
            if is_training:
                initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) 
                self.x = initial_state
                helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)
                
                self.outputs, self.last_state, self.last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=20)
    
      
                weights = tf.ones(shape=[batch_size,seq_length])
                self.loss =   tf.contrib.seq2seq.sequence_loss(logits=self.outputs.rnn_output, targets=self.Y, weights=weights)
                self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)                
                
            else:
                
                SOS_token=0
                EOS_token = output_dim-1
                
                # 여기 초기 값을 위에서 정의된 값으로 잘 가져와야 된다.
                # attention_initial_state: <tf.Tensor 'DynamicDecoder_1/tile_batch_3/Reshape:0' shape=(6, 6) dtype=float32> 로 부터
                # initial_state: AttentionWrapperState(cell_state=<tf.Tensor 'DynamicDecoder_1/tile_batch_3/Reshape:0' shape=(6, 6) dtype=float32>, attention=<tf.Tensor 'DynamicDecoder_1/AttentionWrapperZeroState/zeros_2:0' shape=(6, 13) dtype=float32>, time=<tf.Tensor 'DynamicDecoder_1/AttentionWrapperZeroState/zeros_1:0' shape=() dtype=int32>, alignments=<tf.Tensor 'DynamicDecoder_1/AttentionWrapperZeroState/zeros:0' shape=(6, 20) dtype=float32>, alignment_history=<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x00000183D039CCF8>, attention_state=<tf.Tensor 'DynamicDecoder_1/AttentionWrapperZeroState/zeros_3:0' shape=(6, 20) dtype=float32>)
                # 정리하면, cell_state를 가지고, AttentinWrapperState를 만드는 것이다.
                initial_state = cell.zero_state(batch_size * beam_width,tf.float32).clone(cell_state=attention_initial_state)
                self.x = initial_state
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell,embedding=embedding,start_tokens=tf.tile([SOS_token], [batch_size]),
                                                               end_token=EOS_token,initial_state=initial_state,beam_width=beam_width,output_layer=output_layer)
                self.outputs, self.last_state, self.last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,maximum_iterations=20)  # impute_finished=True ==> error occurs
            
                print('aa')

        
#######################################

vocab_size = 6
SOS_token = 0
EOS_token = 5

x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)

index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}


Y = tf.convert_to_tensor(y_data)
print("data shape: ", x_data.shape)
sess = tf.Session()

output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =6
seq_length = x_data.shape[1]
embedding_dim = 8



h0 = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,output_dim]).astype(np.float32))
model = BeamDecode(batch_size=batch_size,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,h0=h0,seq_length=seq_length,is_training=True)

# h0의 크기가 맞아야 한다.
test_batch_size=2
test_model = BeamDecode(batch_size=test_batch_size,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,h0=h0[:test_batch_size],is_training=False)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(2000):
    loss , _ = sess.run([model.loss,model.opt],feed_dict={model.X: x_data,model.Y: y_data})
    if i % 100 == 0:
        print(i, 'loss: {}'.format(loss))


print(sess.run([model.x[0],test_model.x[0]]))  # 두 array 행의 값이 모두 같은 값임을 알 수 있다. BeamSearchDecoder의 init state값이 잘 전달된 것을 알 수 있다.

result = sess.run(test_model.outputs.predicted_ids)
print(result)
# result_all = [[index_to_char[x[0]],index_to_char[x[1]]] for x in result[0]]
# result_all= list(map(list, zip(*result_all)))

result_all = [list(map(lambda a: index_to_char[a] ,x)) for x in result[0].T]

print(result_all)

##################################################################################
https://github.com/tensorflow/tensorflow/issues/11904

import tensorflow as tf
from tensorflow.python.layers.core import Dense


BEAM_WIDTH = 5
BATCH_SIZE = 128


# INPUTS
X = tf.placeholder(tf.int32, [BATCH_SIZE, None])
Y = tf.placeholder(tf.int32, [BATCH_SIZE, None])
X_seq_len = tf.placeholder(tf.int32, [BATCH_SIZE])
Y_seq_len = tf.placeholder(tf.int32, [BATCH_SIZE])


# ENCODER         
encoder_out, encoder_state = tf.nn.dynamic_rnn(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128), 
    inputs = tf.contrib.layers.embed_sequence(X, 10000, 128),
    sequence_length = X_seq_len,
    dtype = tf.float32)


# DECODER COMPONENTS
Y_vocab_size = 10000
decoder_embedding = tf.Variable(tf.random_uniform([Y_vocab_size, 128], -1.0, 1.0))
projection_layer = Dense(Y_vocab_size)


# ATTENTION (TRAINING)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units = 128, 
    memory = encoder_out,
    memory_sequence_length = X_seq_len)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128),
    attention_mechanism = attention_mechanism,
    attention_layer_size = 128)


# DECODER (TRAINING)
training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs = tf.nn.embedding_lookup(decoder_embedding, Y),
    sequence_length = Y_seq_len,
    time_major = False)
training_decoder = tf.contrib.seq2seq.BasicDecoder(
    cell = decoder_cell,
    helper = training_helper,
    initial_state = decoder_cell.zero_state(BATCH_SIZE,tf.float32).clone(cell_state=encoder_state),
    output_layer = projection_layer)
training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder = training_decoder,
    impute_finished = True,
    maximum_iterations = tf.reduce_max(Y_seq_len))
training_logits = training_decoder_output.rnn_output


# BEAM SEARCH TILE
encoder_out = tf.contrib.seq2seq.tile_batch(encoder_out, multiplier=BEAM_WIDTH)
X_seq_len = tf.contrib.seq2seq.tile_batch(X_seq_len, multiplier=BEAM_WIDTH)
encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=BEAM_WIDTH)


# ATTENTION (PREDICTING)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units = 128, 
    memory = encoder_out,
    memory_sequence_length = X_seq_len)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128),
    attention_mechanism = attention_mechanism,
    attention_layer_size = 128)


# DECODER (PREDICTING)
predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    cell = decoder_cell,
    embedding = decoder_embedding,
    start_tokens = tf.tile(tf.constant([1], dtype=tf.int32), [BATCH_SIZE]),
    end_token = 2,
    initial_state = decoder_cell.zero_state(BATCH_SIZE * BEAM_WIDTH,tf.float32).clone(cell_state=encoder_state),
    beam_width = BEAM_WIDTH,
    output_layer = projection_layer,
    length_penalty_weight = 0.0)
predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder = predicting_decoder,
    impute_finished = False,
    maximum_iterations = 2 * tf.reduce_max(Y_seq_len))
predicting_logits = predicting_decoder_output.predicted_ids[:, :, 0]

print('successful')
