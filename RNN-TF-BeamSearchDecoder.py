# coding: utf-8
import tensorflow as tf
import numpy as np
tf.reset_default_graph()


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

