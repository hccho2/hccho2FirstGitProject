# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
tf.reset_default_graph()


class DynamicRNN():
    def __init__(self,batch_size,hidden_dim,output_dim,embedding_dim,seq_length=1,is_training=True):
        
        with tf.variable_scope('DynamicRNN',reuse = tf.AUTO_REUSE) as scope:
            if not is_training:
                seq_length = 1
            self.X = tf.placeholder(tf.int32,shape=[None,None])   # batch_size, seq_length
            self.Y = tf.placeholder(tf.int32,shape=[None,None])
            
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
        
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            embedding = tf.get_variable("embedding", shape=[output_dim,embedding_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X) # batch_size  x seq_length x embedding_dim
            
            if is_training:
                self.initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) 
            else:
                self.initial_state = tf.placeholder(tf.float32,shape=[batch_size,hidden_dim])
            self.outputs, self.last_state = tf.nn.dynamic_rnn(cell,inputs,sequence_length=[seq_length]*batch_size,initial_state=self.initial_state)    
    
            weights = tf.ones(shape=[batch_size,seq_length])
            self.loss =   tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.Y, weights=weights)
            self.opt = tf.train.AdamOptimizer(0.1).minimize(self.loss)
        
def dynamic_rnn_test():

    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    Y = tf.convert_to_tensor(y_data)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =6
    seq_length = x_data.shape[1]
    embedding_dim = 8

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    with tf.variable_scope('test') as scope:
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) 
        outputs, last_state = tf.nn.dynamic_rnn(cell,inputs,sequence_length=[seq_length]*batch_size,initial_state=initial_state)    

        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
      
        p = sess.run(tf.nn.softmax(outputs)).reshape(-1,output_dim)
        print("loss: {:20.6f}".format(sess.run(loss)))
        print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )


def dynamic_rnn_class_test():
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
    
    model = DynamicRNN(batch_size=batch_size,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,seq_length=seq_length,is_training=True)
    test_model = DynamicRNN(batch_size=1,hidden_dim=hidden_dim,output_dim=vocab_size,embedding_dim=embedding_dim,seq_length=1,is_training=False)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    for i in range(1000):
        loss , _ = sess.run([model.loss,model.opt],feed_dict={model.X: x_data,model.Y: y_data})
        if i % 100 == 0:
            print(i, 'loss: {}'.format(loss))
    
    
    
    x_data = np.array([[SOS_token]], dtype=np.int32)
    result_all = []
    initial_state = np.zeros([1,hidden_dim])
    for i in range(20):
        result,initial_state = sess.run([test_model.outputs,test_model.last_state], feed_dict={test_model.X: x_data,test_model.initial_state: initial_state})
        result = np.argmax(result,axis=-1)
        x_data = result
        result_all.append(index_to_char[result[0][0]])
        if result[0][0] == EOS_token:
            break
        #print(result)
        
    
    print(result_all)
    
    
if __name__ == '__main__':
    #dynamic_rnn_test()
    dynamic_rnn_class_test()
    print('Done')