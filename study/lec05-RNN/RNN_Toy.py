#  coding: utf-8
"""
간단한 모델이라 gpu가 cpu보다 느림.

"""

import copy, numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense
np.random.seed(0)

def my_ManyParallel_uint2bits(in_intAr,Nbits):
    ''' convert (numpyarray of uint => array of Nbits bits) for many bits in parallel'''
    inSize_T= in_intAr.shape
    in_intAr_flat=in_intAr.flatten()
    out_NbitAr= np.zeros((len(in_intAr_flat),Nbits))
    for iBits in range(Nbits):
        out_NbitAr[:,iBits]= (in_intAr_flat>>iBits)&1
    out_NbitAr= out_NbitAr.reshape(inSize_T+(Nbits,))
    return out_NbitAr 

class BinaryAddition():
    def __init__(self,binary_dim,hidden_dim=16):
        self.binary_dim = binary_dim  #length
        self.hidden_dim = hidden_dim
        
        self.build()

    def build(self):
        
        
        self.X = tf.placeholder(tf.float32,shape=[None,self.binary_dim,2])
        self.Y = tf.placeholder(tf.float32,shape=[None,self.binary_dim])
        
        batch_size = tf.shape(self.X)[0]
        seq_length = tf.shape(self.X)[1]        
        
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_dim)
        initial_state = cell.zero_state(batch_size, tf.float32)
        

        
        simple_mode = True
        if simple_mode:
            
            # outprojection을 2가지 방법으로...
            Mode = 0 
            if Mode == 0:
                cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1,activation=tf.nn.sigmoid)
                self.outputs, _states = tf.nn.dynamic_rnn(cell,self.X,initial_state=initial_state,dtype=tf.float32)
                self.loss = tf.nn.l2_loss(tf.reshape(self.outputs,shape=[batch_size,-1])-self.Y)
            else:
                self.outputs, _states = tf.nn.dynamic_rnn(cell,self.X,initial_state=initial_state,dtype=tf.float32)
                self.outputs = tf.layers.dense(self.outputs,1,activation=tf.nn.sigmoid)
                self.loss = tf.nn.l2_loss(tf.reshape(self.outputs,shape=[batch_size,-1])-self.Y)


        else:
        
            
            helper = tf.contrib.seq2seq.TrainingHelper(self.X, tf.tile([self.binary_dim],[batch_size]))
    
            output_layer = Dense(1,activation=tf.nn.sigmoid, name='output_projection')    
            
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)  
            self.outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True)
            
            self.outputs= self.outputs.rnn_output
            
            self.loss = tf.nn.l2_loss(tf.reshape(self.outputs,shape=[batch_size,-1])-self.Y)
               
class BinaryMultiply():
    def __init__(self,binary_dim,hidden_dim=256):
        self.binary_dim = binary_dim  #length
        self.hidden_dim = hidden_dim
        
        self.build()

    def build(self):
        
        
        self.X = tf.placeholder(tf.float32,shape=[None,self.binary_dim,2])
        self.Y = tf.placeholder(tf.float32,shape=[None,self.binary_dim])
        
        batch_size = tf.shape(self.X)[0]
        seq_length = tf.shape(self.X)[1]        
        
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        

        
        simple_mode = True

            
        self.outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,self.X,
                                                      initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,dtype=tf.float32)
        
        self.outputs = tf.concat(self.outputs, axis=2)
        
        self.outputs = tf.layers.dense(self.outputs,1,activation=tf.nn.sigmoid)
        self.loss = tf.nn.l2_loss(tf.reshape(self.outputs,shape=[batch_size,-1])-self.Y)

def run():
    # a,b --> a+b 를 구하는 모델
    with tf.device('/cpu:0'):
        batch_size=2
        binary_dim = 8
        int2binary = {}  # {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0, 1], ... }
    
        
        largest_number = pow(2,binary_dim)
        binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
        for i in range(largest_number):
            int2binary[i] = binary[i]
        
        model = BinaryAddition(binary_dim)
        opt = tf.train.AdamOptimizer(0.001).minimize(model.loss)
    
    
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            x = np.zeros([batch_size,binary_dim,2])
            y = np.zeros([batch_size,binary_dim])
            for i in range(30000):    
                # generate a simple addition problem (a + b = c)
                a_int = np.random.randint(largest_number/2,size=(batch_size,2)) # int version
                
                
                for j in range(batch_size):
                    x[j] = np.vstack([int2binary[a_int[j,0]],int2binary[a_int[j,1]]]).T[::-1,:]
                    y[j] = int2binary[a_int[j,0]+a_int[j,1]][::-1]
            
    
                sess.run([opt],feed_dict={model.X: x, model.Y: y})
                
                
                if i% 1000 == 0:
                    loss,output = sess.run([model.loss,model.outputs],feed_dict={model.X: x, model.Y: y})
                
                    pred = 1*(output.reshape(batch_size,-1)>0.5)
                    print('step {}: loss: {}'.format(i,loss))
                    print('Pred:\n', pred)
                    print('True:\n', y.astype(np.int32))
                    
                    out = 0
                    for index,k in enumerate(pred.T):
                        out += k*pow(2,index)
                    
                    print (str(a_int[:,0]) + " + " + str(a_int[:,1]) + " -> " + str(out), ( a_int[:,0] + a_int[:,1] == out))


def run2():
    # (a,b) --> a+3*b를 구하는 모델
    with tf.device('/cpu:0'):
        batch_size=2
        binary_dim = 10
        int2binary = {}  # {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0, 1], ... }
    
        
        largest_number = pow(2,binary_dim)
        A=np.arange(largest_number).astype('uint16')
        binary = my_ManyParallel_uint2bits(A,binary_dim)  # 왼쪽이 낮은 자리
        for i in range(largest_number):
            int2binary[i] = binary[i]
        
        model = BinaryAddition(binary_dim)
        opt = tf.train.AdamOptimizer(0.01).minimize(model.loss)
    
    
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            x = np.zeros([batch_size,binary_dim,2])
            y = np.zeros([batch_size,binary_dim])
            for i in range(30000):    
                # generate a simple addition problem (a + 2*b = c)
                a_int = np.random.randint(largest_number/4,size=(batch_size,2)) # int version
                
                
                for j in range(batch_size):
                    x[j] = np.vstack([int2binary[a_int[j,0]],int2binary[a_int[j,1]]]).T
                    y[j] = int2binary[a_int[j,0]+3*a_int[j,1]]
            
    
                sess.run([opt],feed_dict={model.X: x, model.Y: y})
                
                
                if i% 1000 == 0:
                    loss,output = sess.run([model.loss,model.outputs],feed_dict={model.X: x, model.Y: y})
                
                    pred = 1*(output.reshape(batch_size,-1)>0.5)
                    print('step {}: loss: {}'.format(i,loss))
                    print('Pred:\n', pred)
                    print('True:\n', y.astype(np.int32))
                    
                    out = 0
                    for index,k in enumerate(pred.T):
                        out += k*pow(2,index)
                    
                    print (str(a_int[:,0]) + " + 3 x " + str(a_int[:,1]) + " -> " + str(out), ( a_int[:,0] + 3*a_int[:,1] == out))
                    
def run3():
    # (a,b) --> axb를 구하는 모델 --> train 실패
    with tf.device('/cpu:0'):
        batch_size=20
        binary_dim = 10
        int2binary = {}  # {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0, 1], ... }
    
        
        largest_number = pow(2,binary_dim)
        A=np.arange(largest_number).astype('uint16')
        binary = my_ManyParallel_uint2bits(A,binary_dim)  # 왼쪽이 낮은 자리
        for i in range(largest_number):
            int2binary[i] = binary[i]
        
        model = BinaryMultiply(binary_dim)
        opt = tf.train.AdamOptimizer(0.1).minimize(model.loss)
    
    
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            x = np.zeros([batch_size,binary_dim,2])
            y = np.zeros([batch_size,binary_dim])
            for i in range(30000):    
                # generate a simple addition problem (a + 2*b = c)
                a_int = np.random.randint(largest_number/32,size=(batch_size,2)) # int version
                
                
                for j in range(batch_size):
                    x[j] = np.vstack([int2binary[a_int[j,0]],int2binary[a_int[j,1]]]).T
                    y[j] = int2binary[a_int[j,0]*a_int[j,1]]
            
    
                sess.run([opt],feed_dict={model.X: x, model.Y: y})
                
                
                if i% 1000 == 0:
                    loss,output = sess.run([model.loss,model.outputs],feed_dict={model.X: x, model.Y: y})
                
                    pred = 1*(output.reshape(batch_size,-1)>0.5)
                    print('step {}: loss: {}'.format(i,loss))
                    #print('Pred:\n', pred)
                    #print('True:\n', y.astype(np.int32))
                    
                    out = 0
                    for index,k in enumerate(pred.T):
                        out += k*pow(2,index)
                    
                    print (str(a_int[:,0]) + " x " + str(a_int[:,1]) + "\n -> " + str(out), ( a_int[:,0] *a_int[:,1] == out))  
                    
if __name__ == "__main__":    
    s=time.time()
    run3()
    
    e=time.time()
    
    print('done: {} sec'.format(e-s))  