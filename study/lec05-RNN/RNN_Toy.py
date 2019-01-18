#  coding: utf-8
import copy, numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

class BinaryAddition():
    def __init__(self,binary_dim,hidden_dim=16):
        self.binary_dim = binary_dim  #length
        self.hidden_dim = hidden_dim
        
        self.build()

    def build(self):
        
        
        self.X = tf.placeholder(tf.float32,shape=[None,self.binary_dim,2])
        self.Y = tf.placeholder(tf.float32,shape=[None,self.binary_dim])
        
        
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_dim)
        
        
        batch_size = tf.shape(self.X)[0]
        seq_length = tf.shape(self.X)[1]
        
        
        
        initial_state = cell.zero_state(batch_size, tf.float32)
        

        helper = tf.contrib.seq2seq.TrainingHelper(self.X, tf.tile([self.binary_dim],[batch_size]))

        output_layer = Dense(1,activation=tf.nn.sigmoid, name='output_projection')    
        
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)  
        self.outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True)
        
        self.loss = tf.nn.l2_loss(tf.reshape(self.outputs.rnn_output,shape=[batch_size,-1])-self.Y)
               

def run():
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
            
                pred = 1*(output.rnn_output.reshape(2,-1)>0.5)
                print('step {}: loss: {}'.format(i,loss))
                print('Pred:\n', pred)
                print('True:\n', y.astype(np.int32))
                
                out = 0
                for index,k in enumerate(pred.T):
                    out += k*pow(2,index)
                
                print (str(a_int[:,0]) + " + " + str(a_int[:,1]) + " = " + str(out), ( a_int[:,0] + a_int[:,1] == out))



if __name__ == "__main__":    
    s=time.time()
    run()
    
    e=time.time()
    
    print('done: {} sec'.format(e-s))  