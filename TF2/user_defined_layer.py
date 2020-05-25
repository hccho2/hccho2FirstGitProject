

'''
https://www.tensorflow.org/tutorials/customization/custom_layers
https://www.tensorflow.org/guide/keras/rnn



1. layer class는 tf.keras.layers.Layer를 상속받아서 구현한다.
2. 기존에 있는 layer를 황용하는 경우에는 tf.keras.Model을 상속받는다.

'''



import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant

# tf.keras.layers.Layer로 부터 상속받아야 한다. tf.keras.Model은 안되나? tf.keras.Model은 Layer들의 집합성격이 강하다.
# tf.keras.layers.Layer ----> OK
# tf.keras.Model  ----> 아래, test_mode 1, 2에서는 OK.   3에서는 error      tf.keras.Model는 pytorch의 nn.Module로 보면 된다.
class MyCell(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super(MyCell, self).__init__(name='')
        self.hidden_dim = hidden_dim
        self.rnn_cell = tf.keras.layers.LSTMCell(hidden_dim)
        self.dense = tf.keras.layers.Dense(2*hidden_dim)
        
        self.state_size = self.rnn_cell.state_size
        self.output_size = 2*hidden_dim  # self.rnn_cell.output_size

    def call(self, inputs, states,training=None):
        output, states = self.rnn_cell(inputs,states)
        output = self.dense(output)
        return output,states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
        return [tf.zeros((batch_size,self.hidden_dim),dtype=dtype), tf.zeros((batch_size,self.hidden_dim),dtype=dtype)]



def simple_layer():
    class MyDenseLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs):
            super(MyDenseLayer, self).__init__()
            self.num_outputs = num_outputs
        
        def build(self, input_shape):
            # input data가 처음 들어오면, 그 shape을 보고, weight를 만든다.
            self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]), self.num_outputs])
        
        def call(self, input,training=None):
            return tf.matmul(input, self.kernel)
    
    layer = MyDenseLayer(10)
    inputs = tf.random.normal([10,5])
    print(layer(inputs,training=True))
    print(layer.trainable_variables)


def simple_layer2():
    # 이런 구조는 pytorch의 nn.Module을 상속받고, forward를 정의하는 구조와 유사하다.
    # tf.kears.Model을 상속받고, call을 정의한다.
    class ResnetIdentityBlock(tf.keras.Model):
        def __init__(self, kernel_size, filters):
            super(ResnetIdentityBlock, self).__init__(name='')
            filters1, filters2, filters3 = filters
            
            self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
            self.bn2a = tf.keras.layers.BatchNormalization()
            
            self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()
            
            self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
            self.bn2c = tf.keras.layers.BatchNormalization()
    
        def call(self, input_tensor, training=False):
            x = self.conv2a(input_tensor)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)
            
            x = self.conv2b(x)
            x = self.bn2b(x, training=training)
            x = tf.nn.relu(x)
            
            x = self.conv2c(x)
            x = self.bn2c(x, training=training)
            
            x += input_tensor
            return tf.nn.relu(x)
    
    
    block = ResnetIdentityBlock(1, [1, 2, 3])
    print(block(tf.zeros([1, 2, 3, 3])))
    print([x.name for x in block.trainable_variables])

def simple_rnn():
    class NestedCell(tf.keras.layers.Layer):
        def __init__(self, unit_1, unit_2, unit_3, **kwargs):
            self.unit_1 = unit_1
            self.unit_2 = unit_2
            self.unit_3 = unit_3
            self.state_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
            self.output_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
            super(NestedCell, self).__init__(**kwargs)
    
        def build(self, input_shapes):
            # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
            i1 = input_shapes[0][1]
            i2 = input_shapes[1][1]
            i3 = input_shapes[1][2]
        
            self.kernel_1 = self.add_weight( shape=(i1, self.unit_1), initializer='uniform', name='kernel_1')
            self.kernel_2_3 = self.add_weight( shape=(i2, i3, self.unit_2, self.unit_3), initializer='uniform', name='kernel_2_3')
    
        def call(self, inputs, states):
            # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
            # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
            input_1, input_2 = tf.nest.flatten(inputs)
            s1, s2 = states
        
            output_1 = tf.matmul(input_1, self.kernel_1)
            output_2_3 = tf.einsum('bij,ijkl->bkl', input_2, self.kernel_2_3)
            state_1 = s1 + output_1
            state_2_3 = s2 + output_2_3
        
            output = (output_1, output_2_3)
            new_states = (state_1, state_2_3)
        
            return output, new_states
    
        def get_config(self):
            return {'unit_1':self.unit_1,  'unit_2':unit_2, 'unit_3':self.unit_3}
    
    
    ###########################
    ###########################
    unit_1 = 10
    unit_2 = 20
    unit_3 = 30
    
    i1 = 32
    i2 = 64
    i3 = 32

    
    cell = NestedCell(unit_1, unit_2, unit_3)
    rnn = tf.keras.layers.RNN(cell)
    
    input_1 = tf.keras.Input((None, i1))
    input_2 = tf.keras.Input((None, i2, i3))
    
    outputs = rnn((input_1, input_2))
    
    model = tf.keras.models.Model([input_1, input_2], outputs)
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
def user_defined_cell_test():

    # User Defined cell인 MyCell test
    batch_size = 3
    hidden_dim = 5
    seq_length = 4
    feature_dim = 7
    
    cell = MyCell(hidden_dim)   # User Defined Cell

    test_mode = 3
    if test_mode==1:
        # 1 time step 처리
        inputs = tf.random.normal([batch_size, feature_dim])
        states =  cell.get_initial_state(inputs=None, batch_size=batch_size,dtype=tf.float32)
        outputs, states = cell(inputs,states,training=True)
        print(outputs)
        print(states)
    elif test_mode==2:
        # 여러 step을 loop로 처리
        inputs = tf.random.normal([batch_size, seq_length, feature_dim])
        
        states =  [tf.zeros([batch_size,hidden_dim]),tf.zeros([batch_size,hidden_dim])]
        outputs_all = []
        for i in range(seq_length):
            outputs, states = cell(inputs[:,i,:], states)
            outputs_all.append(outputs)
        
        
        outputs_all = tf.stack(outputs_all,axis=1)
        print(outputs_all)
        print(states)
    elif test_mode==3:
        # tf.keras.layers.RNN을 만들어 batch로 처리.
        rnn = tf.keras.layers.RNN(cell,return_sequences=True, return_state=True)
        
        inputs = tf.random.normal([batch_size, seq_length, feature_dim])
        states = rnn.get_initial_state(inputs)
        
        whole_seq_output, final_memory_state, final_carry_state = rnn(inputs,states)
        
        print(whole_seq_output.shape, whole_seq_output)
        print(final_memory_state.shape, final_memory_state)
        print(final_carry_state.shape, final_carry_state)
        



def user_defined_cell_decoder_test():
    # User Defined cell인 MyCell + decoder test
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
    

    
    # Sampler
    sampler = tfa.seq2seq.sampler.TrainingSampler()
    
    # Decoder
    
    method = 2
    if method==2:
        #decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
        decoder_cell = MyCell(hidden_dim)
        # decoder init state:
        
        #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
        init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        
    else:
        decoder_cell = tf.keras.layers.StackedRNNCells([MyCell(hidden_dim),MyCell(2*hidden_dim)])
        init_state = decoder_cell.get_initial_state(inputs=tf.zeros_like(x_data,dtype=tf.float32))  # inputs의 batch_size만 참조하기 때문에
    
    
    projection_layer = tf.keras.layers.Dense(output_dim)
    
    
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
    
    outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size)
    logits = outputs.rnn_output
    
    print(logits.shape)






if __name__ == '__main__':
    simple_layer()
    #simple_layer2()
    #simple_rnn()
    #user_defined_cell_test()  # User Defined cell인 MyCell test
    #user_defined_cell_decoder_test()  # User Defined cell인 MyCell + decoder test

    print('Done')


