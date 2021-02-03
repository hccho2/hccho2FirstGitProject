

'''
https://www.tensorflow.org/tutorials/customization/custom_layers
https://www.tensorflow.org/guide/keras/rnn



1. layer class는 tf.keras.layers.Layer를 상속받아서 구현한다.
2. 기존에 있는 layer를 활용하는 경우에는 tf.keras.Model을 상속받는다.


사용자 정의 층을 구현하는 가장 좋은 방법은 tf.keras.Layer 클래스를 상속하고 다음과 같이 구현하는 것입니다. 
 1. __init__ 에서 층에 필요한 매개변수를 입력 받습니다. 
 2. build, 입력 텐서의 크기를 얻고 남은 초기화를 진행할 수 있습니다 .
 3. call, 정방향 연산(forward computation)을 진행 할 수 있습니다.

변수를 생성하기 위해 build가 호출되길 기다릴 필요가 없다는 것에 주목하세요. 
또한 변수를 __init__에 생성할 수도 있습니다. 그러나 build에 변수를 생성하는 유리한 점은 층이 작동할 입력의 크기를 기준으로 나중에 변수를 만들 수 있다는 것입니다. 
반면에, __init__에 변수를 생성하는 것은 변수 생성에 필요한 크기가 명시적으로 지정되어야 함을 의미합니다.


다른 층을 포함한 모델을 만들기 위해 사용하는 메인 클래스는 tf.keras.Model입니다. 다음은 tf.keras.Model을 상속(inheritance)하여 구현한 코드입니다.

'''



import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant

# tf.keras.layers.Layer로 부터 상속받아야 한다. tf.keras.Model은 안되나? tf.keras.Model은 Layer들의 집합성격이 강하다.
# tf.keras.layers.Layer ----> OK
# tf.keras.Model  ----> 아래, test_mode 1, 2에서는 OK.   3에서는 error      tf.keras.Model는 pytorch의 nn.Module로 보면 된다.
class MyCell(tf.keras.layers.Layer):
    # RNN + FC
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

class MyCell2(tf.keras.layers.Layer):
    # Residual cell
    def __init__(self, hidden_dim):
        super(MyCell2, self).__init__(name='')
        self.hidden_dim = hidden_dim
        self.rnn_cell = tf.keras.layers.LSTMCell(hidden_dim)
        
        self.state_size = self.rnn_cell.state_size
        self.output_size = hidden_dim  # self.rnn_cell.output_size

    def call(self, inputs, states,training=None):
        output, states = self.rnn_cell(inputs,states)
        output = output + inputs
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
    
    seq_length = 4
    feature_dim = 7
    hidden_dim = feature_dim
    
    #cell = MyCell(hidden_dim)   # User Defined Cell
    cell = MyCell2(hidden_dim)

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

############# evalution에서도 dropout이 적용되는 layer############################
class MCDropout(tf.keras.layers.Dropout):
    def call(self,inputs):
        super().call(inputs,training=True)



class MyModel(tf.keras.Model):
    def __init__(self,input_shape):
        super(MyModel, self).__init__()

        weight_decay = 1e-4

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())
        
        self.model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())        
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(0.2))      


        self.model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())
        
        self.model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())        
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(0.3))    


        self.model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())
        
        self.model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ELU())        
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(0.4))   

        self.model.add(tf.keras.layers.Flatten())


        self.built = True  # summary가 제대로 작동한다.

    def call(self,x,training=None):  # training의 default 값으로 None이 좋다(Ture/False보다)
        output = self.model(x)
        return output
    

def model_test():
    
    input_shape = (32,32,3)
    
    model = MyModel(input_shape)
    model.summary()
    
    
    x = np.random.randn(2,32,32,3)
    
    out = model(x)
    print(out.shape)

if __name__ == '__main__':
    #simple_layer()
    #simple_layer2()
    #simple_rnn()
    #user_defined_cell_test()  # User Defined cell인 MyCell test
    #user_defined_cell_decoder_test()  # User Defined cell인 MyCell + decoder test

    model_test()
    print('Done')

