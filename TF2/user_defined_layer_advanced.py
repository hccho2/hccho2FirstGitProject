

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


>>> tf.keras.initializers.serialize(tf.keras.initializers.RandomUniform())
{'class_name': 'RandomUniform', 'config': {'minval': -0.05, 'maxval': 0.05, 'seed': None}}

>>> tf.keras.initializers.deserialize()

>>> tf.keras.initializers.deserialize('uniform')
<tensorflow.python.ops.init_ops_v2.RandomUniform object at 0x0000000012F21F28>
>>> tf.keras.initializers.deserialize('normal')
<tensorflow.python.ops.init_ops_v2.RandomNormal object at 0x0000000012F21EB8>


'''



import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
from tensorflow.python.util import nest
class MinimalRNNCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)
    
    @property
    def state_size(self):
        return self.units
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer=tf.keras.initializers.RandomUniform(), name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units), initializer='zeros', name='bias')
        self.built = True
    
    def call(self, inputs, states):
        prev_output = states[0] if nest.is_sequence(states) else states  # tf.keras.layers.RNN에서는 state를 tuple로 다룬다.
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel) + self.bias
        return output, output


hidden_dim = 7

mycell = MinimalRNNCell(hidden_dim)





batch_size = 3
seq_length = 5
input_dim = 7
hidden_dim = 4

mycell = MinimalRNNCell(hidden_dim)

initial_state =  mycell.get_initial_state(inputs=None, batch_size=batch_size,dtype=tf.float32)


print(initial_state)
inputs = tf.random.normal([batch_size, seq_length, input_dim])

state = initial_state
output_all = []
for i in range(seq_length):
    output, state = mycell(inputs[:,i,:],state)
    output_all.append(output)

output_all = tf.stack(output_all,axis=1)
print(output_all)

# tf.keras.layers.RNN으로  batch 처리
rnn = tf.keras.layers.RNN(mycell,return_sequences=True)
output_all2 = rnn(inputs,initial_state)   # output_all과 같은 결과
print(output_all2)













