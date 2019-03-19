# -*- coding: utf-8 -*-
"""
https://keras.io/layers/writing-your-own-keras-layers/

There are only three methods you need to implement:

- build(input_shape): this is where you will define your weights. This method must set self.built = True at the end, which can be done by calling super([Layer], self).build().
- call(x): this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor.
- compute_output_shape(input_shape): in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows Keras to do automatic shape inference.
"""



import numpy as np
import tensorflow as tf
tf.reset_default_graph()


class MyKerasWrapper(tf.keras.layers.Wrapper):
    def __init__(self,hidden_dim,is_training=True,name=None,**kwargs):
        
        layer = tf.keras.layers.Dense(units=hidden_dim,activation=tf.nn.relu)
        super(MyKerasWrapper, self).__init__(layer, name=name)  # --> self.layer가 만들어 진다.  layer가 여러개 있는 경우는 그 중 첫번째 것만 넘겨서 처리 한다.
        self._track_checkpointable(layer, name='layer')
            
    def build(self, input_shape):
        
        super(MyKerasWrapper, self).build(input_shape)  # Be sure to call this at the end
        #self.built = True  #super를 부르면 True로 변해 있다.
    def call(self, inputs):
        return self.layer(inputs)
    
class MyKerasWrapper2(tf.keras.layers.Wrapper):
    def __init__(self,hidden_dim,is_training=True,name=None,**kwargs):
        
        layer = tf.keras.layers.Conv1D(filters=hidden_dim,kernel_size=2,activation=tf.nn.tanh,padding='causal')
        self.layer2 = tf.keras.layers.Conv1D(filters=hidden_dim,kernel_size=3 ,activation=None,padding='causal')
        super(MyKerasWrapper2, self).__init__(layer, name=name)  # --> self.layer가 만들어 진다.  layer가 여러개 있는 경우는 그 중 첫번째 것만 넘겨서 처리 한다.
        self._track_checkpointable(layer, name='layer')
            
    def build(self, input_shape):
        
        super(MyKerasWrapper2, self).build(input_shape)  # Be sure to call this at the end
        #self.built = True  #super를 부르면 True로 변해 있다.
    def call(self, inputs):
        out = self.layer(inputs)
        out = self.layer2(out)
        return inputs + out
    
def test():
    
    batch_size=2
    c_in=2
    c_out=5
    T=10
    
    mylayer = MyKerasWrapper(c_out,name='AAA')
    mylayer2 = MyKerasWrapper2(c_out,name='BBB')    
    
    x = np.random.normal(size=[batch_size,T,c_in])
    xx = tf.convert_to_tensor(x)
    
    y = mylayer(xx)  # instance를 만들면, base class의 __call__ 내에서  self.build --> self.call 순으로 실행된다. 
    z = mylayer2(y)



    print('Done')
    
if __name__ == '__main__':
    test()
    print('Done')