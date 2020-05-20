
import tensorflow as tf


def simple_layer():
    class MyDenseLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs):
            super(MyDenseLayer, self).__init__()
            self.num_outputs = num_outputs
        
        def build(self, input_shape):
            self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]), self.num_outputs])
        
        def call(self, input):
            return tf.matmul(input, self.kernel)
    
    layer = MyDenseLayer(10)
    print(layer(tf.zeros([10, 5])))
    print(layer.trainable_variables)



if __name__ == '__main__':
    simple_layer()