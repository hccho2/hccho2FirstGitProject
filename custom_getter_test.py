# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

'''
def pretrained_initializer(varname, weight_file, embedding_weight_file=None):
    
    weight = .........
    
    
    def ret(shape, **kwargs):
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret

'''


def custom_getter(getter, name, *args, **kwargs):
    # 각 vaiable마다 여러번 호출된다.
    # name에 따라 어떤 initialization 값을 return할 지 판단하면 된다.
    print(name)  # name.startswith('hccho'), name.endswith('bias')
    kwargs['trainable'] = False
    kwargs['initializer'] = lambda shape,**kwargs: np.zeros(shape)  # 여기서는 lambda 함수를 이용했지만, 초기화 시킬 값을 가져올 수 있는 함수를 넣어주면 됨. eg. pretrained_initializer
    #kwargs['initializer'] = _pretrained_initializer(name, weight_file, embedding_weight_file)
    return getter(name, *args, **kwargs)


X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,1]]).T


with tf.variable_scope('hccho',custom_getter=custom_getter):

    x = tf.placeholder(tf.float32, [None,3])
    y = tf.placeholder(tf.float32, [None,1])
    L1 = tf.layers.dense(x,units=4, activation = tf.sigmoid,name='L1')
    L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')

with tf.variable_scope('hccho2',):
    L3 = tf.layers.dense(L2,units=1, activation = tf.sigmoid,name='L3')
train = tf.train.AdamOptimizer(learning_rate=1).minimize( tf.reduce_mean( 0.5*tf.square(L3-y)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # custom_getter에 해당하는 변수도 이 때, initialization된다.
    print(sess.run('hccho2/L3/kernel:0'))
    print(sess.run('hccho/L1/kernel:0'))
    
    for i in range(10):
        sess.run(train, feed_dict={x: X, y: Y})
    
    print(sess.run('hccho2/L3/kernel:0'))
    print(sess.run('hccho/L1/kernel:0'))


print('trainable_variables: ', tf.trainable_variables())
print('global_variables: ', tf.global_variables())
