# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

'''
def pretrained_initializer(varname, weight_file, embedding_weight_file=None):
    # elmo 모델에서 활용. bilm/model.py에 있는 코드
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
    L1 = tf.layers.dense(x,units=4, activation = tf.sigmoid,name='L1')  # hccho/L1/kernel, hccho/L1/bias 2개가 있으므로, custom_getter가 2번 
    L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2') # hccho/L2/kernel, hccho/L2/bias 2개가 있으므로, custom_getter가 2번 
    
    tf.add_to_collection('hccho_collection', [L1,L2])

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
print('-'*10)
print('tf.hccho_collection', tf.get_collection('hccho_collection') )


############################################################################################
############################################################################################
############################################################################################

def custom_getter(getter, name, *args, **kwargs):
    g_0 = getter("%s/0" % name, *args, **kwargs)
    g_1 = getter("%s/1" % name, *args, **kwargs)
    with tf.name_scope("custom_getter"):
        return g_0 + g_1  # or g_0 * const / ||g_0|| or anything you want



with tf.variable_scope("scope", custom_getter=custom_getter):
    v = tf.get_variable("v", [1, 2, 3])   # 내부적으로 2개의 variable이 생성된다.



sess = tf.Session()

sess.run(tf.global_variables_initializer())

sess.run(v)
tf.trainable_variables()


############################################################################################
############################################################################################
############################################################################################
# RL에서 main, target network이 있을 때, soft update를 exponential weighted average로 할 때
# explicit한 update를 하지 않고, tf.train.ExponentialMovingAverage를 활용.

ema = tf.train.ExponentialMovingAverage(decay=0.9)

def ema_getter(getter, name, *args, **kwargs):
    return ema.average(getter(name, *args, **kwargs))  # shadow variable에 접근

def build_net(s,reuse=None, custom_getter=None):
    trainable = True if reuse is None else False
    with tf.variable_scope('MyNet', reuse=reuse, custom_getter=custom_getter):
        out = tf.layers.dense(s,units=1,trainable=trainable)
        return out
    
x1 = tf.placeholder(tf.float32,shape=[None,2])
target = tf.placeholder(tf.float32,shape=[None,1])
y1 = build_net(x1)
a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MyNet')
target_update = [ema.apply(a_params)]  # trainable weight로 부터 shadow weight를 만든다.


x2 = tf.placeholder(tf.float32,shape=[None,2])
# network의 trainable weight에 대한 exponential moving average를 weight로 가지는 network을 만든다.
# weight 자체는 trainable하지 않다.
y2 = build_net(x2, reuse=True, custom_getter=ema_getter)


loss = tf.losses.mean_squared_error(y1,target)
with tf.control_dependencies(target_update):
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)  # train_op가 계산되기 전에 target_update가 먼저 계산된다.

sess = tf.Session()
sess.run(tf.global_variables_initializer())


shadow_variables= [ema.average(tf.trainable_variables()[0]),ema.average(tf.trainable_variables()[1])]
shadow_variables = [ema.average(a_params[0]),ema.average(a_params[1])]

print('Before:')
print(sess.run(tf.trainable_variables()))



data_x = np.random.randn(2,2)
data_y = np.array([[1.0],[1.5]])

for i in range(100):
    _,l_= sess.run([train_op,loss],feed_dict={x1: data_x, target:data_y })
    if i%100:
        print(i,l_)

print('After:')
print(sess.run(tf.global_variables()))
print('=='*10)
print('trainable variables:')
A = sess.run(tf.trainable_variables())
print(A)
print('-'*10)
print('shadow variables:')
B = sess.run(shadow_variables)
print(B)

print('===='*10)
print('two network results')
print(sess.run([y1,y2],feed_dict={x1: data_x, x2:data_x }))
print('===='*10)
print('check')
print(data_x.dot(A[0])+A[1])   # y1과 동일
print(data_x.dot(B[0])+B[1])   # y2와 동일


