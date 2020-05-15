
'''
2.x 환경에서 1.x style로 코딩하기

'''



def test1():
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    x = tf.placeholder(tf.float32,[None,3])
    y = tf.placeholder(tf.float32,[None,1])
    z = tf.keras.layers.Dense(units=1)(x)  # 또는 z = tf.layers.dense(x, units=1)
    
    loss = tf.losses.mean_squared_error(y,z)   # 제곱하여 batch에 대한 평균
    
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    
    print(z)
    
    data_x = np.random.randn(2,3)
    data_y = np.random.randn(2,1)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([x,y,z,loss], feed_dict={x: data_x,y: data_y}))


    for i in range(100):
        _, l = sess.run([train_op,loss],feed_dict={x: data_x,y: data_y})
        print(i, l)

    print(sess.run([x,y,z,loss], feed_dict={x: data_x,y: data_y}))




def test2():
    import numpy as np
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
    
    
    
    
    
    
    x = tf1.placeholder(tf.float32,[None,3])
    y = tf1.placeholder(tf.float32)
    z = tf.keras.layers.Dense(units=1)(x)  # 또는 z = tf1.layers.dense(x, units=1)
    
    
    print(z)
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())
    print(sess.run(z, feed_dict={x: np.random.randn(2,3)}))    
