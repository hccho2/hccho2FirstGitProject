
'''
2.x 환경에서 1.x style로 코딩하기

'''
import numpy as np


def test1():
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

def test_RNN():
    import tensorflow.compat.v1 as tf
    
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 1
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)


    train_mode = True
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        
        method = 0
        if method == 0:
            cells = []
            for _ in range(num_layers):
                #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim)
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
                #cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim)
                cells.append(cell)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)    
        else:
            #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim)
            cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim,num_proj=7)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)



        # TF2에는 OutputProjectionWrapper이 없다.
        # Tf2에는 Helper가 없다 ---> Sampler
        

        init_state = cell.zero_state(batch_size, tf.float32)
        new_state = init_state
        
        output_all  = []
        for i in range(seq_length):
            output, new_state = cell(inputs[:,i,:],new_state)
            output_all.append(output)

        output_all = tf.stack(output_all,axis=1)  # --> [(N,D),(N,D), ..., (N,D)]   ---> (N,T,D)


if __name__ == '__main__':
    #test1()
    #test2()
    test_RNN()
    
    print('Done')

