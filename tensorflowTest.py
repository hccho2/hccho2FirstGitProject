
def test1():
    import tensorflow as tf
    print( tf.__version__)
    
    hello = tf.constant("Hello, TensorFlow!")
    sess = tf.Session()
    
    print( sess.run(hello))
    



def test2():
    import tensorflow as tf
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1,node2)
    
    sess = tf.Session()
    print(node1 , "\n",  node2,"\n", node3)
    print( sess.run(node3) )

def test22():
    import tensorflow as tf
    
    nodes={}
    
    nodes['1'] = tf.constant(3.0, tf.float32)
    nodes['2'] = tf.constant(4.0)
    nodes['3'] = tf.add(nodes['1'],nodes['2'])
    
    sess = tf.Session()
    print(nodes['1'] , "\n",  nodes['2'],"\n", nodes['3'])
    print( sess.run(nodes['3']) )

    sess.close()    


def test3():
    import tensorflow as tf
    
    x=1
    y = x+9
    print(y)
    
    x = tf.constant(1, name = 'x')
    y = tf.Variable(x+9, name = 'y')
    print(x,y)
    
    model = tf.global_variables_initializer()
    
    
    with tf.Session() as session:
        session.run(model)
        print(session.run(y))
def testOneHot():
    import tensorflow as tf
    import numpy as np
    Y1=np.array([[2],[4]])
    Y2=tf.one_hot(Y1,5)  # shape=(2,1,5)
    Y3=tf.reshape(Y2,[-1,5]) # shape=(2,5)
    
    sess = tf.Session()
    
    print(sess.run(Y2), sess.run(Y3))

def testPlaceholde():
    import tensorflow as tf
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    
    
    #result_node = a + b
    result_node = tf.add(a,b)

    sess = tf.Session()
    print(sess.run(result_node, feed_dict={a: 3.5, b: 2.7}))
    print(sess.run(result_node, feed_dict={a: [1,2], b: [2,5]}))
    
    
def testLinearRegression():
    import tensorflow as tf    
    
    x_train = [1,2,3]
    y_train = [1,2,3]
    
    W = tf.Variable(tf.random_normal([1]),name='Weight')
    b = tf.Variable(tf.random_normal([1]),name='bias')
    
    hypothesis = x_train * W + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(cost)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(4000):
        sess.run(train)
        
        if i%100 ==0:
            print(i,sess.run(cost), sess.run(W), sess.run(b))

def testLinearRegression2():
    import tensorflow as tf    
    
    x_train = tf.placeholder(tf.float32)
    y_train = tf.placeholder(tf.float32)
    
    W = tf.Variable(tf.random_normal([1]),name='Weight')
    b = tf.Variable(tf.random_normal([1]),name='bias')
    
    hypothesis = x_train * W + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(cost)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #merged = tf.merge_all_summaries()
    #writer = tf.summary.FileWriter("/tmp/test_logs", sess.graph)
    for i in range(2000):
        cost_, W_, b_, train_ = sess.run([cost,W,b,train], feed_dict={x_train: [1,2,3], y_train: [10,20,30]})
        
        if i%100 ==0:
            print(i,cost_, W_, b_)
    
    # Test our model        
    print(sess.run(hypothesis, feed_dict={x_train: [6.0, 5.4]}))        

def test_gpu():
    import tensorflow as tf

    c = []
    for d in ['/gpu:2', '/gpu:3']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print sess.run(sum) 
    
def optimization_test():
    def myfun(x):
        return tf.reduce_sum(x*x-2*x)
    vector = tf.Variable(7., 'vector')
    loss = myfun(vector)

    #optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100, 'disp': True})
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,method='L-BFGS-B', options={'maxiter': 100, 'disp': True})


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        print(session.run(vector))    
        
def embedding():
    tf.reset_default_graph()

    x_data = np.array([[0, 3, 1, 2, 4],[1, 3, 1, 2, 3],[2, 4, 0, 2, 4]], dtype=np.int32) # (batch_size,seq_length)

    input_dim = 5; embedding_dim = 6;
    init = np.arange(input_dim*embedding_dim).reshape(input_dim,-1) # sample initialization

    sess = tf.InteractiveSession()

    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        embedding = tf.get_variable("embedding", initializer=init) # shape=(input_dim, embedding_dim)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # shape=(batch_size, seq_length, embedding_dim)

        sess.run(tf.global_variables_initializer())
        print(embedding)
        print("inputs",inputs)

        print(sess.run(embedding))
        print(sess.run(inputs))
 
def test_legacy_seq2seq():
    tf.reset_default_graph()

    x_data = np.array([[0, 3, 1, 2, 4],[1, 3, 1, 2, 3],[2, 4, 0, 2, 4]], dtype=np.int32)
    init = np.arange(30).reshape(5,-1)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    batch_size = len(x_data)
    hidden_dim =6
    num_layers = 2
    seq_length = x_data.shape[1]
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)    


        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data)



        initial_state = cell.zero_state(batch_size, tf.float32) # num_layers tuple. batch x hidden_dim
        inputs = tf.split(inputs,seq_length,1)
        inputs = [tf.squeeze(input_,[1]) for input_ in inputs]

        outputs, last_state =  tf.contrib.legacy_seq2seq.rnn_decoder(inputs,initial_state,cell)




        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        print(sess.run(outputs)) #seq_length, batch_size, hidden_dim

        print("\n\nlast_state: ",last_state)  # last_state이 마지막 값은 output의 마지막과 같은 값
        print(sess.run(last_state)) # num_layers, batch_size, hidden_dim

def test_seq2seq():
    from tensorflow.python.layers.core import Dense
    tf.reset_default_graph()

    x_data = np.array([[0, 3, 1, 2, 4, 3],[1, 3, 1, 2, 3, 2],[2, 4, 0, 2, 4, 1]], dtype=np.int32)

    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    input_dim = 5
    output_dim = input_dim
    batch_size = len(x_data)
    hidden_dim =6
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 1
    init = np.arange(input_dim*embedding_dim).reshape(input_dim,-1)
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn
        cells = []
        for _ in range(num_layers):
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)    
        #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)

        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data)  # batch_size  x seq_length x embedding_dim

        if init_state_flag==0:
             initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        else:
            if state_tuple_mode:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)
                
            else:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)

        helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))


        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True)

        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        print("\n",sess.run(outputs.rnn_output)) #batch_size, seq_length, outputs

        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim

        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        
        print(sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print(sess.run(output_layer.trainable_weights[1]))  # bias

        

        
        
def test_bidirectional(): 
    import tensorflow as tf
    import numpy as np
    tf.reset_default_graph()
    x_data = np.array([[0, 3, 1],[1, 0, 0]], dtype=np.int32)
    x_data = np.expand_dims(x_data,2).astype(np.float32)
    cell_f = tf.contrib.rnn.BasicRNNCell(num_units=2)
    cell_b = tf.contrib.rnn.BasicRNNCell(num_units=2)

    (encoder_fw_outputs, encoder_bw_outputs),(encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f,cell_bw=cell_b,inputs=x_data,dtype=tf.float32)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    print("\nencoder_fw_outputs: ", sess.run(encoder_fw_outputs))
    print("\nencoder_bw_outputs: ", sess.run(encoder_bw_outputs))


    print("\nencoder_fw_final_state: ", sess.run(encoder_fw_final_state))
    print("\nencoder_bw_final_state: ", sess.run(encoder_bw_final_state))    
        


 def get_info_from_checkpoint():
    tf.reset_default_graph()
    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    checkpoint_dir = 'D:\\hccho\\ML\\cs231n\\assignment3\\save-double-layer'

    var_list = checkpoint_utils.list_variables(checkpoint_dir)
    sess = tf.Session()
    for v in var_list: 
        print(v) # tuple(variable name, [shape])
        vv = checkpoint_utils.load_variable(checkpoint_dir, v[0])
        print(vv) #values        
        
if __name__ == "__main__":   
    test1()
    
    #test2()
    #test3()
    #testPlaceholde()
    #testLinearRegression()
    #testLinearRegression2()
    
    
    
