
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

def RNN_test()
    import tensorflow as tf
    import numpy as np
    
    tf.reset_default_graph()
    
    # One hot encoding for each char in 'hello'
    h = [1, 0, 0, 0]; e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]; o = [0, 0, 0, 1]
    
    x_data = np.array([[h, e, l, l, o],[e, o, l, l, l],[l, l, e, e, l]], dtype=np.float32)
    batch_size = len(x_data)
    mode = 0
    
    hidden_dim = 2
    
    with tf.variable_scope('3_batches') as scope:
    
        if mode == 0:
            cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
        #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        else:
            cells = []
            for _ in range(3):
                cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
                #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 4)  # output에 FC layer를 추가하여 원하는 size로 변환해 준다.
    outputs, _states = tf.nn.dynamic_rnn(cell,x_data,initial_state=initial_state,dtype=tf.float32)
    
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("output", sess.run(outputs))
    print("hidden state",sess.run(_states))
    sess.close()
	
	
	
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
    import numpy as np
    import tensorflow as tf
    
    
    from tensorflow.python.layers.core import Dense
    tf.reset_default_graph()
    
    vocab_size = 5
    SOS_token = 0
    EOS_token = 4
    
    x_data = np.array([[SOS_token, 3, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[1,2,0,3,2,EOS_token],[3,2,3,3,1,EOS_token],[3,1,1,2,0,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =6
    num_layers = 2
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True
    init_state_flag = 0
    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
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
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        if init_state_flag==0:
             initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
        else:
            if state_tuple_mode:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)
                
            else:
                h0 = tf.random_normal([batch_size,hidden_dim]) #h0 = tf.cast(np.random.randn(batch_size,hidden_dim),tf.float32)
                initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
    
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다.
	outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
    
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
    
    
        sess.run(tf.global_variables_initializer())
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
    
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )
        
def test_bidirectional(): 
    import tensorflow as tf
    import numpy as np
    tf.reset_default_graph()
    x_data = np.array([[0, 3, 1],[1, 0, 0]], dtype=np.int32)
    x_data = np.expand_dims(x_data,2).astype(np.float32)
    
    #cell_f = tf.contrib.rnn.BasicRNNCell(num_units=2)
    #cell_b = tf.contrib.rnn.BasicRNNCell(num_units=2)

    cell_f = tf.contrib.rnn.BasicLSTMCell(num_units=2)
    cell_b = tf.contrib.rnn.BasicLSTMCell(num_units=2)

    (encoder_fw_outputs, encoder_bw_outputs),(encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f,cell_bw=cell_b,inputs=x_data,dtype=tf.float32)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    print("\nencoder_fw_outputs: ", sess.run(encoder_fw_outputs))
    print("\nencoder_bw_outputs: ", sess.run(encoder_bw_outputs))


    print("\nencoder_fw_final_state: ", sess.run(encoder_fw_final_state))
    print("\nencoder_bw_final_state: ", sess.run(encoder_bw_final_state))    

"""
BasicRNNCell:
encoder_fw_outputs:  
[[[ 0.          0.        ], [-0.37422162,0.9775176 ],[-0.6056877   0.57990843]]
 [[-0.13036166  0.63284284], [-0.41194066 -0.16159871],[ 0.3657935   0.44170365]]]

encoder_bw_outputs:  
[[[ 0.3409077  -0.8065934 ], [-0.43896067 -0.98964894], [-0.22973818 -0.58141154]]
 [[-0.22973818 -0.58141154], [ 0.          0.        ], [ 0.          0.        ]]]

encoder_fw_final_state:  
[[-0.6056877   0.57990843],[ 0.3657935   0.44170365]]

encoder_bw_final_state:  
[[ 0.3409077  -0.8065934 ],[-0.22973818 -0.58141154]]   


BasicLSTMCell:
encoder_fw_outputs:  
[[[ 0.          0.        ],[ 0.13082808 -0.10455302],[ 0.13346125 -0.10499903]]
[[ 0.04281872 -0.03912188], [ 0.03448014 -0.02246729],[ 0.02988851 -0.009888  ]]]

encoder_bw_outputs:  
[[[-0.04539058 -0.2564498 ], [-0.00913273 -0.5504859 ],[-0.01013058 -0.18332982]]
 [[-0.01013058 -0.18332982], [ 0.          0.        ],[ 0.          0.        ]]]

encoder_fw_final_state:  
LSTMStateTuple(c=array([[ 0.23159406, -0.27031744],[ 0.059204  , -0.02017506]], dtype=float32), 
			   h=array([[ 0.13346125, -0.10499903],[ 0.02988851, -0.009888  ]], dtype=float32))

encoder_bw_final_state:  
LSTMStateTuple(c=array([[-0.08561244, -0.71315455],[-0.02546103, -0.3122089 ]], dtype=float32), 
			   h=array([[-0.04539058, -0.2564498 ],[-0.01013058, -0.18332982]], dtype=float32))
"""        
        

 def get_info_from_checkpoint():
    import tensorflow as tf
    tf.reset_default_graph()
    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    checkpoint_dir = 'D:\\hccho\\cs231n-Assignment\\assignment3\\save-sigle-layer\\model.ckpt-1000000.ckpt' # 구체적으로 명시
    #checkpoint_dir = 'D:\\hccho\\cs231n-Assignment\\assignment3\\save-sigle-layer # 디렉토리만 지정 ==> 가장 최근
    var_list = checkpoint_utils.list_variables(checkpoint_dir)
    #sess = tf.Session()
    for v in var_list: 
        print(v) # tuple(variable name, [shape])
        vv = checkpoint_utils.load_variable(checkpoint_dir, v[0])
        print(vv) #values   
        
=======================
Bahdanau attention weight
encoder_hidden_size = 300   = context vector size
decoder_hidden_size = 110
BahdanauAttention_depth = 99
attention_layer_size=77 

[<tf.Variable 'embed/embeddings:0' shape=(103, 100) dtype=float32_ref>, 
 <tf.Variable 'rnn/gru_cell/gates/kernel:0' shape=(400, 600) dtype=float32_ref>, 
 <tf.Variable 'rnn/gru_cell/gates/bias:0' shape=(600,) dtype=float32_ref>, 
 <tf.Variable 'rnn/gru_cell/candidate/kernel:0' shape=(400, 300) dtype=float32_ref>, 
 <tf.Variable 'rnn/gru_cell/candidate/bias:0' shape=(300,) dtype=float32_ref>, 
 
 <tf.Variable 'decode/memory_layer/kernel:0' shape=(300, 99) dtype=float32_ref>, 
  <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/bahdanau_attention/query_layer/kernel:0' shape=(110, 99) dtype=float32_ref>,  ==>   decoder_hidden_size x BahdanauAttention_depth
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/bahdanau_attention/attention_v:0' shape=(99,) dtype=float32_ref>, 
 
 ==> context weight 계산 
 
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/gru_cell/gates/kernel:0' shape=(287, 220) dtype=float32_ref>, ==> (input 100 + decoder_hidden_size 110 + attention_layer_size 77)  x 2*decoder hidden
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/gru_cell/gates/bias:0' shape=(220,) dtype=float32_ref>, 
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/gru_cell/candidate/kernel:0' shape=(287, 110) dtype=float32_ref>, 
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/gru_cell/candidate/bias:0' shape=(110,) dtype=float32_ref>, 
 

 
 <tf.Variable 'decode/decoder/output_projection_wrapper/attention_wrapper/attention_layer/kernel:0' shape=(410, 77) dtype=float32_ref>,  None이 아닐 때, ==> tf.contrib.seq2seq.AttentionWrapper 에서 (encoder_hidden_size : decoder_hidden_size) ==>attention_layer_size  ==> attention
 
 
 <tf.Variable 'decode/decoder/output_projection_wrapper/kernel:0' shape=(77, 103) dtype=float32_ref>,   ==> outprojectionwrapper 또는 out_layer를 통해, attention(output)의 크기를 원하는 크기를 바꾼다.
 <tf.Variable 'decode/decoder/output_projection_wrapper/bias:0' shape=(103,) dtype=float32_ref>]

======================
# tf.layers.dense 의 input tensor가 3차원일 때:
# 예: input.shape (2,3,4) x units = 5  ==> (2,3,5)가 만들어지고, weight는 (4,5) size 이다
def dense_test():
    tf.reset_default_graph()
    A0 = np.arange(24).reshape(2,3,4).astype(np.float32)
    A = tf.convert_to_tensor(A0)
    init = np.arange(20).reshape(4,5).astype(np.float32)
    x = tf.layers.dense(A,5,kernel_initializer=tf.constant_initializer(init),activation=None)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    graph = tf.get_default_graph()
    
    xx = sess.run(x)
    w = sess.run(graph.get_tensor_by_name('dense/kernel:0'))
    print(xx)
    print(w)
	
	
if __name__ == "__main__":   
    test1()
    
    #test2()
    #test3()
    #testPlaceholde()
    #testLinearRegression()
    #testLinearRegression2()
    
    
    
