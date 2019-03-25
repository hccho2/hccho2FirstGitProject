# remove warning
tf.logging.set_verbosity(tf.logging.ERROR)

######################################################################
np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
######################################################################
# 아래의 api는 어떻게 사용하는지 한번 정리해야 겠다~~

tf.scatter_update
tf.gether
tf.gather_nd
tf.scatter_add
tf.sequence_mask
tf.slice


######################################################################

w0= np.array([[[  4,   3,   6,   7],
     [ -5,   5,   0,  -6],
     [ -4,   9,  -6,   1]],
     [[ -7,   1,  -3,  -1],
      [ -6,  16,  10,  12],
      [  6,   5,   8, -15]]]).astype(np.float32)

weights0 = tf.get_variable('weights0', shape=w0.shape)
weights1 = tf.get_variable('weights1', shape=w0.shape,initializer=tf.constant_initializer(w0))
weights3 = tf.get_variable('weights3', initializer=w0)
weights2 = tf.Variable(w0, name='weights2')






def get_network_size():
    print ('network size: {:,}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))


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
    
    h = [1, 0, 0, 0]; e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]; o = [0, 0, 0, 1]
    mode = 1
    
    x_data = np.array([[h, e, l, l, o],[e, o, l, l, l],[l, l, e, e, l]], dtype=np.float32)
    batch_size = len(x_data)
    hidden_size = 2
    if mode == 0:
        cell = rnn.BasicRNNCell(num_units=hidden_size)
    #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    else:
        cells = []
        for _ in range(3):
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    # output에 FC layer를 추가하여 원하는 size로 변환해 준다. 필요 없으면 빼도 됨
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 4)
    
    
    outputs, _states = tf.nn.dynamic_rnn(cell,x_data,initial_state=initial_state,dtype=tf.float32)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs))
    print(sess.run(_states))
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
        
=======================
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
 
def init_from_checkpoint():
    checkpoint_dir = 'D:\\hccho\\multi-speaker-tacotron-tensorflow-master\\logs\\moon_2018-08-18_20-01-48\\model.ckpt-48000' # 구체적으로 명시
    #checkpoint_dir = 'D:\\hccho\\cs231n-Assignment\\assignment3\\save-sigle-layer # 디렉토리만 지정 ==> 가장 최근
    var_list = checkpoint_utils.list_variables(checkpoint_dir)
    
    
    #1 직접 선언한 variable 초기화
    vv = checkpoint_utils.load_variable(checkpoint_dir, var_list[100][0])
    w = tf.get_variable('var1', shape=vv.shape)
    tf.train.init_from_checkpoint(checkpoint_dir,{var_list[100][0]: w})  # initializer 해야 값이 할당된다.
    #2 간접적으로 선언된 variable 초기화
    vv2 = checkpoint_utils.load_variable(checkpoint_dir, var_list[140][0])
    X = np.arange(2*128).reshape(2,128).astype(np.float32)
    Y = tf.layers.dense(tf.convert_to_tensor(X),units=128)
    tf.train.init_from_checkpoint(checkpoint_dir,{var_list[140][0]: 'dense/kernel'})
    graph = tf.get_default_graph()
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ww,kk = sess.run([w,graph.get_tensor_by_name('dense/kernel:0')])
    
    print(np.allclose(ww,vv))
    print(np.allclose(kk,vv2))


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
# 즉, 4개의 숫자들 간의 계산으로 새로운 5개의 숫자를 만들어낸다. 첫번째, 두번째 index들 간의 계산은 이루어지지 않는다.
# (batch_size, Time Length, embedding_dim)으로 해석하면, batch나 Time간의 계산은 되지 않고, embedding만 연산된다.
# https://github.com/tensorflow/tensorflow/issues/8175
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

======================


def tf_binary_image()
    import skimage.io as io
    import matplotlib.pyplot as plt
    cat_img = io.imread('cat.jpeg')  # integer numpy array, (194, 260, 3)
    
    
    cat_string = cat_img.tostring()  # b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\ ....'
    
    reconstructed_cat_1d = np.fromstring(cat_string, dtype=np.uint8)  # array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)
    reconstructed_cat_img = reconstructed_cat_1d.reshape(cat_img.shape)
    print(np.allclose(cat_img, reconstructed_cat_img))
    
    
    
    with tf.gfile.FastGFile('cat.jpeg', 'rb') as f:
        cat_img2 = f.read()  # b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\
        cat_img2 = tf.image.decode_jpeg(cat_img2, channels=3)
        cat_img2 = tf.image.resize_images(cat_img2, size=cat_img.shape[:-1])
        
    sess = tf.Session()
    cat_img2 = sess.run(cat_img2).astype(np.uint8) 
    
    # numerical 문제로 같은 값은 아니다.
    print(np.mean([cat_img, cat_img2],axis=(1,2,3)))
    
    io.imshow(np.concatenate([cat_img,cat_img2],axis=1))
    
    plt.show()
#############################################################
def TFRecord_reading1():
    # tfrecord 파일에 있는 data를 thread에 넣어 놓고 sess.run 할 때마다 뽑아 사용하기
    # tfrecord에서binary data가 저장되어 있는데, tf.image.decode_jpeg로 이용해서 0~255 사이 값으로 변환한다.
    from skimage import io
    from matplotlib import pyplot as plt
    
    filename = 'D:\\hccho\\CycleGAN-TensorFlow-master\\data\\tfrecords\\apple.tfrecords'
    
    
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    
    
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(serialized_example, features={'image/file_name': tf.FixedLenFeature([], tf.string), 'image/encoded_image': tf.FixedLenFeature([], tf.string),})
    
    
    image_buffer = features['image/encoded_image']
    file_name_buffer = features['image/file_name']
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.resize_images(image, size=(256, 256))
    
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image/127.5 -1.0
    
    image.set_shape([256, 256, 3])
    
    
    # image와 file_name_buffer를 같이 shuffle_batch로 해야, data쌍이 맞다.
    images,file_names = tf.train.shuffle_batch( [image,file_name_buffer], batch_size=5, num_threads=8, capacity=1500, min_after_dequeue=100 )
    
    sess = tf.Session()
    
    # 이부분이 반드시 있어야 됨.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    
    sess.run(tf.global_variables_initializer())
    a,z=sess.run([images,file_names])   # images, file_names가 쌍으로 
    b=sess.run(images)                  # images만 사용해도, 내부적으로는 file_names도 소모 
    
    print(a.shape,b.shape)
    print(np.mean([a[0],b[0]],axis=(1,2,3)))
    print(z)
    io.imshow(np.concatenate(a,axis=1))
    plt.show() 
    ########################
    
    a,z=sess.run([images,file_names])   # images, file_names가 쌍으로 
    print(z)
    
    io.imshow(np.concatenate(a,axis=1))
    plt.show()  
#############################################################    
def TFRecord_reading2():
    # tfrecord 파일에서 전체 data 뽑아내기
    # tfrecord에서binary data가 저장되어 있는데, tf.image.decode_jpeg로 이용해서 0~255 사이 값으로 변환한다.
    filename = 'D:\\hccho\\CycleGAN-TensorFlow-master\\data\\tfrecords\\apple.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    
    
    reconstructed_images = []
    reconstructed_file_names = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        image_buffer = example.features.feature['image/encoded_image'].bytes_list.value[0]   # binary data
        image = tf.image.decode_jpeg(image_buffer, channels=3)  # binary data가 tensor로 변환된다.
        image = tf.image.resize_images(image, size=(256, 256))
        reconstructed_images.append(image)    
        
        file_name_buffer = example.features.feature['image/file_name'].bytes_list.value[0] # tensor 아님
        reconstructed_file_names.append(file_name_buffer)
        
    print(len(reconstructed_images))
    
    sess = tf.Session()
    x = sess.run(reconstructed_images[101])   # 0.0~255.0 사이의 float값
    
    print(x.shape, reconstructed_file_names[101])
    io.imshow(x/127.5 -1.0)
    plt.show()
#############################################################
def TFRecord_reading3():
    import skimage.io as io
    import matplotlib.pyplot as plt
    def mydecode(serialized_example):
        features = tf.parse_single_example(serialized_example, features={'image/file_name': tf.FixedLenFeature([], tf.string), 'image/encoded_image': tf.FixedLenFeature([], tf.string),})
        
        
        image_buffer = features['image/encoded_image']
        file_name_buffer = features['image/file_name']
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.resize_images(image, size=(256, 256))
        
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image/127.5 -1.0
        
        image.set_shape([256, 256, 3])
        return image,file_name_buffer
	
    # tf.data.TFRecordDataset 이용하는 방식인데, 위에서 만든 example인 TFRecord_reading1()과 유사
    # 이 방식은 Coordinator 없이 iterator를 이용
    filename = 'D:\\hccho\\CycleGAN-TensorFlow-master\\data\\tfrecords\\apple.tfrecords'
    my_dataset = tf.data.TFRecordDataset(filename)
    
    my_dataset = my_dataset.map(mydecode)
    my_dataset = my_dataset.repeat()
    my_dataset = my_dataset.shuffle(buffer_size=100)
    
    iterator = tf.data.Iterator.from_structure(my_dataset.output_types, my_dataset.output_shapes)
    init_op = iterator.make_initializer(my_dataset)
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(init_op)
        x,y = sess.run(next_element)
        io.imshow(x)
        plt.title(y)
        plt.show()
    
        x,y = sess.run(next_element)
        io.imshow(x)
        plt.title(y)
        plt.show()    
#############################################################
def shuffle_batch():
    # shuffle_batch를 이용하는 또 다른 방식
    # 전체 data를 tf.train.slice_input_producer에 넣어 처리
    myDataX = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
    myDataY = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T
    
    X = tf.convert_to_tensor(myDataX, tf.float32)
    Y = tf.convert_to_tensor(myDataY, tf.float32)    
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
    batch_size= 4
    x, y = tf.train.shuffle_batch(input_queues,num_threads=8,batch_size=batch_size, capacity=batch_size*64, 
                                  min_after_dequeue=batch_size*32, allow_smaller_final_batch=False)       
    
    
    with tf.Session() as sess:
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        print(sess.run([x,y]))

        
        print(sess.run([x,y]))

        coord.request_stop()
        coord.join(threads)  
    
    print('Done')
	
#############################################################
def expand_and_concat():
    tf.reset_default_graph()
    y = tf.placeholder(tf.float32, [100,200,30], 'y')
    x = tf.placeholder(tf.float32, [200,40], 'x')
    x = tf.expand_dims(x,0)  # (1, 200, 40)
    x = tf.tile(x,[100,1,1]) # (100, 200, 40)
    
    z = tf.concat([x,y],axis=2) # (100, 200, 70)

#############################################################
def patial_initialization():
    # 초기화 되지 않은 변수만 초기화
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            self.sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    
    sess.run(init_new_vars_op)
#############################################################
def instance_normalization_test():

    import tensorflow as tf
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None,16,16,3])
    param_initializers = {'beta':tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32),'gamma': tf.ones_initializer()}
    #param_initializers = {'beta':tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32)}
    y = tf.contrib.layers.instance_norm(x,param_initializers=param_initializers)
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()
    beta = graph.get_tensor_by_name('InstanceNorm/beta:0')
    gamma = graph.get_tensor_by_name('InstanceNorm/gamma:0')
    print(sess.run([beta,gamma]))


def FC_vs_Conv2d():
    import tensorflow as tf
    import numpy as np
    tf.reset_default_graph()
    
    x = np.arange(12).reshape(3,4).astype(np.float32)
    w = np.arange(20).reshape(4,5).astype(np.float32)
    z1 = np.matmul(x,w)
    
    # kernel size=1, stride=1이므로, padding은 same이나 valid나 동일함.
    y=tf.layers.conv2d(tf.convert_to_tensor(x.reshape(3,1,1,4)),filters=5, kernel_size=1, strides=1,kernel_initializer=tf.constant_initializer(w.reshape(1,1,4,5)),use_bias=False)
    w=tf.layers.conv1d(tf.convert_to_tensor(x.reshape(3,1,4)),filters=5, kernel_size=1, strides=1,kernel_initializer=tf.constant_initializer(w.reshape(1,4,5)),use_bias=False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z2,z3 =sess.run([y,w])
        z2 = np.squeeze(z2)
        z3 = np.squeeze(z3)
    print(z1,z2,z3) 
###############################################
# kernel_size=1, strides=1 인 경우에는 valid padding 과 same padding은 동일한 결과를 준다.

T=4
x = np.arange(T*4).reshape(1,T,4).astype(np.float32)
x = tf.convert_to_tensor(x)
w = np.arange(20).reshape(4,5).astype(np.float32)
z1=tf.layers.conv1d(x,filters=5, 
                   kernel_size=1, strides=1,kernel_initializer=tf.constant_initializer(w.reshape(1,4,5)),
                   use_bias=False,padding='valid')

z2=tf.layers.conv1d(x,filters=5, 
                   kernel_size=1, strides=1,kernel_initializer=tf.constant_initializer(w.reshape(1,4,5)),
                   use_bias=False,padding='same')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(z1))
print(sess.run(z2))




###############################################
def conv2d_transpose():
    init = tf.contrib.layers.xavier_initializer(uniform=False)
    #init = tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32)
    X = np.arange(2*10*10*1).reshape(2,10,10,1).astype(np.float32)
    X=tf.convert_to_tensor(X)
    Y = tf.layers.conv2d(X,filters=2, kernel_size=3, strides=2,padding='same',kernel_initializer=init,use_bias=False)
    YY = tf.layers.conv2d_transpose(X,filters=2, kernel_size=(3,3), strides=(2,1),padding='same',kernel_initializer=init,use_bias=False)
    
    weight = tf.get_variable('weight',shape=(3,3,2,1),dtype=tf.float32)  # shape의 마지막 2개는 out_channel, in_channel 순이다.
    YYY = tf.nn.conv2d_transpose(X,weight,output_shape=(2,20,10,2),strides=[1,2,1,1],padding='SAME')
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()
    w = graph.get_tensor_by_name('conv2d/kernel:0')
    
    ww = sess.run(w)


###############################################
import threading
import tensorflow as tf
def basic_queue():
    
    mode = 1
    if mode==1:
        data_queue = tf.train.string_input_producer(["a.txt","b.txt","c.txt"],shuffle=False)
    elif mode==2:
        data_queue = tf.train.input_producer([1.1,2.2,3.33,4.45],shuffle=False)
        
        
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    
        for step in range(10):
    
            print(sess.run(data_queue.dequeue()) )
    
        coord.request_stop()
        coord.join(threads)

def basic_queue2():
    
    mode = 1
    if mode==1:
        QUEUE_LENGTH = 20
        data_queue = tf.FIFOQueue(QUEUE_LENGTH,"float")
        enq_ops1 = data_queue.enqueue_many(([1.0,2.0,3.0],) )
        enq_ops2 = data_queue.enqueue_many(([4.0,5.0,6.0],) )
        enq_ops3 = data_queue.enqueue_many(([6.0,7.0,8.0],) )
        qr = tf.train.QueueRunner(data_queue,[enq_ops1,enq_ops2,enq_ops3])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = qr.create_threads(sess, coord=coord, start=True)
    
        for step in range(10):
    
            print(sess.run(data_queue.dequeue()) )
    
        coord.request_stop()
        coord.join(threads)
##########################################################
def Thread_Example():
    import sys
    import threading
    
    class DestinationThread(threading.Thread):
        def __init__(self,target,args,kwargs):
            super(DestinationThread, self).__init__(target=target, args=args, kwargs=kwargs)
        def run(self):
            self._target(*self._args, **self._kwargs)
    
    def func(a, k):
        print("func(): a=%s, k=%s" % (a, k))
    
    thread = DestinationThread(target=func, args=(1,), kwargs={"k": 2})
    thread.start()
    thread.join()

###############################################

def map_structure_test():
    from tensorflow.python.util import nest
    
    def myfunc(inputs, outputs):
        # map_structure: 아래 예에서 inputs+0.00099 --> lambda의 inp에 전달, outputs/10.0 --> lambda의 out에 전달된다.
        return nest.map_structure(lambda inp, out: inp + out,inputs+0.00099, outputs/10.0)
    
    f = lambda inp, out: inp + out
    a = tf.constant([1.0,2.0])
    b = tf.constant([100.0,200.0])
    sess = tf.Session()
    print(sess.run(a+b))
    print(sess.run(myfunc(a,b)))
    print(sess.run(f(a,b)))
    
    [101. 202.]
    [11.00099 22.00099]
    [101. 202.]

###############################################
# manual gradient update

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
gradients, variables = zip(*optimizer.compute_gradients(loss))  # optimizer.compute_gradients(loss) 가 'list of (gradient, variable) pairs' return하기 때문에 zip으로 
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)   # 필요에 따라

optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),global_step=global_step)

또는 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
gradients_variables = optimizer.compute_gradients(loss)) 

optimize = optimizer.apply_gradients(gradients_variables),global_step=global_step)



###############################################
import tensorflow as tf
pred=tf.placeholder(dtype=tf.bool,name='bool')
cond = tf.constant([[1,-1],[-2,3]])
x = tf.constant([[10,20],[30,40]])
y = tf.constant([[-10,-20],[-30,-400]])
z1 = tf.where(tf.less(cond,0),x,y) # 조건식에 array가 오고, elementwise 계산된다.
z2 = tf.cond(tf.less(-1,0),lambda : x, lambda : y)  # 조건식에 scalar 값
sess = tf.Session()
print(sess.run(z1))
print(sess.run(z2))


[[ -10   20]
 [  30 -400]]
[[10 20]
 [30 40]]


###############################################
def recurrence(last_output, current_input):
    # 이 함수의 return 갯수는 last_output의 len과 같고, initalizer의 length와도 일치해야 한다.
    return (last_output[1], last_output[0] + last_output[1])
N = tf.placeholder(tf.int32, shape=(), name='N')
fibonacci = tf.scan(fn=recurrence,elems=tf.range(N), initializer=(10,37) )
with tf.Session() as session:
    o_val = session.run(fibonacci, feed_dict={N: 8})
    print("output:", o_val)
"""    
last_output = (10,37), current_input = 0  ==> (37,47)

last_output = (37,47), current_input = 1  ==> (47,84)

last_output = (47,84), current_input = 2  ==> (84,131)
"""

###############################################
# global_step & exponential_decay
result = []
sess = tf.Session()

lr = 0.01
w = tf.get_variable("test", [1], dtype=tf.float32)

global_step = tf.Variable(5, name='global_step', trainable=False)  # 초기값에 5을 넣으면, 5에서 출발함.

# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
lr = tf.train.exponential_decay(lr, global_step=global_step, decay_steps = 1000, decay_rate = 0.95)  # global_step이 1000이 될때, lr*0.95에 도달하는 속도로 감소

optimizer = tf.train.AdamOptimizer(lr).minimize(tf.abs(w),global_step=global_step)
sess.run(tf.global_variables_initializer())
for i in range(40000):
    _, lr_ = sess.run([optimizer, lr])
    result.append(lr_)
print(sess.run(global_step))
plt.plot(result)


###############################################
def dilation_conv_compare():
    """
    https://github.com/ibab/tensorflow-wavenet  의 dilation convolution 구현과
    tensorflow의 tf.layers.conv1d에서 dilation_rate을 지정했을 때와 비교
    ==> 결론: 일치함.
    
    
    """
    def time_to_batch(value, dilation, name=None):
        with tf.name_scope('time_to_batch'):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])
    
    def batch_to_time(value, dilation, name=None):
        with tf.name_scope('batch_to_time'):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])
    def causal_conv(value, filter_, dilation, name='causal_conv'):
        with tf.name_scope(name):
            filter_width = tf.shape(filter_)[0]
            if dilation > 1:
                transformed = time_to_batch(value, dilation)  # (?, ?, 32)
                conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
                restored = batch_to_time(conv, dilation)
            else:
                restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
            # Remove excess elements at the end.
            out_width = tf.shape(value)[1] - (filter_width - 1) * dilation # 이미 valid padding을 했기 때문에, 자를게 남아 있나? -->남아 있다. time_to_batch를 거치면서 추가적인 padding이 되었기 때문
            result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])  # index [0,0,0]에서 부터 크기 [-1,out_width, -1] 크기를 잘라낸다.
            return result
    t = tf.constant([0,1,2,3,4,2,0,2,4,3,2,0,3,1,2,3,4,2,0,2,4,3,2,0,3,1,2,3,4,2,0,2,4,3,])
    t = tf.reshape(t,(2,-1))
    
    
    dilation = 3
    filter_width=2 
    quantization_channels=5
    residual_channels=32
    
    input_batch=tf.one_hot(t, quantization_channels)
    
    
    xx = tf.layers.conv1d(input_batch,filters=residual_channels,kernel_size=filter_width,padding='valid',dilation_rate=dilation,use_bias=False)
    graph = tf.get_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    yy=sess.run(xx)
    
    w0 = sess.run(graph.get_tensor_by_name('conv1d/kernel:0'))
    
    ###############
    #weights_filter = tf.get_variable('weight',shape=[filter_width,quantization_channels,residual_channels])
    weights_filter = tf.get_variable('weight',initializer= tf.constant(w0))
    
    x=causal_conv(input_batch, weights_filter, dilation)
    sess.run(tf.initialize_variables([weights_filter]))
    y=sess.run(x)
    
    print(np.array_equal(y,yy))  # numpy.testing.assert_allclose  <--- 오차 범위내에서 
###############################################
# dilation 연산을 행렬곱으로 변환하여 연산한 결과와 비교
def dilation_check():
    batch_size=2
    T=10
    c_in=2
    c_out=3
    kernel_size=4
    dilation = 3
    strides = 1
    
    T = dilation*(kernel_size-1) + 1  # 이렇게 잡아여, 연산후 길이가 1이 된다.
    x = np.random.normal(size=[batch_size,T,c_in])
    xx = x[:,0::dilation,:]
    
    x = tf.convert_to_tensor(x)
    xx = tf.convert_to_tensor(xx)
    w = np.random.normal(size=[kernel_size,c_in,c_out]).astype(np.float64)
    z1=tf.layers.conv1d(x,filters=c_out,kernel_size=kernel_size, strides=1,dilation_rate=3,kernel_initializer=tf.constant_initializer(w),
                       use_bias=False,padding='valid')
    
       
    linearized_weights = tf.reshape(tf.convert_to_tensor(w),[-1,c_out]) #(kernel_size,c_in,c_out) ==> (kernel_size*c_in,c_out)
    z2 =  tf.matmul(tf.reshape(xx,[batch_size,-1]),linearized_weights)  # xx: (batch_size,kernel_size,c_in) ==> (batch_size,kernel_size*c_in)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    z1_=sess.run(z1)
    z2_=sess.run(z2)
    
    print(z1_)
    print(z2_)
###############################################
 def dilation_speed_test():
    # conv1d로 dilation했을 때와, matmul로 했을 때의 속도 비교
    # ==> 결론: matmul이 gpu에서는 훨씬 빠르고(45초 vs 4.5초), cpu에서는 약간 빠르다(9.76 초 vs 9.19초)
    batch_size=2
    
    c_in=256
    c_out=256
    kernel_size=4
    dilation = 3
    strides = 1
    
    T = dilation*(kernel_size-1) + 1  # 이렇게 잡아여, 연산후 길이가 1이 된다.
    x = np.random.normal(size=[batch_size,T,c_in]).astype(np.float32)
    xx = x[:,0::dilation,:]
    
    x = tf.convert_to_tensor(x)
    xx = tf.convert_to_tensor(xx)
    w = np.random.normal(size=[kernel_size,c_in,c_out]).astype(np.float64)
    layer  = tf.layers.Conv1D(filters=c_out,kernel_size=kernel_size,dilation_rate=dilation, strides=1,kernel_initializer=tf.constant_initializer(w),
                       use_bias=False,padding='valid')
    
    layer.build((1,1,c_in))   #마지막 dim만 의미가 있다.  build가 없다면, z1=layer(x)를 해서 kernel이 잡힌다.
    
    z2 = tf.matmul(tf.reshape(xx,(batch_size,-1)), tf.reshape(layer.kernel,(-1,c_out)))
    z1 = layer(x)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    s = time.time()
    for i in range(20000):
        z1_= sess.run(z1)
    e = time.time()
    
    print(e-s,"sec")
    
    
    s = time.time()
    for i in range(20000):
        z2_= sess.run(z2)
    e = time.time()
    
    print(e-s,"sec")
    print(np.allclose(np.squeeze(z1_),z2_))
    

###############################################
def padding_test():
    # valid 를 적용할 때, 어떤 부분을 잘라내는것인가? 정답은 끝부분을 잘라냄. 가운데를 도려내는 방식은 아님.
    X = np.arange(11).reshape(1,-1,1).astype(np.float32)
    X = tf.convert_to_tensor(X)
    Y = tf.layers.conv1d(X,filters=1,kernel_size=3,strides=3,padding='valid',use_bias=False,kernel_initializer= tf.initializers.ones())
    
    sess = tf.Session()
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    xx,yy=sess.run([X,Y])
    
    w0 = sess.run(graph.get_tensor_by_name('conv1d/kernel:0'))
    
    print(w0)
    print(xx)
    print(yy)
###############################################
# wavenet ibab구현 방식
# placeholder를 만들 때, batch 크기에 대한 것이 없는 상태로 만들고, dequeue_many를 활용하여 batch 만큼 묶는다.
# enqueue는 data를 만들고, dequeue는 data를 꺼낸다.
# start_threads(thread_main call) --> thread_main
# 외부에서 dequeueX, dequeueY로 필요한 data를 가져가는 구조.
# queue를 2개로 분리하는 것이 좋지 못하기 때문에, 이 후 사람들(e.g. wavenet vocoder-azraelkuan)은 구현은 queue를 하나로 묶어서 구현하고 있다.
import threading
class MyDataFeed():
    def __init__(self,coord):
        self.coord = coord
        self.threads = []
        
        
        #dequeue_many를 사용하므로, batch 크기는 지정할 필요 없다.
        self.placeholder_dataX1 = tf.placeholder(dtype=tf.float32, shape=[2,3])
        self.placeholder_dataX2 = tf.placeholder(dtype=tf.int32, shape=[1])
        queue_size= 32
        self.queueX = tf.FIFOQueue(queue_size, [tf.float32,tf.int32],shapes=[(2,3),(1)])
        self.enqueueX = self.queueX.enqueue([self.placeholder_dataX1,self.placeholder_dataX2])
        
        # 위에서 만든 queueX와 별도로 queueY를 만들었는데, data의 pair가 맞아야 한다면, 분리하여 만드는 것이 바람직하지 않다.
        self.placeholder_dataY = tf.placeholder(dtype=tf.int32, shape=[1])
        self.queueY = tf.FIFOQueue(queue_size, [tf.int32],shapes=[(1)])
        self.enqueueY = self.queueY.enqueue([self.placeholder_dataY])
        
                
    def thread_main(self, sess):
        stop = False
        while not stop:
            
            for _ in range(10):  # 한번에 처리하고 싶은 만큼
                if self.coord.should_stop():
                    stop = True
                    break
                x = np.random.normal(0,1,6).reshape(2,3)
                #enqueueX,enqueueY의 쌍은 start_threads에서 thread=1일 때만 맞다. 그런데, thread = 1 일 때는 data생성 속도가 느려, 같은 data를 반복해서 내보는 경우가 있다. 
                #그래서 pair를 이루는 data를 지금과 같이 queueX,queueY로 분리하는 것은 좋지 못하다. queueX하나에서 묶어 처리하는 것이 바람직 하다.
                sess.run(self.enqueueX, feed_dict={self.placeholder_dataX1: x, self.placeholder_dataX2: [100*x[0][1]]})
                sess.run(self.enqueueY, feed_dict={self.placeholder_dataY: [100*x[0][0]]})
                
                
    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
    
    
    def dequeueX(self, num_batch):  # num_elements <--- batch_size를 의미함.
        return self.queueX.dequeue_many(num_batch)
    def dequeueY(self, num_batch):
        return self.queueY.dequeue_many(num_batch)    

coord = tf.train.Coordinator()
mydatafeed = MyDataFeed(coord)
my_batchX = mydatafeed.dequeueX(num_batch=2)
my_batchY = mydatafeed.dequeueY(num_batch=2)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

threads = tf.train.start_queue_runners(sess=sess, coord=coord)
mydatafeed.start_threads(sess,n_threads=1)  # n_threads 개수가 충분해야 속도을 맞출 수 있다. 부족하면, 같은 걸 return한다. --> thread >1이면 my_batchX,my_batchY 쌍의 깨진다.

try:
    for step in range(7):
        a,b,c = sess.run([my_batchX[0],my_batchX[1],my_batchY])  #queue에 할당된 op가 관련된 계산일 때만, Queue가 data를 가져온다.
        print(step,a,b,c)
        

except KeyboardInterrupt:
    # Introduce a line break after ^C is displayed so save message
    # is on its own line.
    print('KeyboardInterrupt')  # 이건 잘 작동하지 않음
finally:
    print('finally')
    coord.request_stop()
    coord.join(threads)

###############################################
# tacotron의 keith ito의 구현방식
# placeholder로 retrun 되는 것이 명확하다.
# data feed 용 class를 만들때, threading.Thread를 상속받아 sub class로 만든다.
import threading

class MyDataFeed2(threading.Thread):
    def __init__(self,coord,batch_size):
        super(MyDataFeed2, self).__init__()
        self.coord = coord
        self.batch_size = batch_size
        self.placeholder_dataX1 = tf.placeholder(dtype=tf.float32, shape=[None,2,3])
        self.placeholder_dataX2 = tf.placeholder(dtype=tf.int32, shape=[None,1])
        queue_size= 32
        self.queueX = tf.FIFOQueue(queue_size, [tf.float32,tf.int32])
        self.enqueueX = self.queueX.enqueue([self.placeholder_dataX1,self.placeholder_dataX2])
        
        self.X1, self.X2= self.queueX.dequeue()
        self.X1.set_shape(self.placeholder_dataX1.shape)
        self.X2.set_shape(self.placeholder_dataX2.shape)
    def run(self):
        try:
            while not self.coord.should_stop():
                self.make_batches()
        except Exception as e:
            self.coord.request_stop(e)       
    def start_in_session(self, session):
        self.sess = session
        self.start()
              
    def make_batches(self):
        
        for _ in range(10): # batch size만큼의 data를 원하는 만큼 만든다.
            x = np.random.normal(0,1,self.batch_size*6).reshape(-1,2,3)
            self.sess.run(self.enqueueX, feed_dict={self.placeholder_dataX1: x, self.placeholder_dataX2: 100*x[:,0,1].reshape(self.batch_size,-1)})


coord = tf.train.Coordinator()
mydatafeed = MyDataFeed2(coord,batch_size=2)


with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        start_step = 0
        mydatafeed.start_in_session(sess)
        
        while not coord.should_stop():
	    # sess.run이 돌아갈 때마다, data 1set가 사용된다.
            a,b = sess.run([mydatafeed.X1,mydatafeed.X2])
            print(start_step,a,b)
            
            a = sess.run(mydatafeed.X1)
            print(start_step,a)

            b = sess.run(mydatafeed.X2)  # 바로 위의 a와는 별도.
            print(start_step,b)
	
            start_step +=1
            if start_step >= 10:
                # error message가 나오지만, 여기서 멈춘 것은 맞다.
                raise Exception('End xxx')
    except Exception as e:
        print('finally')
        coord.request_stop(e)

###############################################
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    w1 = tf.get_variable(name='w', shape=(2, 3, 5))

with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    w2 = tf.get_variable(name='w')  # 이미 선언된 변수 가져오기

###############################################
def TF_Variables_Update():
    tf.reset_default_graph()
    
    A = [ tf.Variable(initial_value=tf.zeros(shape=[3,4], dtype=tf.float32), name='A1', trainable=False), 
         tf.Variable(initial_value=tf.zeros(shape=[7,4], dtype=tf.float32), name='A2', trainable=False)]
    
    B = tf.ones(shape=[4,1],dtype=tf.float32)
    C = tf.matmul(A[0],B)   
    x = tf.placeholder(tf.float32)
    
    method = 2
    if method == 0:
        A[0] = tf.scatter_update(A[0],tf.range(tf.shape(A[0])[0]), tf.concat([A[0][1:],tf.ones([1,4])*x],axis=0))
    elif method==1:
        A[0] = tf.scatter_update(A[0],tf.range(tf.shape(A[0])[0]-1), A[0][1:])
        A[0] = tf.scatter_update(A[0],[2], tf.ones([1,4])*x)                                           
    elif method==2:
        # 이 방법은 항상 같은 결과를 얻지 못한다. 이 방법은 사용하며 안됨
        X = []
        X.append(tf.scatter_update(A[0],tf.range(tf.shape(A[0])[0]-1), A[0][1:])) 
        X.append(tf.scatter_update(A[0],[2], tf.ones([1,4])*x))                                                           
                  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if method==2:
            sess.run(X,feed_dict={x:2.0})
            sess.run(X,feed_dict={x:4.0})
            b = sess.run(X,feed_dict={x:5.0})
    
    	    # [A,X]를 run 하는데 있어, X는 X[0],X[1]이 있다. X[0],A,X[1]순으로 연산이 될 경우도 있다.
            a,_ = sess.run([A,X],feed_dict={x:-3.0})  # [A,X]를 같이 연산하기 때문에, A,X의 계산순서가 일정하지 못하다.  
            print(a) 
        else:
            a = sess.run(A,feed_dict={x:2.0})
            print(a)
            
            a = sess.run(A,feed_dict={x:4.0})
            print(a)
                     
            a = sess.run(A,feed_dict={x:5.0})
            print(a)
              
            a = sess.run(A,feed_dict={x:-3.0})
            print(a) 

###############################################
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
assert_op = tf.assert_equal(x,y,data=[x,y])
with tf.control_dependencies([assert_op]):
    c = x+y

with tf.Session() as sess:
    print(sess.run(c,feed_dict={x:3,y:4}))

###############################################
# array의 특정 index 추출. gather, gather_nd
x = tf.placeholder(tf.float32,shape=[3,3])
y = tf.placeholder(tf.int32,shape=[None,2])
z = tf.gather_nd(x,y)

with tf.Session() as sess:
    w = sess.run(z,feed_dict={x:[ [1.1, 2.2, 3.3] ,[4.4, 5.5, 6.6], [7.7, 8.8, 9.9] ], y:[[0,1],[1,2]]})

###############################################
Tensorflow matmul: shape(N,m,n)과 shape(N,n,k) ==> (N,m,k)
shape(N,m,n)과 (

###############################################
x = np.arange(20).reshape(2,2,5).astype(np.float32)
y = tf.convert_to_tensor(x)
# axix = 0을 제외하고, 평균, 분산으로 normalization
z = tf.contrib.layers.layer_norm(y)


graph = tf.get_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(z)
gamma,beta = sess.run([graph.get_tensor_by_name('LayerNorm/gamma:0'),graph.get_tensor_by_name('LayerNorm/beta:0')])
print("gamma", gamma)
print("beta", beta)
print("result", result)
m = np.mean(x,axis=(1,2),keepdims=True)
s = np.std(x,axis=(1,2),keepdims=True)
print("check", (x-m)/s)

###############################################
a = np.arange(20).reshape(4,5).astype(np.float32)      
"""
array([[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.],
       [10., 11., 12., 13., 14.],
       [15., 16., 17., 18., 19.]], dtype=float32)
"""
x = tf.constant(a)

# gather는 첫번째 index에 대해서만 추출이 가능하다.
y1 = tf.gather(x,0) # shape(5,)  [0., 1., 2., 3., 4.]
y2 = tf.gather(x,[0])  # shape(1,5)  [[0., 1., 2., 3., 4.]]
y3 = tf.gather(x,[[1,2],[2,3]]) # [ a[1],a[2]] ,[a[2],a[3]] ]

z= tf.slice(x,begin= [2,1], size=[2,3])



w1 = tf.gather_nd(x,[[0],[2]])  # [[ 0.,  1.,  2.,  3.,  4.],[10., 11., 12., 13., 14.]]
w2 = tf.gather_nd(x,[[0,2],[2,1]])  # [ 2., 11.]

###############################################
	
	
	
###############################################


###############################################
	
	
###############################################
	
###############################################
	
	
###############################################
	
	
###############################################
	
if __name__ == "__main__":   
    test1()
    
    #test2()
    #test3()
    #testPlaceholde()
    #testLinearRegression()
    #testLinearRegression2()
    
    
    
