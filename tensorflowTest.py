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

def TFRecord_reading1():
    # tfrecord에서binary data가 저장되어 있는데, tf.image.decode_jpeg로 이용해서 0~255 사이 값으로 변환한다.
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
    a,z=sess.run([images,file_names])
    b=sess.run(images)
    
    print(a.shape,b.shape)
    print(np.mean([a[0],b[0]],axis=(1,2,3)))
    
    print(z)
    
    io.imshow(np.concatenate(a,axis=1))
    plt.show()
    
def TFRecord_reading2():
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

def expand_and_concat():
    tf.reset_default_graph()
    y = tf.placeholder(tf.float32, [100,200,30], 'y')
    x = tf.placeholder(tf.float32, [200,40], 'x')
    x = tf.expand_dims(x,0)  # (1, 200, 40)
    x = tf.tile(x,[100,1,1]) # (100, 200, 40)
    
    z = tf.concat([x,y],axis=2) # (100, 200, 70)


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
    
    y=tf.layers.conv2d(tf.convert_to_tensor(x.reshape(3,1,1,4)),filters=5, kernel_size=1, strides=1,kernel_initializer=tf.constant_initializer(w.reshape(1,1,4,5)),use_bias=False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z2 =sess.run(y)
        z2 = np.squeeze(z2)
        
    print(z1,z2)    



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
        return nest.map_structure(lambda inp, out: inp + out,inputs+0.0002, outputs)
    
    f = lambda inp, out: inp + out
    a = tf.constant([1.0,2.0])
    b = tf.constant([100.0,200.0])
    sess = tf.Session()
    print(sess.run(a+b))
    print(sess.run(myfunc(a,b)))
    print(sess.run(f(a,b)))

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
    
    
    
