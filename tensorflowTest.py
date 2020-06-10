os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU Only Mode
============
error: 
WARNING:tensorflow:Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x000001BECB4B9080>> 
could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. 
When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. 
Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x000001BECB4B9080>>: 
AssertionError: Bad argument number for Name: 3, expecting 4

https://github.com/tensorflow/autograph/issues/1
pip install --user gast==0.2.2
==================

with tf.variable_scope('hccho1'):

    x = tf.placeholder(tf.float32,shape=[2,3])
    x = tf.convert_to_tensor(np.random.randn(2,3).astype(np.float32))
    x = tf.convert_to_tensor(np.array([[-0.6587036 ,  0.67638916, -0.07040939],[ 0.02193491, -0.13528223,  1.2818061 ]], dtype=np.float32))
    y = tf.layers.dense(x,units=10)    
    
with tf.variable_scope('hccho2'): 
    z = tf.layers.batch_normalization(y)

loss = tf.reduce_mean(z)

# optimization관련 op를 만들때도 scope가 있어야 한다. 그렇지 않으면,
# optimization 관련 variable이 위에 있는 scope를 찾아가서 붙는다.
with tf.variable_scope('hccho3'):   # optimization관련 op를 만들때도 scope가 있어야 한다. 
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hccho1') +tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hccho2')

print(sess.run(z))

saver = tf.train.Saver(var_list)
saver.restore(sess,'c:\\a\\model.ckpt')
print(sess.run(z))


saver = tf.train.Saver(var_list)
saver.save(sess, 'c:\\a\\model.ckpt')

######################################################################


# remove warning
# TensorFlow에서는 5가지의 로깅 타입을 제공하고 있습니다. ( DEBUG, INFO, WARN, ERROR, FATAL ) INFO가 설정되면, 그 이하는 다 출력된다.
tf.logging.set_verbosity(tf.logging.ERROR)
# TF 1.14
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

모두 출력하지 않게...
import logging
logging.getLogger('tensorflow').disabled = True
######################################################################
tf.shape(x) --> op
x.get_shape().as_list()  --> list. 이 경우는 [None,3] 이런 식으로 될 수도 있다.
######################################################################

np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
######################################################################
def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)

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


	#
	grad1 = tf.gradients(cost,[W,b])  # gradiend 값만
	grad2 = optimizer.compute_gradients(cost,[W,b])  # gradient와 trainable variable이 쌍으로.

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

=================================================
import tensorflow as tf
import numpy as np
def test_gpu():
    c = []
    for d in ['/gpu:0']:    # ['/gpu:1','/gpu:2']
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print (sess.run(sum)) 

test_gpu()
===============================================================
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
    """	
    with open('var_list.txt', 'w') as f:
        for item in var_list:
            f.write("%s\n" % item[0])
   """

    #sess = tf.Session()
    for v in var_list: 
        print(v) # tuple(variable name, [shape])
        vv = checkpoint_utils.load_variable(checkpoint_dir, v[0])
        print(vv) #values   
 
def init_from_checkpoint():
    #ckpt로 부터 특정 값만 뽑아내어, 선언한 변수 초기화 하기.
    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    
    
    checkpoint_dir = 'D:\\hccho\\Tacotron-2-hccho\\ver1\\logdir-tacotron2\\moon+son_2019-02-27_00-21-42\\model.ckpt-56000' # 구체적으로 명시
    #checkpoint_dir = 'D:\\hccho\\cs231n-Assignment\\assignment3\\save-sigle-layer' # 디렉토리만 지정 ==> 가장 최근
    var_list = checkpoint_utils.list_variables(checkpoint_dir)
    
    
    #1 직접 선언한 variable을 ckpt로부터 값 불러와 초기화
    vv = checkpoint_utils.load_variable(checkpoint_dir, var_list[50][0])  #  var_list[50][0]<--name,  var_list[50][1]<-- shape
    w = tf.get_variable('var1', shape=vv.shape)
    tf.train.init_from_checkpoint(checkpoint_dir,{var_list[50][0]: w})  # var_list[50]에 있는 값이 w로 할당된다. sess.run(tf.global_variables_initializer()) 해야 값이 할당된다.
    
    
    
    #2 tf.layers.dense로 간접적으로 선언된 variable을 ckpt로부터 값 불러와 초기화
    vv2 = checkpoint_utils.load_variable(checkpoint_dir, var_list[141][0])
    X = np.arange(3*16).reshape(3,16).astype(np.float32)  # var_list[141]의  shape확인 후, 잡았다.
    Y = tf.layers.dense(tf.convert_to_tensor(X),units=2048,name='hccho')
    tf.train.init_from_checkpoint(checkpoint_dir,{var_list[141][0]: 'hccho/kernel'})
    
    
    graph = tf.get_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    ww,kk = sess.run([w,graph.get_tensor_by_name('hccho/kernel:0')])
    print(np.allclose(ww,vv))
    print(np.allclose(kk,vv2))
=======================
참고: 
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v2_50"))
variables_in_checkpoint = tf.train.list_variables('.\\ckpt\\resnet_v2_50.ckpt')

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
    #w = sess.run('dense/kernel:0')   <----- 이렇게 해도된다. tensor이름만 으로도 된다.
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
    myDataY = np.array([[0,1,2,3,4,5,6,7,8,9,10,11]]).astype(np.float32).T
    
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
        
    	for i in range(20):
            print(sess.run([x,y]))

        coord.request_stop()
        coord.join(threads)  
    
    print('Done')
#############################################################
def data_gen():
    batch_size = 2
    def g():
        while True:
            a = np.random.randn(batch_size,3).astype(np.float32)
            #b = np.random.randint(10, size=(batch_size,1))
            b = a.astype(np.int32)
            yield a,b
    #dataset = tf.data.Dataset.from_generator(g, (tf.float32, tf.int32))
    dataset = tf.data.Dataset.from_generator(g, (tf.float32, tf.int32), (tf.TensorShape([None,3]), tf.TensorShape([None,3])))
    iterator = dataset.make_one_shot_iterator()
    
    X,Y = iterator.get_next()
    
    sess = tf.Session()
    
    for i in range(5):
        x,y = sess.run([X,Y])
        print(i, x,y)
#############################################################
from konlpy.tag import Kkma,Okt
def Make_Batch():
    # 이 example도 data가 simple할 때는 가능하지만, mini batch별로 조작을 어떻게 해야하는지???  ---> dataset.batch, dataset.mpa순서 조정으로 
    from tensorflow.keras import preprocessing
    samples = ['너 오늘 아주 이뻐 보인다', 
               '나는 오늘 기분이 더러워', 
               '끝내주는데, 좋은 일이 있나봐', 
               '나 좋은 일이 생겼어', 
               '아 오늘 진짜 너무 많이 정말로 짜증나', 
               '환상적인데, 정말 좋은거 같아']
    
    label = [[1], [0], [1], [1], [0], [1]]
    MAX_LEN = 5
    
    tokenizer = preprocessing.text.Tokenizer(oov_token="<UKN>")   # oov: out of vocabulary
    tokenizer.fit_on_texts(samples+['SOS','EOS'])
    print(tokenizer.word_index)
    sequences = tokenizer.texts_to_sequences(samples)
    '''
    [[5, 2, 6, 7, 8],
     [9, 2, 10, 11],
     [12, 3, 4, 13],
     [14, 3, 4, 15],
     [16, 2, 17, 18, 19, 20, 21],
     [22, 23, 24, 25]]
    '''
    
    okt = Okt()
    samples2 = [okt.morphs(x) for x in samples]
    
    tokenizer2 = preprocessing.text.Tokenizer(oov_token="<UKN>")   # oov: out of vocabulary
    tokenizer2.fit_on_texts(samples2+['SOS','EOS'])
    print(tokenizer2.word_index)
    sequences2 = tokenizer2.texts_to_sequences(samples2)
    '''
    [[7, 2, 8, 9, 10],
     [4, 11, 2, 12, 13, 14],
     [15, 5, 3, 6, 16, 17],
     [4, 3, 6, 18],
     [19, 2, 20, 21, 22, 23, 24],
     [25, 26, 27, 5, 28, 3, 29, 30]]
    
    '''
    
    sequences3 = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, padding='post',truncating='post')
    '''
    array([[ 5,  2,  6,  7,  8],
           [ 9,  2, 10, 11,  0],
           [12,  3,  4, 13,  0],
           [14,  3,  4, 15,  0],
           [16,  2, 17, 18, 19],
           [22, 23, 24, 25,  0]])
    '''
    
    word_index = tokenizer.word_index
    
    BATCH_SIZE = 2
    EPOCH = 2
    
    def mapping_fn(X, Y=None):
        # dataset.map(mapping_fn) ---> dataset.batch(BATCH_SIZE) 순서인 경우
        # X: <tf.Tensor 'args_0:0' shape=(5,) dtype=int32>  Y: <tf.Tensor 'args_1:0' shape=(1,) dtype=int32>
        
        # dataset.batch(BATCH_SIZE) --> dataset.map(mapping_fn) 순서인 경우
        # X: <tf.Tensor 'args_0:0' shape=(?, 5) dtype=int32>, Y: <tf.Tensor 'args_1:0' shape=(?, 1) dtype=int32>
        
        data_X = {'xx': X}  # dict로 보낼 때, key는 
        data_Y = Y # label = {'yy': Y}로 해도 된다.
        return data_X, data_Y  # data_X은 dict, data_Y은 numpy array로 
    
    dataset = tf.data.Dataset.from_tensor_slices((sequences3, label))
    
    dataset = dataset.shuffle(len(sequences3))
    
    dataset = dataset.map(mapping_fn)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next_data)) # <---tuple, next_data[0],next_data[1]로 각각 접근 가능
            except tf.errors.OutOfRangeError:
                break
#############################################################
def mapping_fn(a,b):
    # a,b는 tensor이다.
    print('----',a,b)
    return -a, b
def mapping_fn2(a,b,c):
    # a,b는 tensor이다.
    print('----',a,b,c)
    return -a, b
X1 = np.random.randn(10,5,8)
X2 = np.random.randn(10,4)

dataset = tf.data.Dataset.from_tensor_slices((X1,X2))

dataset = dataset.shuffle(buffer_size=10000)

# dataset.map, dataset.batch의 순서가 중요하다. 
# dataset.map, dataset.batch  순서이면 mapping_fn에는 batch로 묶이지 않은 data가 넘어 간다.
# dataset.batch, dataset.map  순서이면 mapping_fn에는 batch로 묶인 data가 넘어간다.

dataset = dataset.map(mapping_fn)
#dataset = dataset.map(lambda x,y: mapping_fn(x,y,0.5))
dataset = dataset.batch(2)

dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()


i,j = iterator.get_next()


with tf.Session() as sess:
	
	ii,jj = sess.run([i,j])
	print(ii.shape,jj.shape)

	ii,jj = sess.run([i,j])
	print(ii.shape,jj.shape)


##############################################################
1. 작은 data
tf.data.Dataset.from_tensor_slices((numpy array1, numpy array2, ...)) <------------- 1G 이하 data


2. 큰 data를 이용하기 위해서는 tfrecord 파일로 저장해야됨. npz는 잘 안됨
# tf.data.TFRecordDataset이 seriialized file을 다루기 때문.
# https://www.tensorflow.org/guide/datasets#consuming_tfrecord_data <---- Tensorflow에서는 tfrecord파일을 사용하라고 추천하고 있다.
# npz를 사용하려면, tf.data.Dataset.from_tensor_slices + mapping function을 정의해서 사용할 수 있음.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]  
또는 filenames = tf.placeholder(tf.string, shape=[None]) <--- tain, valid data 구분이 있을 때는 이렇게 하는 것이 좋다.( feed function이 분리되어 있으면 위에 처럼 해도...)




dataset = tf.data.TFRecordDataset(filenames)

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

3. data의 길이가 일정하지 않은 것은 npz로 묶어서 저장할 수 없다. npz에는 data 1개씩.
   길이가 다른 data를 여러개 묶어 저장하려면, tfrecord로 저장해야 한다.
#############################################################
# https://stackoverflow.com/questions/53938962/in-tensorflow-dataset-api-how-to-use-padded-batch-so-that-a-pads-with-a-specifi
# http://cs230.stanford.edu/blog/datapipeline/
def padded_batch_test():
   # 위에서는 dataset.batch(BATCH_SIZE)를 사용했는데, 여기서는 ataset.padded_batch를 사용
   # 길이가 일정하지 않아, 바로  tf.data로 넘어가지 않아, tfrecords에 저장했다가 다시 읽음.
	cells = np.array([[0,1,2,3], [2,3,4], [3,6,5,4,3], [3,9]])
	mells = np.array([[0,1], [2], [3,2], [9,1,2]])
	print(cells)

	writer = tf.python_io.TFRecordWriter('test.tfrecords')
	for index in range(mells.shape[0]):
		example = tf.train.Example(features=tf.train.Features(feature={
			'num_value':tf.train.Feature(int64_list=tf.train.Int64List(value=mells[index])),
			'list_value':tf.train.Feature(int64_list=tf.train.Int64List(value=cells[index]))
		}))
		writer.write(example.SerializeToString())
	writer.close()

	#Generate Samples with batch size of 2

	filenames = ["test.tfrecords"]
	dataset = tf.data.TFRecordDataset(filenames)
	def _parse_function(example_proto):
	# tf.VarLenFeature: Configuration for parsing a variable-length input feature.
	# 참고로 FixedLenFeature도 있다.

		keys_to_features = {'num_value':tf.VarLenFeature(tf.int64),
							'list_value':tf.VarLenFeature(tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, keys_to_features)
		return tf.sparse.to_dense(parsed_features['num_value']), \
			   tf.sparse.to_dense(parsed_features['list_value'])
	# Parse the record into tensors.
	dataset = dataset.map(_parse_function)
	# Shuffle the dataset
	dataset = dataset.shuffle(buffer_size=1)
	# Repeat the input indefinitly
	dataset = dataset.repeat()  
	dataset = dataset.prefetch(20)  # buffer_size = 20

	# Generate batches
	# padded_shapes: None이면 가장 큰 data 기준.
	dataset = dataset.padded_batch(2, padded_shapes=([None],[8]), 
		  padding_values=(tf.constant(-99, dtype=tf.int64), tf.constant(-188, dtype=tf.int64)))
	# Create a one-shot iterator
	iterator = dataset.make_one_shot_iterator()
	i, data = iterator.get_next()

	with tf.Session() as sess:
		print(sess.run([i, data]))
		print(sess.run([i, data]))
	
출력 결과:
[array([[  0,   1],[  2, -99]], dtype=int64), 
 array([[   0,    1,    2,    3, -188, -188, -188, -188],[   2,    3,    4, -188, -188, -188, -188, -188]], dtype=int64)]
[array([[  3,   2, -99],[  9,   1,   2]], dtype=int64), 
 array([[   3,    6,    5,    4,    3, -188, -188, -188],[   3,    9, -188, -188, -188, -188, -188, -188]], dtype=int64)]

############################################################
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



def FC_vs_Conv2d_2():
	# (3,3,5)이미지를 [flatten 후, FC(eg 45->4)를 거치는 것]과 [kernel_size=3, out_channel = 4로 conv2d]

	x0 = np.random.randn(2,3,3,5)
	x = tf.convert_to_tensor(x0)
	xx = tf.layers.flatten(x)
	w = np.random.randn(3,3,3,4)
	ww = w.reshape(-1,4)
	z1=tf.layers.conv2d(x,filters=4,kernel_size=3,kernel_initializer=tf.constant_initializer(w), use_bias=False,padding='valid')
	z2 =tf.layers.dense(xx,units=4,kernel_initializer=tf.constant_initializer(ww),use_bias=False)



	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	print(sess.run(z1))
	print(sess.run(z2))






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
gradients_variables = optimizer.compute_gradients(loss)

optimize = optimizer.apply_gradients(gradients_variables,global_step=global_step)



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
def dilation_check1():
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

def dilation_check2():
    # dilation_check1은 output 길이가 1이 되는 특수한 경우. 이번에는 output 길이 2가 되는 경우.
    batch_size=2
    T=10
    c_in=2
    c_out=3
    kernel_size=4
    dilation = 3
    strides = 1
    
    T = dilation*(kernel_size-1) + 2  # 이렇게 잡아여, 연산후 길이가 2이 된다.
    x = np.random.normal(size=[batch_size,T,c_in])
    
    #xx = x[:,0::dilation,:]
    xx = np.concatenate([x[:,0::dilation,:],x[:,1::dilation,:]],axis=1)  # 길이 2이기 때문에...2개만 concat
    
    x = tf.convert_to_tensor(x)
    xx = tf.convert_to_tensor(xx)
    w = np.random.normal(size=[kernel_size,c_in,c_out]).astype(np.float64)
    z1=tf.layers.conv1d(x,filters=c_out,kernel_size=kernel_size, strides=1,dilation_rate=3,kernel_initializer=tf.constant_initializer(w),
                       use_bias=False,padding='valid')
    
       
    linearized_weights = tf.reshape(tf.convert_to_tensor(w),[-1,c_out]) #(kernel_size,c_in,c_out) ==> (kernel_size*c_in,c_out)
    z2 =  tf.matmul(tf.reshape(xx,[-1,kernel_size*c_in]),linearized_weights)  # xx: (batch_size,kernel_size,c_in) ==> (batch_size,kernel_size*c_in)
    z2 = tf.reshape(z2,[batch_size,-1,c_out])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    z1_=sess.run(z1)
    z2_=sess.run(z2)
    
    print(z1_)
    print(z2_)

def dilation_check2():
    # 모든 길이에 대해 일반화
    batch_size=2
    T=10
    c_in=2
    c_out=3
    kernel_size=4
    dilation = 3
    strides = 1
    
    out_len=9
    T = dilation*(kernel_size-1) + out_len  # 이렇게 잡아여, 연산후 길이가 2이 된다.
    x = np.random.normal(size=[batch_size,T,c_in])
    
    #xx = x[:,0::dilation,:]
    xx = []
    for i in range(out_len):
        xx.append(x[:,i::dilation,:][:,:kernel_size,:])
    xx = np.concatenate(xx,axis=1)
    
    x = tf.convert_to_tensor(x)
    xx = tf.convert_to_tensor(xx)
    w = np.random.normal(size=[kernel_size,c_in,c_out]).astype(np.float64)
    z1=tf.layers.conv1d(x,filters=c_out,kernel_size=kernel_size, strides=1,dilation_rate=3,kernel_initializer=tf.constant_initializer(w),
                       use_bias=False,padding='valid')
    
       
    linearized_weights = tf.reshape(tf.convert_to_tensor(w),[-1,c_out]) #(kernel_size,c_in,c_out) ==> (kernel_size*c_in,c_out)
    z2 =  tf.matmul(tf.reshape(xx,[-1,kernel_size*c_in]),linearized_weights)  # xx: (batch_size,kernel_size,c_in) ==> (batch_size,kernel_size*c_in)
    z2 = tf.reshape(z2,[batch_size,-1,c_out])
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
assert_op = tf.assert_equal(x,y,data=[x,y])   # data는 assert_equal이 false일 때, 메시지로 출력할 값
with tf.control_dependencies([assert_op]):
    c = x+y  # 연산을 하기 전에 control_dependencies를 먼저 연산한다.

with tf.Session() as sess:
    print(sess.run(c,feed_dict={x:3,y:4}))

###############################################
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.Variable(0)
assert_op = tf.assign(z,z+1)   # data는 assert_equal이 false일 때, 메시지로 출력할 값
with tf.control_dependencies([assert_op]):
    c = x+y  # 연산을 하기 전에 control_dependencies를 먼저 연산한다.

with tf.control_dependencies([assert_op]):
    d = 2*x+y  # 연산을 하기 전에 control_dependencies를 먼저 연산한다.


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([c,d,z],feed_dict={x:3,y:4}))  # c,d가 각각 tf.control_dependencies([assert_op] 이다. assert_op를 중복계산하지 않는다.
    print(sess.run([c],feed_dict={x:3,y:4}))  # z=1 --> z=2
    print(sess.run([d],feed_dict={x:3,y:4})) # z=2 --> z=3
    print(sess.run([z])) # z=3




###############################################
# 여기서는 shadow variable만 만들어 본다. 다음 example에서 shadow variable을 이용한 network을 만든다.
#http://ruishu.io/2017/11/22/ema/

ema = tf.train.ExponentialMovingAverage(decay=0.9)


data =np.random.randn(2,3)
target = np.random.randn(2,1)

s = tf.placeholder(tf.float32, [None, 3], 's')
t = tf.placeholder(tf.float32, [None, 1], 't')
with tf.variable_scope('Actor'):
    net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1')
    a = tf.layers.dense(net, 1, activation=tf.nn.tanh, name='a')

loss = tf.losses.mean_squared_error(a,target)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.control_dependencies([train_op]):
    # shadow variables을 생성한다. sess.run(x1) 할 때, shadow variable이 update된다.
    x1 = ema.apply(tf.trainable_variables())  
x2 = ema.average(tf.trainable_variables()[0])  # ema.aveage로 shaow varialble에 접근할 수 있다.

sess= tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tf.global_variables()[:4]))
print('--'*10)
print(sess.run(tf.global_variables()[-4:]))
sess.run(x1, feed_dict={s: data, t: target})
print('=='*10)
print(sess.run(tf.global_variables()[:4]))
print('--'*10)
print(sess.run(tf.global_variables()[-4:]))
print('**'*10)
print(sess.run(x2))

###############################################
###############################################
# shadow variable을 이용하여 network 만들기
def build(s, name,reuse=None, custom_getter=None):
    with tf.variable_scope(name,reuse=reuse, custom_getter=custom_getter):
        a = tf.layers.dense(s, 1, activation=None, name='a')
    return a

### step 1
ema = tf.train.ExponentialMovingAverage(decay=0.9)

data =np.random.randn(2,3)
target = np.random.randn(2,1)

s = tf.placeholder(tf.float32, [None, 3], 's')
t = tf.placeholder(tf.float32, [None, 1], 't')

a = build(s,'Actor')
loss = tf.losses.mean_squared_error(a,target)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.control_dependencies([train_op]):
    # shadow variables을 생성한다. sess.run(x1) 할 때, shadow variable이 update된다.
    x1 = ema.apply(tf.trainable_variables())  
w = ema.average(tf.trainable_variables()[0])  # ema.aveage로 shaow varialble에 접근할 수 있다.
b = ema.average(tf.trainable_variables()[1])
###########
### step 2: 이제 shadow variable을 이용한 newtwork을 마들 수 있다.
def ema_getter(getter, name, *args, **kwargs):
    return ema.average(getter(name, *args, **kwargs))  # shadow variable에 접근

# custom_getter가 같은 이름 'Actor'를 이용해서 shadow variables를 가져온다.
aa = build(s,'Actor', reuse=True, custom_getter=ema_getter)  # 반드시 name은 같아야 하고, reuse=True이어야 한다.



# test step 1
sess= tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tf.global_variables()[:2]))
print('--'*10)
print(sess.run(tf.global_variables()[-2:]))
sess.run(x1, feed_dict={s: data, t: target})
print('=='*10)
print(sess.run(tf.global_variables()[:2]))
print('--'*10)
print(sess.run(tf.global_variables()[-2:]))
print('**'*10)

shadow_weight = sess.run(w)
shadow_bias = sess.run(b)
print(shadow_weight,shadow_bias)

# test step 2
print('직접계산: ', data.dot(shadow_weight)+shadow_bias)
print('shadow network: ', sess.run(aa,feed_dict={s: data}) )

###############################################
# array의 특정 index 추출. gather, gather_nd
x = tf.placeholder(tf.float32,shape=[3,3])
y = tf.placeholder(tf.int32,shape=[None,2])
z = tf.gather_nd(x,y)

with tf.Session() as sess:
    w = sess.run(z,feed_dict={x:[ [1.1, 2.2, 3.3] ,[4.4, 5.5, 6.6], [7.7, 8.8, 9.9] ], y:[[0,1],[1,2]]})

########
X = np.array([0,2,4,6])
a=np.array([1,2,1,2,0])
X[a] # ---> array([2, 4, 2, 4, 0])

Y = tf.convert_to_tensor(X)
b = tf.convert_to_tensor(a,dtype=tf.int32)
c = tf.gather(Y,b)
sess = tf.Session()
sess.run(c) # array([2, 4, 2, 4, 0])

########
x = np.array([[0.49593696, 0.504063  ],
              [0.4912244 , 0.50877565],
              [0.48871803, 0.51128197],
              [0.48469874, 0.5153013 ],
              [0.4801116 , 0.5198884 ]])
a = np.array([0,0,1,1,0]).astype(np.int32)

X = tf.convert_to_tensor(x)
A = tf.convert_to_tensor(a)

w = tf.stack([tf.range(tf.shape(X)[0]),A],axis=-1)  # tf.shape(X)[0] <--- batch_size. 여기서는 5
z = tf.gather_nd(X,w)  # ---> array([0.49593696, 0.4912244 , 0.51128197, 0.5153013 , 0.4801116 ]

# 다른 방법
ww = tf.range(0, tf.shape(X)[0]) * tf.shape(X)[1] + A
# [0,1,2,3,4]  --> [0,2,4,6,8] --> [0,2,5,7,8]
zz = tf.gather(tf.reshape(X, [-1]), ww)  # ---> array([0.49593696, 0.4912244 , 0.51128197, 0.5153013 , 0.4801116 ]

# 다른 방법2
one_hot_A = tf.one_hot(A,tf.shape(X)[1])
zzz = tf.reduce_sum(x*one_hot_A,axis=-1) # ---> array([0.49593696, 0.4912244 , 0.51128197, 0.5153013 , 0.4801116 ]

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

# tf.contrib.layers.layer_norm   example 2
	
a = np.random.randn(2,2,3,4)
X = tf.placeholder(tf.float32,shape=[2,2,3,4])

# begin_norm_axis=1 --> (N,H,W,C)에서 (H,W,C)크기의 tensor L2 norm값이 HxWxC(평균=1)가 된다.
# begin_norm_axis=3 --> (N,H,W,C)에서 마지막 (C)크기의 tesor L2 norm값이 C(평균=1)가 된다.

# begin_params_axis=3 이면, 마지막 차원 즉 C개의 beta,gamma변수가 생성된다.
Y = tf.contrib.layers.layer_norm(X,begin_norm_axis=1,begin_params_axis=3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a,b,c,d = sess.run([X,Y,'LayerNorm/beta:0','LayerNorm/gamma:0'],feed_dict={X:a})
	
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
def my_func(x):
    return 2*x*x-3.0

x = tf.placeholder(tf.float32,shape=None)

y = tf.contrib.eager.py_func(func=my_func, inp=[x], Tout=tf.float32)

x_= [1.0,-1.0]
with tf.Session() as sess:
  y_ = sess.run(y, feed_dict={x: x_})
print(x_, y_)
=====
# numpy 함수, argument 없이
a = np.random.choice(2,size=(2,3),p=[0.5,0.5],replace=True)

A = tf.py_func(func=lambda: np.random.choice(2,size=(2,3),p=[0.5,0.5],replace=True),inp=[],Tout=tf.int64)
========
# 함수로 처리. function return type에 민감함.
drop_rate=0.2

def word_mask(x,drop_rate):
    size = x.get_shape().as_list()
    mask = tf.py_func(func=lambda s: np.random.choice(2,size=s,p=[drop_rate,1-drop_rate],replace=True),inp=[size],Tout=tf.int64)
    return mask*x

input = np.array([[2,3,4],[2,1,0]],dtype=np.int64)
x = tf.convert_to_tensor(input)


masked_x = word_mask(x,drop_rate)
	
###############################################
# cpu에서는 'channels_first'가 작동하지 않는다.
x = np.random.normal(size=[2,48,48]).astype(np.float32)
x = tf.convert_to_tensor(x)
y  = tf.layers.conv1d(x,filters=48,kernel_size=3, strides=1,use_bias=False,padding='valid',data_format='channels_last') # 'channels_first'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_ = sess.run(y)
print(y_)

###############################################
from tensorflow.python.ops.parallel_for.gradients import jacobian
	
# tf.nn.ctc_loss: and the largest value (num_classes - 1) is reserved for the blank label
#{a: 0, b: 1, c: 2, blank: 3}
	
def CTC_Loss():
    # ctc_loss v1에서는 sparse matrix가 들어가기 때문에, gt(label)에 0번 character가 포함되어 있으면, 0번에 대한 loss를 계산못한다.
    # v2에서도 sparse를 넣어주면 같은 결과가 나온다. 
    # 이는 0번을 padding으로 인식하는 문제가 있기 때문이다.
    # 따라서, 0번에는 의미 있는 charcter를 부여하면 안된다.
    # v2에서 label에 sparse가 아닌, dense를 넣어주어야 한다.
    
    
    batch_size=2
    output_T=5
    target_T=3 # target의 길이. Model이 만들어 내는 out_T는 target보다 길다.
    num_class = 4 # 0, 1, 2는 character이고, 마지막 3은 blank이다.
    
    x = np.arange(40).reshape(batch_size,output_T,num_class).astype(np.float32)
    x = np.random.randn(batch_size,output_T,num_class)
    x = np.array([[[ 0.74273746,  0.07847633, -0.89669566,  0.87111101],
            [ 0.35377891,  0.87161664,  0.45004634, -0.01664156],
            [-0.4019564 ,  0.59862392, -0.90470981, -0.16236736],
            [ 0.28194173,  0.82136263,  0.06700599, -0.43223688],
            [ 0.1487472 ,  1.04652007, -0.51399114, -0.4759599 ]],
    
           [[-0.53616811, -2.025543  , -0.06641838, -1.88901458],
            [-0.75484499,  0.24393693, -0.08489008, -1.79244747],
            [ 0.36912486,  0.93965647,  0.42183299,  0.89334628],
            [-0.6257366 , -2.25099419, -0.59857886,  0.35591563],
            [ 0.72191422,  0.37786281,  1.70582983,  0.90937337]]]).astype(np.float32)
    
    xx = tf.convert_to_tensor(x)
    xx = tf.Variable(xx)
    logits = tf.transpose(xx,[1,0,2])
    
    yy = np.random.randint(0,num_class-1,size=(batch_size,target_T))  # low=0, high=3 ==> 0,1,2
    yy = np.array([[1, 2, 2],[1, 0, 1]]).astype(np.int32)
    #yy = np.array([[1, 2, 2,0,0,0],[1,0,2,0,0,0]]).astype(np.int32)  # 끝에 붙은 0은 pad로 간주한다. 중간에 있는 0은 character로 간주
    
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(yy, zero)
    indices = tf.where(where)
    values = tf.gather_nd(yy, indices)
    targets = tf.SparseTensor(indices, values, yy.shape)
    
    
    # preprocess_collapse_repeated=False  ---> label은 반복되는 character가 있을 수 있으니, 당연히 False
    # ctc_merge_repeated=False  ---> 모델이 예측한 반복된 character를 merge하지 않는다. 이것은 ctc loss의 취지와 다르다.
    loss0 = tf.nn.ctc_loss(labels=targets,inputs=logits,sequence_length=[output_T]*batch_size,ctc_merge_repeated=False) 
    # 이 loss0는 의미 없음.
    
    loss1 = tf.nn.ctc_loss(labels=targets,inputs=logits,sequence_length=[output_T]*batch_size)
    loss2 = tf.nn.ctc_loss_v2(labels=yy,logits=logits,label_length =[target_T]*batch_size,
                              logit_length=[output_T]*batch_size,logits_time_major=True,blank_index=num_class-1)
    
    
    # lables에 sparse tensor를 넣으면, v1과 결과가 같다. 
    loss3 = tf.nn.ctc_loss_v2(labels=targets,logits=logits,label_length =[3,3],
                              logit_length=[output_T]*batch_size,logits_time_major=True,blank_index=num_class-1)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    gradient = optimizer.compute_gradients(loss1)
    
    
    prob = tf.nn.softmax(xx,axis=-1)
    # jacobian을 이용해서 logits에 대한 softmax값의 미분을 구한다.
    a = xx[0,1]
    b = tf.nn.softmax(a)
    grad = jacobian(b,a)
    
    
    # logit에 대한 미분을 softmax에 대한 미분으로 변환하기 위해 grad의 inverse를 곱한다.
    # grad의 역행렬이 존재하지 않는다.
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    l0 = sess.run(loss0)
    l1 = sess.run(loss1)
    l2 = sess.run(loss2)
    l3 = sess.run(loss3)
    print('loss: ',l0, l1,l2,l3)
    g = sess.run(gradient[0][0])
    p = sess.run(prob)
    gg = sess.run(grad)

	
###############################################
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if "discriminator" in var.name]


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith('d_lr')]


d_loss_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(self.d_loss,var_list=d_vars)
###############################################
tensorflow이 seed 고정은 graph level에서의 고정이다.
tf.set_random_seed(1)는 tf.Session() 선언 이후에 위치해야 한다.
	
###############################################
batch_size = 2
num_true = 5 # word2vec에서 정답 label은 1개가 보통
labels = tf.convert_to_tensor(np.array([[0,1,2,3,4],[5,6,7,8,9]],dtype=np.int64))  #(batch_size,num_true)
num_sampled =50
num_classes = 50


a = tf.nn.log_uniform_candidate_sampler(true_classes=labels,num_true=num_true,num_sampled=num_sampled,unique=True,range_max=num_classes)
sess = tf.Session()
b = sess.run(a)

print(np.sum(b[2]))
	
###############################################
# C:\Users\Administrator\.keras\datasets  아래에 다운
#data_set = tf.keras.utils.get_file(fname="imdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)

data_set = tf.keras.utils.get_file(fname="D:\hccho\\CommonDataset\\imdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)	

###############################################
# C:\Users\Administrator\.keras\datasets  아래에 다운
#data_set = tf.keras.utils.get_file(fname="imdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)

# 파일은 지정 디렉토리에 받지만, 압축 푼 것은 ....
#The final location of a file example.txt would therefore be ~/.keras/datasets/example.txt 
# return 값은 Path to the downloaded file
data_set = tf.keras.utils.get_file(fname="D:\hccho\\CommonDataset\\imdb.tar.gz", origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)
	
###############################################
saver=tf.train.Saver(var_list, max_to_keep=5,keep_checkpoint_every_n_hours=10000.0)

saver.save(sess,디렉토리 + preface)

tf.train.get_checkpoint_state(디렉토리) <--- checkpoint 파일 단순 파싱, 이것만으로는 부족.

saver.restore(디렉토리 + preface + global_step)
###############################################
# tf.train.latest_checkpoint(save_dir) ---> get_most_recent_checkpoint()과 차이 없음. checkpoint 파일 파싱.

def get_most_recent_checkpoint(checkpoint_dir):  # preface = 'model.ckpt'
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    
    if checkpoint_paths == []: 
        return ''
    
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint	

def get_most_recent_checkpoint2(checkpoint_dir,preface = 'model.ckpt'):  # preface도 입력
    checkpoint_paths = [path for path in glob("{}/{}-*.data-*".format(checkpoint_dir,preface))]
    
    if checkpoint_paths == []: 
        print('No checkpoint')
        return ''
    
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]
    
    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "{}-{}".format(preface,max_idx))
    
    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint
	
	
	
load_path와 checkpoint_path를  직접  define하고,             load_path = './ckpt'               checkpoint_path = './ckpt/model.ckpt'

restore_path = get_most_recent_checkpoint(load_path)
saver.restore(sess,restore_path)
save.save(sess,checkpoint_path,global_step = 100)
	
	
	
	
	
###############################################
tfrecord파일 만들기
tf.train.Example(--> tf.parse_single_example 꺼낸다.)  vs tf.train.SequenceExample(--> tf.parse_single_sequence_example로 꺼낸다) 차이를 잘 모르겠다.

writer = tf.python_io.TFRecordWriter(output_file)
for 
    example = tf.train.Example(features=tf.train.Features(feature={...}))  또는 tf.train.SequenceExample(context=tf.train.Features(...), feature_lists=tf.train.FeatureLists(...))
    writer.write(example.SerializeToString())
wirte.close()
	
###############################################
Tensorflow로 이미지 파일 읽기:   -> inception3 모델의 input
filename = 'D:\\hccho\\PythonTest\\resource.jpg'
with tf.io.gfile.GFile(filename, "rb") as f:     # tf.gfile.GFile
    encoded_image = f.read()  # binary data

image=tf.image.decode_jpeg(encoded_image)  # <tf.Tensor 'Const:0' shape=() dtype=string>  ---> sess.run하면 uint8
image_float = tf.image.convert_image_dtype(image, dtype=tf.float32) # float32 --> 0~1 사이 값으로 변환. dtype에 정수를 줄 수도 있다.

image = tf.image.resize_images(image,size=[resize_height, resize_width],method=tf.image.ResizeMethod.BILINEAR)	
	
# Crop to final dimensions.
if is_training:
    image = tf.random_crop(image, [height, width, 3])
else:
# Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

# Randomly distort the image.
if is_training:
    image = distort_image(image, thread_id)

# Rescale to [-1,1] instead of [0, 1]
image = tf.subtract(image, 0.5)
image = tf.multiply(image, 2.0)

# 참고로 audio는 
from tensorflow.python.ops import io_ops
audio_meta = io_ops.read_file(audio_path)  # sess.run 하면 binary data
wav_decoder = contrib_audio.decode_wav(audio_meta, desired_channels=1,desired_samples = sr)
	
###############################################	
# receptive field = 38
inputs = tf.placeholder(tf.float32,shape=[1,38,38,1])

L1= tf.layers.conv2d(inputs,filters=10,kernel_size=3,padding='valid',strides=1)  # 36
L2= tf.layers.max_pooling2d(L1,pool_size=2,strides=2,padding='valid') # 18
L3= tf.layers.conv2d(L2,filters=10,kernel_size=3,padding='valid',strides=1) # 16
L4= tf.layers.max_pooling2d(L3,pool_size=2,strides=2,padding='valid') # 8

L5= tf.layers.conv2d(L4,filters=10,kernel_size=3,padding='valid',strides=1) # 6
L6= tf.layers.conv2d(L5,filters=10,kernel_size=3,padding='valid',strides=1) # 4
L7= tf.layers.conv2d(L6,filters=10,kernel_size=4,padding='valid',strides=1) # 1
###############################################	
#Regularization Loss

def my_layer(x,training,name=None):
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
        L1 = tf.layers.conv1d(x,filters=10,kernel_size=2,kernel_regularizer=regularizer,name='L1')
        L2 = tf.layers.batch_normalization(L1,training=training)
        L3 = tf.nn.relu(L2)
    return L3

x = tf.convert_to_tensor(np.random.randn(2,10,3).astype(np.float32))
x1 = tf.convert_to_tensor(np.random.randn(2,20,3).astype(np.float32))
y = my_layer(x,True,"a")
y1 = my_layer(x1,False,"b")

#reg_loss = tf.losses.get_regularization_loss()
reg_loss = tf.compat.v1.losses.get_regularization_loss()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(reg_loss))
###############################################	
# placeholder가 아니어도 feeding 할 수 있다.

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,1]]).T
x = tf.placeholder(tf.float32, [None,3])
y = tf.placeholder(tf.float32, [None,1])
L1 = tf.layers.dense(x,units=4, activation = tf.nn.relu,name='L1')
L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize( tf.reduce_mean( 0.5*tf.square(L2-y)))

graph = tf.get_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
z = sess.run(L1,feed_dict={x:X})

print(sess.run(L2,feed_dict={x:X}))
print(sess.run(L2,feed_dict={L1:z}))  # L1 tensor에 feeding
###############################################	
# collection을 이용해서, debugging이 수월해진다.
	
tf.add_to_collection('hccho_debug', [L1,L2])
또는 tf.add_to_collections(['hccho_collection','hccho_collection2'], [L1,L2])	

---> 
tf.get_collection('hccho_debug')	
	
###############################################	
def Model():
    x = tf.placeholder(tf.float32, [None,3])
    y = tf.placeholder(tf.float32, [None,1])
    L1 = tf.layers.dense(x,units=4, activation = tf.sigmoid,name='L1')
    L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
    train = tf.train.AdamOptimizer(learning_rate=1).minimize( tf.reduce_mean( 0.5*tf.square(L2-y)))
    return x,y,L1,L2,train


with tf.variable_scope('model1'):
    A = Model()
with tf.variable_scope('model2'):
    B = Model()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
data = np.random.randn(2,3)


print('Before assing:')
print(sess.run(A[3],feed_dict={A[0]:data}))
print(sess.run(B[3],feed_dict={B[0]:data}))


update_weights = [tf.assign(v_to, v_from) for (v_to, v_from) in 
   zip(tf.trainable_variables('model1'), tf.trainable_variables('model2'))]
        
sess.run(update_weights)
print('After assing:')
print(sess.run(A[3],feed_dict={A[0]:data}))
print(sess.run(B[3],feed_dict={B[0]:data}))

print('numpy array로 직접 assing:')
v1 = tf.get_default_graph().get_tensor_by_name('model1/L1/kernel:0')
sess.run(tf.assign(v1,np.random.randn(3,4)))
print(sess.run(A[3],feed_dict={A[0]:data}))
print(sess.run(B[3],feed_dict={B[0]:data}))
###############################################	
# cross entropy loss: 정확한 one-hot이 아니라, 강화학습에서의 advantage형태
	
x = np.random.randn(3,4)
p = tf.nn.softmax(x)

action = np.array([1,2,0])
reward_ = np.array([1.1,-2,-3]).astype(np.float32)

reward = tf.convert_to_tensor(reward_)
action_onehot = tf.one_hot(action,4)
target = action_onehot*tf.reshape(reward,[3,-1])

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=x)
sess = tf.Session()
print(sess.run(loss))

##

pp = sess.run(p)
loss2 = - np.log(pp[np.arange(3),action])*np.array([1.1,-2,-3])
print(loss2)
###############################################	
x = np.array([[2.1,3.1],[1.0,2.0]])
y = np.array([[2.3,4.5],[1.0,3.2]])

X = tf.convert_to_tensor(x)
Y = tf.convert_to_tensor(y)

z = tf.losses.mean_squared_error(X,Y)

sess = tf.Session()

sess.run(z)   # np.mean(np.square(x-y)), 모든 원소의 제급 평균(axis 상관없이)
###############################################	
x = np.arange(30).reshape(2,3,5).astype(np.float32)

X = tf.convert_to_tensor(x)
Y = tf.reshape(X,(-1,5))


a = tf.nn.moments(X,axes=[0,1])  # 지정된 axes가 없어진다.
b = tf.nn.moments(Y,axes=[0])  # a,b 결과는 동일
###############################################	
allow_soft_placement=True   ---> 가능하면 할당해라.

log_device_placement=True   ---> 무조건 할당해라. with tf.device('/gpu:0')
###############################################	
dist =tf.distributions.Normal(loc=1.2, scale=1.2 )

a =dist.sample(3)  # 원하는 sample 갯수, loc, scale의 shape의 data가 sample 갯수만큼 만들어진다.
p = dist.prob(a)   # scale= 1.0e-35 ==> nan, inf가 나온다.
p2 = dist.log_prob(a)
###############################################	
sess = tf.Session()
with tf.variable_scope('a'):
    a = myNet()

with tf.variable_scope('b'):
    b = myNet()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'a')
saver.restore(sess, 'path')
###############################################	
	
###############################################
	
###############################################

		       
	
###############################################

		       
	
###############################################

		       
	
###############################################

		       
	
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
    
    
    
