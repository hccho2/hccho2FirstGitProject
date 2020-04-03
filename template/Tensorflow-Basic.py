
#  coding: utf-8
import tensorflow as tf
import numpy as np

"""

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,1]]).T
x = tf.placeholder(tf.float32, [None,3])
y = tf.placeholder(tf.float32, [None,1])

L1 = tf.layers.dense(x,units=4, activation = tf.sigmoid,name='L1')
L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
train = tf.train.AdamOptimizer(learning_rate=1).minimize( tf.reduce_mean( 0.5*tf.square(L2-y)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(60000):
        sess.run(train, feed_dict={x: X, y: Y})
"""

x = tf.placeholder(tf.float32, [None,284])
with tf.variable_scope("hccho"):
    fc1 = tf.layers.dense(x,units=1024, activation = tf.nn.relu,name='fc1')
    fc2 = tf.layers.dense(fc1,units=10, activation = None,name='fc2')    
    out = tf.nn.softmax(fc2)    

image = np.random.randn(5,284)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_ = sess.run(out,feed_dict={x: image})
    

var_list = tf.trainable_variables()   
var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(var_list)
print(var_list2)    

====================================================
def simple_net():
    tf.reset_default_graph()
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
    Y = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T
    
    W = tf.get_variable('weight', dtype=tf.float32,shape=[3,1], initializer=tf.initializers.constant(1))
    b = tf.get_variable('bias',dtype=tf.float32,shape=[1],initializer=tf.initializers.zeros())
    Z = tf.matmul(X,W)+b
    
    loss = tf.nn.l2_loss(Z-Y)
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        
        for step in range(100):
            _, loss_ = sess.run([opt,loss])
            print('{}:  loss = {}'.format(step,loss_))

====================================================

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
A = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])

# Feature Scaling을 하면, learning rate을   1.0e-5 ==> 1.0e-2로 조정해야 함
#A = (A-np.mean(A,0))/np.std(A,0)

B = np.array([[152.],[185.],[180.],[196.],[142.]])


def MultivariateRegressionTF():
    tf.reset_default_graph()

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W1 = tf.Variable(tf.random_normal([3, 5]), name='W1')
    b1 = tf.Variable(tf.random_normal([5]), name='b1')
    W2 = tf.Variable(tf.random_normal([5, 1]), name='W2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')    
    # Hypothesis
    L1 = tf.nn.relu( tf.matmul(X, W1) + b1)
    logit = tf.matmul(L1, W2) + b2
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(logit - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, hy_val, _ = sess.run(
            [cost, logit, train], feed_dict={X: A, Y: B})
        if step % 5000 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    print("W1, b1, W2, b2 ", sess.run([W1,b1,W2,b2]))

def MultivariateRegressionTF2():
    import matplotlib.pyplot as plt
    tf.reset_default_graph()
    
    mydata = np.genfromtxt('mydata2.txt',delimiter=',',dtype=np.float32)
    A = mydata[:,0:2]
    B = mydata[:,-1].reshape(-1,1)  # mydata[:,2:3]
    plt.subplot(131)
    plt.scatter(A[:, 0], A[:, 1], c=B.flatten(),marker=">")
    

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    L1 = tf.layers.dense(X,units=5, activation = tf.nn.relu,name='L1')
    logit = tf.layers.dense(L1,units=1, activation = None,name='L2')
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(logit - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, hy_val, _ = sess.run(
            [cost, logit, train], feed_dict={X: A, Y: B})
        if step % 5000 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    var_list = tf.trainable_variables()
    print(var_list)
    print(sess.run(var_list))
    graph = tf.get_default_graph()
    print(sess.run(graph.get_tensor_by_name(name='L1/kernel:0')))

def LogisticRegressionTF4():    
    # veiwing gradient values within tensorflow
    tf.reset_default_graph()
    
    #AA2 = AddFeatures(AA[:,1:],3)
    AA2 = A
    
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, AA2.shape[1]])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    
    mode = 1
    
    if mode == 0:
        logits = tf.layers.dense(X,units=1, activation = None,name='L1')
    else:
        L1 = tf.layers.dense(X,units=5, activation = tf.nn.relu,name='L1')
        logits = tf.layers.dense(L1,units=1, activation = None,name='L2')
    
    

    
    # Simplified cost/loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y ))
    
    # Minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    
    predicted = tf.cast(logits >= 0, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    
    total_cost = []
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        for step in range(10000):
            _, l, acc,p = sess.run([train,loss,accuracy,predicted], feed_dict={X: AA2, Y: B})
            if step % 1000 == 0:
                print(step, l)
                total_cost.append(l)
        # Accuracy report
        print("\nAccuracy: ", acc, "\nloss", l)    
    plt.subplot(132)
    plt.scatter(A[:, 0], A[:, 1], c=p.flatten(),marker=">")    
    plt.subplot(133)
    plt.plot(total_cost)
    sess.close()  
    
    
def sin_fitting():
    # x(1 dim) --> y(1 dim) 예측
    X = np.linspace(0, 15, 301)
    Y =  np.sin(2*X - 0.1)+ np.random.normal(size=len(X), scale=0.2)
    X = X.reshape(-1,1)
    Y= Y.reshape(-1,1)
    p1, =plt.plot(X,Y, label='origin')
    
    x = tf.placeholder(tf.float32, [None,1])
    y = tf.placeholder(tf.float32, [None,1])
    
    L1 = tf.layers.dense(x,units=10, activation = tf.nn.sigmoid,name='L1')
    L2 = tf.layers.dense(L1,units=10, activation = tf.nn.sigmoid,name='L2')
    L3 = tf.layers.dense(L2,units=1, activation = None,name='L3')
    loss = tf.reduce_mean( 0.5*tf.square(L3-y))
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(30000):
            sess.run(train, feed_dict={x: X, y: Y})
            if j%4000 ==0:
                loss_ = sess.run(loss, feed_dict={x: X, y: Y})
                print("{}: loss = {}".format(j,loss_ ))
    
        Y_ = sess.run(L3,feed_dict={x:X})
        p2, =plt.plot(X,Y_,label='prediction')
        plt.legend(handles=[p1,p2])

def sin_fitting_rnn():
    # 결과가 좋지 않다. sine 값은 위로 올라가는 0과 아래오 내려가는 0이 섞여 있는데, 다음 값을 예측하는 것이 되지 않는다.
    # 결과가 좋아지려면, data을 생성할 때, noise를 작게 주어야 한다. scale = 0.02 또는 0.01. 
    # 그리고, 아래에서 출발점을 0.0으로 주면, 구분이 되지 않는다. 출발점을 1.0로 주면 sine curve가 예측된다.
    # tf.contrib.seq2seq.InferenceHelper 의 사용법
    from tensorflow.python.layers.core import Dense
    XX = np.linspace(0, 15, 301)   # data를 생성하는데만 사용
    YY =  np.sin(2*XX - 0.1)+ np.random.normal(size=len(XX), scale=0.2) # train에 사용
    XX = XX.reshape(-1,1)
    YY= YY.reshape(-1,1)
    #plt.plot(XX,YY)
    
    batch_size = 5
    seq_length = 10
    output_dim =1
    
    hidden_dim=50
    
    X = tf.placeholder(tf.float32,shape=[None,None,1])
    Y = tf.placeholder(tf.float32,shape=[None,None,1])
    
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
    
    helper = tf.contrib.seq2seq.TrainingHelper(X, np.array([seq_length]*batch_size))
    
    output_layer = Dense(output_dim, name='output_projection')
    
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)
    outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=seq_length)
    
    loss =   tf.reduce_mean(tf.square(outputs.rnn_output - Y))
     
    opt = tf.train.AdamOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        x_batch = []
        for j in range(batch_size):
            idx = np.random.randint(0,len(XX)-seq_length-1)
        
            x_batch.append(YY[idx:idx+seq_length+1])
        x_batch=np.array(x_batch)
        loss_,_ =sess.run([loss,opt],feed_dict={X: x_batch[:,:seq_length,:],Y:x_batch[:,1:,:]})
        
        if i%100==0:
            print("{} loss: = {}".format(i,loss_))    
    
    ##################################################
    batch_size_test =1
    def _sample_fn(decoder_outputs):
        return decoder_outputs
    def _end_fn(sample_ids):
        # infinite
        return tf.tile([False], [batch_size_test])
    helper = tf.contrib.seq2seq.InferenceHelper(
        sample_fn=_sample_fn,
        sample_shape=[1],
        sample_dtype=tf.float32,
        start_inputs=[[0.0]],
        end_fn=_end_fn,
    )
    initial_state = cell.zero_state(batch_size_test, tf.float32)
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)
    outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=200)
    
    
    test_output =  sess.run(outputs.rnn_output)
    test_output = np.squeeze(test_output,axis=0)
    print(test_output.shape)
    
    plt.plot(test_output)    
        
    # 출발점을 잘 설정해 보자.  ==> 이렇게 해도 안됨. 원래 예측한 sequence와 동일.
    # 초기 입력값을 seq_length만큼 주는 것라면, 처음부터 n to 1 모델로 갔어야지....
    seq_length = 5
    initial_state = cell.zero_state(batch_size_test, tf.float32)

    batch_size = 1
    helper = tf.contrib.seq2seq.TrainingHelper(X, np.array([seq_length]*batch_size))
    decoder3 = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)
    outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder3,output_time_major=False,impute_finished=True,maximum_iterations=seq_length)

    idx = np.random.randint(0,len(XX)-seq_length-1)
    input_data = [YY[idx:idx+seq_length]]
    h0 = sess.run(last_state, feed_dict={X: input_data})



    helper = tf.contrib.seq2seq.InferenceHelper(
        sample_fn=_sample_fn,
        sample_shape=[1],
        sample_dtype=tf.float32,
        start_inputs=[list(YY[idx+seq_length])], # input_data의 다음 값
        end_fn=_end_fn,
    )

    initial_state = h0
    decoder4 = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)
    outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder4,output_time_major=False,impute_finished=True,maximum_iterations=200)

    test_output2 =  sess.run(outputs.rnn_output)
    test_output2 = np.squeeze(test_output2,axis=0)
    print(test_output.shape)

    plt.plot(test_output2) 
        
if __name__ == "__main__":
    MultivariateRegressionTF()
    MultivariateRegressionTF2()





