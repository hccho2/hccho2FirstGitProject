
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

if __name__ == "__main__":   
    test1()
    
    #test2()
    #test3()
    #testPlaceholde()
    #testLinearRegression()
    #testLinearRegression2()
    
    
    
