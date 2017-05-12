# coding: utf-8


def MNIST_LogisticRegression():
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    import random
    mnist = input_data.read_data_sets("../MachineLearning", one_hot=True)

    nb_classses = 10
    data_feature = 784

    X = tf.placeholder(tf.float32,[None,data_feature])
    Y = tf.placeholder(tf.float32,[None,nb_classses])

    W = tf.Variable(tf.random_normal([data_feature,nb_classses]))
    b = tf.Variable(tf.random_normal([nb_classses]))

    logits = tf.matmul(X,W)+b
    hypothesis = tf.nn.softmax(logits)


    #cost = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


    #################################

    training_epochs = 15
    batch_size = 100


    is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                c,_ =sess.run([cost,optimizer],feed_dict={X:batch_xs, Y:batch_ys})
                avg_cost += c / total_batch

            print("Epoch: ", "%4d" % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost))

        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

        r = random.randint(0,mnist.test.num_examples-1)
        print("Label: ", sess.run(tf.arg_max(mnist.test.labels[r:r+1],1)))
        print("Prediction: ", sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))
        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation='nearest')
        plt.show()

    plt.close()


def MNIST_NN(Xavier=True):
    import tensorflow as tf
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    from tensorflow.examples.tutorials.mnist import input_data

    tf.set_random_seed(777)  # reproducibility

    mnist = input_data.read_data_sets("../MachineLearning", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset

    # parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    # input place holders
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    # weights & bias for nn layers
    if Xavier == True:
        W1 = tf.get_variable("W1",shape=[784,256],initializer=tf.contrib.layers.xavier_initializer())
    else:
        W1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    if Xavier == True:
        W2 = tf.get_variable("W2",shape=[256,256],initializer=tf.contrib.layers.xavier_initializer())
    else:
        W2 = tf.Variable(tf.random_normal([256, 256]))
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    if Xavier == True:
        W3 = tf.get_variable("W3",shape=[256,10],initializer=tf.contrib.layers.xavier_initializer())
    else:
        W3 = tf.Variable(tf.random_normal([256, 10]))
    b3 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L2, W3) + b3


    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    result = sess.run(hypothesis, feed_dict={X: mnist.test.images[r:r + 1]})
    print("Prediction: ", sess.run(tf.arg_max(result,1)) )

    np.set_printoptions(precision=4)
    result = sess.run(tf.nn.softmax(result))
    print (result)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    sess.close()
def MNIST_NN2(layer_size_list,Xavier=True):
    import tensorflow as tf
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    from tensorflow.examples.tutorials.mnist import input_data

    tf.set_random_seed(777)  # reproducibility

    mnist = input_data.read_data_sets("../MachineLearning", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset

    # parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    # input place holders
    
    X = tf.placeholder(tf.float32, [None, layer_size_list[0]])
    Y = tf.placeholder(tf.float32, [None, layer_size_list[-1]])

    last_layer_index = len(layer_size_list)-1
    W={}
    b={}
    L={}
    L[0] = X

    for i in range(1,len(layer_size_list)):
        if Xavier == True:
            W[i] = tf.get_variable("W"+str(i),shape=[layer_size_list[i-1],layer_size_list[i]],initializer=tf.contrib.layers.xavier_initializer())
        else:
            W[i] = tf.Variable(tf.random_normal([layer_size_list[i-1],layer_size_list[i]]))        
        b[i] = tf.Variable(tf.random_normal([layer_size_list[i]]))
        if i == last_layer_index:
            L[i] =tf.matmul(L[i-1], W[i]) + b[i]
        else:
            L[i] = tf.nn.relu(tf.matmul(L[i-1], W[i]) + b[i])    
    

    hypothesis = L[last_layer_index]
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    result = sess.run(hypothesis, feed_dict={X: mnist.test.images[r:r + 1]})
    print("Prediction: ", sess.run(tf.arg_max(result,1)) )

    np.set_printoptions(precision=4)
    result = sess.run(tf.nn.softmax(result))
    print (result)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    
    sess.close()

if __name__ == "__main__":
    #MNIST_LogisticRegression()
    #MNIST_NN(Xavier=True)
    MNIST_NN2(layer_size_list=[784,256,256,10])
