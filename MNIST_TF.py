# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
def MNIST_LogisticRegression():
    # 결론 accuray에 큰 영향을 주는 것이 Adam이다. 그 다음이 Weight 초기화이다. noraml로 초기화 할 때, 표준편차를 0.02정도의 낮은 값이나 xavier.
    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)

    nb_classses = 10
    data_feature = 784
    learning_rate = 0.001

    X = tf.placeholder(tf.float32,[None,data_feature])
    Y = tf.placeholder(tf.float32,[None,nb_classses])

    W = tf.Variable(tf.random_normal([data_feature,nb_classses],mean=0.0,stddev=0.02))
    #W = tf.get_variable("W",shape=[data_feature,nb_classses],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([nb_classses]))

    logits = tf.matmul(X,W)+b
    hypothesis = tf.nn.softmax(logits)


    #cost = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #################################

    training_epochs = 100
    batch_size = 512


    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)  # 55000/batch_size

            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                c,_ =sess.run([cost,optimizer],feed_dict={X:batch_xs, Y:batch_ys})
                avg_cost += c / total_batch
                
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})  # test data 10000개
            print("Epoch: ", "%4d" % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost), 'test acc = {:.4f}'.format(test_acc))

        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

        r = random.randint(0,mnist.test.num_examples-1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))  #shape = (1,10)
        print("Prediction: ", sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))  #shape=(1,784)
        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation='nearest')
        plt.show()

    plt.close()

def MNIST_LogisticRegression_Tensorboard():
    # > tensorboard --logdir=./tensorbaord-logs
    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)

    nb_classses = 10
    data_feature = 784

    X = tf.placeholder(tf.float32,[None,data_feature])
    Y = tf.placeholder(tf.float32,[None,nb_classses])
    logdir = "./tensorbaord-logs"  # 디렉토리는 없으면 만든다.
    
    with tf.variable_scope("FC1") as scope:
        W = tf.Variable(tf.random_normal([data_feature,nb_classses]),name="weight")
        b = tf.Variable(tf.random_normal([nb_classses]),name="bias")

        logits = tf.matmul(X,W)+b
        hypothesis = tf.nn.softmax(logits)


        #cost = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


        
    
        #################################
    
        training_epochs = 15
        batch_size = 100
    
    
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        #tensorboard
        cost_summary = tf.summary.scalar("cost",cost)        
        acc_train_summary = tf.summary.scalar("acc_train",accuracy)
        acc_test_summary = tf.summary.scalar("acc_test",accuracy)
        #merged = tf.summary.merge_all()
        merged = tf.summary.merge([cost_summary,acc_train_summary])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter(logdir, sess.graph)  # tensorboade Writer
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                c,_,summary_str = sess.run([cost,optimizer,merged],feed_dict={X:batch_xs, Y:batch_ys})
                avg_cost += c / total_batch
                writer.add_summary(summary_str, total_batch*epoch+i)
                
            test_acc,summary_str = sess.run([accuracy,acc_test_summary], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            writer.add_summary(summary_str, epoch)
            print("Epoch: ", "%4d" % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost), 'test acc = {:.4f}'.format(test_acc))

        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

        r = random.randint(0,mnist.test.num_examples-1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))  #shape = (1,10)
        print("Prediction: ", sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))  #shape=(1,784)
        plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation='nearest')
        plt.show()

    plt.close()


def MNIST_NN(Xavier=True):
    # 3layer net
    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset

    # parameters
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 512

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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #accuracy check
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'test acc = {:.4f}'.format(test_acc))

    print('Learning Finished!')

    # Test model and check accuracy

    print('Accuracy for test data:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    result = sess.run(hypothesis, feed_dict={X: mnist.test.images[r:r + 1]})
    print("Prediction: ", sess.run(tf.argmax(result,1)) )

    np.set_printoptions(precision=4)
    result = sess.run(tf.nn.softmax(result))
    print (result)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    sess.close()
def MNIST_NN2(layer_size_list,Xavier=True,Dropout=False,KeepProb=0.7):

    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset

    # parameters
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 512

    # input place holders
    
    X = tf.placeholder(tf.float32, [None, layer_size_list[0]])
    Y = tf.placeholder(tf.float32, [None, layer_size_list[-1]])
    keep_prob = tf.placeholder(tf.float32)
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
            if Dropout==True:
                L[i] = tf.nn.dropout(L[i],keep_prob=keep_prob)
    

    hypothesis = L[last_layer_index]
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: KeepProb}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch


        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'test acc = {:.4f}'.format(test_acc))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1.0}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    result = sess.run(hypothesis, feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0})
    print("Prediction: ", sess.run(tf.argmax(result,1)) )

    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    result = sess.run(tf.nn.softmax(result))
    print (result)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    
    sess.close()


def MNIST_NN3(layer_size_list,Xavier=True,KeepProb=0.7):
    # 실험적으로 activation function 없이 dropout만 적용  ==> 잘 안됨(activation 없이 layer를 늘려도 accuracy 올라가지 않음)
    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    # more information about the mnist dataset

    # parameters
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 512

    # input place holders
    
    X = tf.placeholder(tf.float32, [None, layer_size_list[0]])
    Y = tf.placeholder(tf.float32, [None, layer_size_list[-1]])
    keep_prob = tf.placeholder(tf.float32)
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
            L[i] = tf.matmul(L[i-1], W[i]) + b[i]
            L[i] = tf.nn.dropout(L[i],keep_prob=keep_prob)
    

    hypothesis = L[last_layer_index]
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: KeepProb}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch


        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'test acc = {:.4f}'.format(test_acc))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1.0}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    result = sess.run(hypothesis, feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0})
    print("Prediction: ", sess.run(tf.argmax(result,1)) )

    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    result = sess.run(tf.nn.softmax(result))
    print (result)
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    
    sess.close()


def MNIST_CNN():
    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)
    nb_classses = 10
    data_feature = 784
    learning_rate = 0.001

    X = tf.placeholder(tf.float32,[None,data_feature])
    Y = tf.placeholder(tf.float32,[None,nb_classses])
    training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    
    # inforgan discriminator structure
    INIT=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32)
    XX = tf.reshape(X,[-1,28,28,1])
    net1 = tf.layers.conv2d(XX, filters=64,  kernel_size=4, strides=2, padding='SAME',kernel_initializer=INIT,activation=lambda x: tf.nn.leaky_relu(x,0.2)) # =(?, 14, 14, 64)
    
    net2 = tf.layers.conv2d(net1, filters=128,  kernel_size=4, strides=2, padding='SAME',kernel_initializer=INIT,activation=None,use_bias=False) # (?, 7, 7, 128)
    net2 = tf.nn.leaky_relu(tf.layers.batch_normalization(net2, training=training),0.2)

    
    net3 = tf.reshape(net2,[-1,7*7*128])
    net3 = tf.layers.dense(net3,units=1024,use_bias=True,kernel_initializer=INIT)
    net3 = tf.nn.leaky_relu(tf.layers.batch_normalization(net3, training=training),0.2)

    
    
    logits = tf.layers.dense(net3,units=10,kernel_initializer=INIT)
    hypothesis = tf.nn.softmax(logits)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    training_epochs = 15
    batch_size = 512
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)  # 55000/batch_size

            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                c,_ =sess.run([cost,optimizer],feed_dict={X:batch_xs, Y:batch_ys, training: True,keep_prob: 0.6})
                avg_cost += c / total_batch
            
            train_acc = sess.run(accuracy,feed_dict={X:batch_xs, Y:batch_ys, training: False,keep_prob: 1.0})    
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,training: False, keep_prob: 1.0})  # test data 10000개
            print("Epoch: ", "%4d" % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost),'train acc = {:.4f}'.format(train_acc) ,'test acc = {:.4f}'.format(test_acc))

        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels,training: False,keep_prob: 1.0}))
        
def MNIST_CNN2():
    # test acc를 98.85 정도에서 99.3정도로 올리기 위해 모두의 딥러닝 강좌와 동일하게 모델 작성
    mnist = input_data.read_data_sets("../CommonDataset/mnist", one_hot=True)
    nb_classses = 10
    data_feature = 784
    learning_rate = 0.001

    X = tf.placeholder(tf.float32,[None,data_feature])
    Y = tf.placeholder(tf.float32,[None,nb_classses])
    training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    

    INIT= tf.contrib.layers.xavier_initializer()
    
    XX = tf.reshape(X,[-1,28,28,1])
    net1 = tf.layers.conv2d(XX, filters=32,  kernel_size=3, strides=1, padding='SAME',activation= tf.nn.relu,kernel_initializer=INIT) # (?, 28, 28, 32)
    net1 = tf.nn.dropout(tf.nn.max_pool(net1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'),keep_prob=keep_prob) # (?, 14, 14, 32)

    net2 = tf.layers.conv2d(net1, filters=64,  kernel_size=3, strides=1, padding='SAME',activation=tf.nn.relu,kernel_initializer=INIT) # (?, 14, 14, 64)
    net2 = tf.nn.dropout(tf.nn.max_pool(net2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'),keep_prob=keep_prob) # (?, 7, 7, 64)

    net3 = tf.layers.conv2d(net2, filters=128,  kernel_size=3, strides=1, padding='SAME',activation=tf.nn.relu,kernel_initializer=INIT) # (?, 7, 7, 128)
    net3 = tf.nn.dropout(tf.nn.max_pool(net3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'),keep_prob=keep_prob) # (?, 4, 4, 128)
    
    
    net4 = tf.reshape(net3,[-1,4*4*128])
    net4 = tf.layers.dense(net4,units=625,activation=tf.nn.relu,kernel_initializer=INIT)
    net4 = tf.nn.dropout(net4,keep_prob=keep_prob)


    logits = tf.layers.dense(net4,units=10,kernel_initializer=INIT)
    hypothesis = tf.nn.softmax(logits)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    training_epochs = 15
    batch_size = 512
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)  # 55000/batch_size

            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                c,_ =sess.run([cost,optimizer],feed_dict={X:batch_xs, Y:batch_ys, training: True,keep_prob: 0.7})
                avg_cost += c / total_batch
            
            train_acc = sess.run(accuracy,feed_dict={X:batch_xs, Y:batch_ys, training: False,keep_prob: 1.0})    
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,training: False, keep_prob: 1.0})  # test data 10000개
            print("Epoch: ", "%4d" % (epoch+1), 'cost = ','{:.9f}'.format(avg_cost),'train acc = {:.4f}'.format(train_acc) ,'test acc = {:.4f}'.format(test_acc))

        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels,training: False,keep_prob: 1.0}))
     
        
if __name__ == "__main__":
    #MNIST_LogisticRegression()
    #MNIST_LogisticRegression_Tensorboard()
    #MNIST_NN(Xavier=True)  # 3 layer net
    #MNIST_NN2(layer_size_list=[784,256,256,10],Xavier=True)
    #MNIST_NN2(layer_size_list=[784,512,512,512,512,10],Xavier=True,Dropout=True,KeepProb=0.7)
    #MNIST_NN3(layer_size_list=[784,512,512,512,512,10],Xavier=True,KeepProb=0.97)
    #MNIST_CNN()
    MNIST_CNN2()