import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

A = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
B = np.array([[152.],[185.],[180.],[196.],[142.]])

def MultivariateRegression():
    learning_rate=1e-5
    N_Data = A.shape[0]
    
    W = np.random.standard_normal(size=(3,1))
    b = np.random.standard_normal(size=1)
    
    for step in range(2001):
        temp = (np.dot(A,W)+b -B)/N_Data
        W -= learning_rate * np.dot(A.T, temp)
        b -= learning_rate * np.sum(temp,axis=0)
        
    print("W,b: ", W, b) 
    print("Cost: ", np.mean((np.dot(A,W)+b -B)**2))
    print("Prediction:", np.dot(A,W)+b)

def NormalEquation():
    AA=np.insert(A,0,np.ones(A.shape[0]),axis=1)
    W = np.dot(np.linalg.inv(np.dot(AA.T,AA)),np.dot(AA.T,B))
    print("W: ", W) 
    print("Cost: ", np.mean((np.dot(AA,W)- B)**2))
    print("Prediction:", np.dot(AA,W))
def MultivariateRegressionTF():


    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    # Hypothesis
    hypothesis = tf.matmul(X, W) + b
    
    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
    
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: A, Y: B})
        if step % 500 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


    print("W, b ", sess.run([W,b]))

if __name__ == "__main__":
    #MultivariateRegression()
    MultivariateRegressionTF()
    #NormalEquation()