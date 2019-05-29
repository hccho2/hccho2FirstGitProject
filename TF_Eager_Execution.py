# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
tf.reset_default_graph()
tf.enable_eager_execution()

def ex1():
    # Data
    x_data = [1, 2, 3, 4, 5]
    y_data = [1, 2, 3, 4, 5]
    
    # W, b initialize
    W = tf.Variable(2.9)
    b = tf.Variable(0.5)
    learning_rate = 0.01
    
    
    # W, b update
    for i in range(100):
        # Gradient descent
        with tf.GradientTape() as tape:
            hypothesis = W * x_data + b
            cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)
        if i % 10 == 0:
          print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
    
    print()
    
    # predict
    print(W * 5 + b)
    print(W * 2.5 + b)

def ex2():
    x_data = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]).astype(np.float32)
    y_data = np.array([[0],[1],[1],[0]]).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))
    
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    for step in range(1000):
        for features, labels  in tfe.Iterator(dataset):
            
            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss = tf.losses.sigmoid_cross_entropy(labels,logits)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        if (step+1) % 100 ==0:    
            print("step: {}, loss: {}".format(step+1, loss))
            
            
            
            
            
if __name__ == "__main__":    
    #ex1()
    ex2()
