# coding: utf-8

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from MNIST_test import *

def plot_decision_boundary(X,y,pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)



np.random.seed(0)
N_Data_Feature = 2
N_Train_Data = 200
N_Test_Data = 50

N_Output_Dim = 2
X, y = sklearn.datasets.make_moons(N_Train_Data+N_Test_Data, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()
Y = change_ont_hot_label(y,N_Output_Dim)

x_train = X[:N_Train_Data]
y_train = Y[:N_Train_Data]
x_test = X[N_Train_Data:]
y_test = Y[N_Train_Data:]


iters_num = 10000  # 반복 횟수를 적절히 설정한다.
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []


hidden_layer_depth=[10,3,3]
n_hidden_layer = len(hidden_layer_depth)
net = MultiLayerNet(input_size=N_Data_Feature, hidden_size_list=hidden_layer_depth, output_size=N_Output_Dim, activation='relu',
                    weight_init_std='relu', weight_decay_lambda=0)

weight_list = []
for i in range(1, n_hidden_layer + 2):
    weight_list.append('W' + str(i))
    weight_list.append('b' + str(i))

for i in range(iters_num):

    grad = net.gradient(x_train, y_train)

    for key in weight_list:
        net.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = net.loss(x_train, y_train)
    train_loss_list.append(loss)

    if i % 1000 == 0:
        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

plot_decision_boundary(x_train,y[:N_Train_Data],lambda x: np.argmax(net.predict_from_learning(x),axis=1))
plt.title("Logistic Regression(train data")
plt.show()
plot_decision_boundary(x_test,y[N_Train_Data:],lambda x: np.argmax(net.predict_from_learning(x),axis=1))
plt.title("Logistic Regression(test data")
plt.show()

