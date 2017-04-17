# coding: utf-8

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from ConvolutionNet_test import *



N_Data = 200
N_Output_Dim = 2
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

Y = change_ont_hot_label(y,N_Output_Dim)

