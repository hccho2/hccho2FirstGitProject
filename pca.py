# coding: utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.fftpack import fft
import librosa
from tensorflow.python.layers.core import Dense
theta = np.arctan(1.5)   # np.pi/4
N=5000

X=np.array([np.random.normal(0,4,N),np.random.normal(0,1,N)])
X = np.matmul([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],X).T


np.mean(X,axis=0)
plt.scatter(X[:,0],X[:,1],s=1)
plt.axis([-10, 10, -10, 10])
plt.show()




C = (1/N)*np.matmul(X.T,X)

p,v = np.linalg.eig(C)  # column vector가 eigenvector

q=np.argsort(-p)
v = v[:,q]


Y=np.matmul(v.T,X.T).T

plt.scatter(Y[:,0],Y[:,1],s=1)
plt.axis([-10, 10, -10, 10])
plt.show()


