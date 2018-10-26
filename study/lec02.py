# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(threshold=np.nan)



mydata = np.genfromtxt('mydata2.txt',delimiter=',',dtype=np.float32)
A = mydata[:,0:2]
B = mydata[:,-1].reshape(-1,1)  # mydata[:,2:3]
plt.subplot(131)
plt.scatter(A[:, 0], A[:, 1], c=B.flatten(),marker=">")
plt.show()
if __name__ == "__main__":    
    pass