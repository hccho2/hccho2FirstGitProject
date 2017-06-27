import numpy as np


mydata = np.genfromtxt('zoo.txt',delimiter=',',dtype=np.float32)

A = mydata[:, 0:-1]
B = mydata[:, [-1]]   # equivalent to mydata[:, -1].reshape(-1,1)
