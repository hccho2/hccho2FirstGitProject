# coding: utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x ** 2)
y2= np.cos(x ** 2)



# scatter
x_train = [1,2,3]
y_train = [1,2,3]

plt.scatter(x_train,y_train)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,5,0,5))

#scatter
AA = np.genfromtxt('mydata.txt',delimiter=',')
A = AA[:,0:2]
B = AA[:,-1].reshape(-1,1)  # AA[:,2:3]

plt.scatter(A[:, 0], A[:, 1], c=B,marker=">")

# 하나씩 그리기
plt.plot(x, y1, label='sin')
plt.show()
plt.close()
plt.plot(x, y2, label='cos', linestyle='--')
plt.show()
plt.close()


# 2개 같이 그리기
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--')
plt.show()
plt.close()

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y1)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y2)
plt.show()
plt.close()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y1)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y2)
plt.show()
plt.close()


# Two subplots, unpack the axes array immediately
f, axes = plt.subplots(1, 2)
axes[0].plot(x, y1)
axes[0].set_title('Sharing Yy axis')
axes[1].scatter(x, y2)
axes[1].axis('off')
plt.show()
plt.close()


# Two subplots, unpack the axes array immediately
img = Image.open("Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\MNIST215.png")
plt.figure()
plt.title('my random fig')
plt.imshow(img)
plt.show()
plt.close()


img2 = Image.open("Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\40.png")
plt.figure()
plt.title('HaHaHa')
plt.imshow(img2)
plt.show()
plt.close()





#f, ax = plt.subplots()
#ax.plot(x, y)
#ax.set_title('Simple plot')
