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

# 2개 같이 그리기
bb, = plt.plot(y1,label='raw')
bb_, = plt.plot(y2,label='prediction')
plt.legend(handles=[bb, bb_])



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




##########################################################
##########################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
"""
3  <-- dimX
5  <-- dimY
1 2 3  <--X axis
1 3 5 7 9  <--Y axis
1.2 3 4  <--- data  dimY x dimX
5 6 33 
1 2 3 
4 2 3.4
5.5 6 6

"""

def ReadFromFile(filename): 

    file = open(filename)
    line = file.readline()
    dimX = int(line)

    line = file.readline()
    dimY = int(line)  

    line = file.readline()
    X = np.fromstring(line,sep=' ',dtype=np.float)

    line = file.readline()
    Y = np.fromstring(line,sep=' ',dtype=np.float)

    X,Y = np.meshgrid(X,Y)

    

    lines = file.readlines()
    lines = ' '.join(lines)
    lines = lines.replace('\n','')
    data  = np.fromstring(lines,sep=' ',dtype=np.float).reshape(dimY,dimX)

    file.close()
    return X,Y,data

 

X1,Y1,data1 = ReadFromFile("mydata3.txt")
X2,Y2,data2 = ReadFromFile("mydata3.txt")

 
fig = plt.figure()

#ax = fig.gca(projection='3d')   # 1개만 그릴 때

ax1 = fig.add_subplot(131,projection='3d')
ax2 = fig.add_subplot(132,projection='3d')
ax3 = fig.add_subplot(133,projection='3d')
surf = ax1.plot_surface(X1,Y1,data1)
surf = ax2.plot_surface(X2,Y2,data2)
surf = ax3.contour(X1,Y1,data1,20)
surf = ax3.contour(X2,Y2,data2)

plt.show()

