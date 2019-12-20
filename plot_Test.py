# coding: utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Simple data to display in various forms  <---- 원하는 함수의 그래프를 그래볼 수 있다.
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x ** 2)
y2= np.cos(x ** 2)
plt.plot(x,y1)

#####################################################
# scatter
x_train = [1,2,3]
y_train = [1,2,3]

plt.scatter(x_train,y_train)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
x1,x2,y1,y2 = plt.axis()
plt.axis((0,5,0,5))

#####################################################
x_train = [1,2,3]
y_train = [1,2,3]
plt.plot(x,y)
plt.title('mu-law')
plt.show()
#####################################################
#scatter
AA = np.genfromtxt('mydata.txt',delimiter=',')
A = AA[:,0:2]
B = AA[:,-1].reshape(-1,1)  # AA[:,2:3]

plt.scatter(A[:, 0], A[:, 1], c=B,marker=">")


#scatter with values
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
x = [1,0,0,0,1]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y,c=x)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))

#plot with values
x=[1,2,3]
y=[9,8,7]

plt.plot(x,y)
for a,b in zip(x, y): 
    plt.text(a, b, str(b))
plt.show() 
  

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
####################################################

import numpy as np
import matplotlib.pyplot as plt
A = np.arange(20).reshape(4,-1)
plt.plot(np.arange(4),A) # A: (n=4,m) --> column방향의 data에 대해서 graph를 m개 그린다.
plt.plot(np.arange(4),A[:,:2])  #  graph를 2개 그린다
####################################################


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




##########################################################
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
XX, YY = np.meshgrid(x, y)
ZZ = f(XX, YY)

plt.title("Contour plots")
plt.contourf(XX, YY, ZZ, alpha=.75, cmap='jet')   # 색으로 경계구분
plt.contour(XX, YY, ZZ, colors='black')  # 경계선만 그림
plt.show()




##########################################################
# 각각의 x축 값으로 그리기
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x ** 2)
y2= np.cos(x ** 2)
y3 = np.sin(x[::2]**2)
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--')
plt.plot(x[::2], y3, label='sin**', linestyle='-')


plt.legend(loc='upper left')
plt.show()
plt.close()





##########################################################
# box plot
a = np.random.randn(50)*4
b = np.random.rand(200)*7
c = np.random.rand(100)*6
plt.figure(figsize=(12, 5))
plt.boxplot([a, b, c],labels=['aa', 'bb', 'cc'], showmeans=True)
plt.show()
plt.hist(a, bins=15, range=[-10,10], color='g', label='aa') # range 범위를 15개 구간으로
plt.show()


##########################################################

# 1 by 4로 이미지 배치
plt.subplot(1,4,1)
img1 = xxxdfdfd
plt.imshow(img1)
plt.title('GT')



plt.subplot(1,4,2)
img2 = ...
plt.imshow(img2)
plt.title('proposal_boxes')




##########################################################






##########################################################



##########################################################



##########################################################



##########################################################



##########################################################



##########################################################



##########################################################
