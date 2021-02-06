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
####################################################
# 2개 같이 그리기
plt.plot([0,1,2,3,4], [2,3,1,4,5], label='A',marker='o')
plt.plot([1,2,3], [4,4,1], label='B', linestyle='--',marker='P')
plt.legend()
plt.show()
plt.close()
########
p1, = plt.plot([0,1,2,3,4], [2,3,1,4,5], label='A',marker='o')
p2, = plt.plot([1,2,3], [4,4,1], label='B', linestyle='--',marker='P')

plt.legend(handles=[p1, p2],loc='upper center')  # loc='upper right'

####################################################

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
x = np.random.normal(size=500)
n, bins, patches = plt.hist(x.numpy(),50, density=False, facecolor='g', alpha=0.75)  # density(y축을 빈도 또는 비율), patches는 각각의 사각형 o
plt.setp(patches[20], 'facecolor', 'r')


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
# 계산 중 graph ---> spyder, jupyte notebook에서는 안됨.

import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')
 
for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17) # 반드시 있어야 됨.
    time.sleep(0.1)  # sleep은 없어도 됨





##########################################################
# 계산 중 graph  2개 배치---> spyder, jupyte notebook에서는 안됨.
import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()

fig = plt.figure()


axes1 = fig.add_subplot(121)
axes1.set_xlim(0, 100)
axes1.set_ylim(-50, +50)
line1, = axes1.plot(xdata, ydata, 'r-')
axes1.set_ylabel('volts')
axes1.set_title('a sine wave')
axes1.set_xlabel('time (s)')

axes2 = fig.add_subplot(122)
axes2.set_xlim(0, 100)
axes2.set_ylim(-50, +50)
line2, = axes2.plot(xdata, ydata, 'b-')
axes2.set_ylabel('test')
axes2.set_title('reward')


for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line1.set_xdata(xdata)
    line1.set_ydata(ydata)
    
    if i %5 ==0:
        line2.set_xdata(xdata[::5])
        line2.set_ydata(ydata[::5])       
    
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
 

plt.show()   # add this if you don't want the window to disappear at the end  ---> 없으면 자동으로 닫힘.


======================
# 계산 중 graph  --- jupyter notebook OK. spyder에서는 안됨
import matplotlib.pyplot as plt
import time
import random
from IPython.display import clear_output
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()

for i in range(100):
    clear_output(True)
    plt.figure(figsize=(30, 5))
    xdata.append(i)
    ydata.append(ysample[i])

    plt.plot(xdata,ydata)
    plt.pause(1e-17)
    plt.show()
    time.sleep(0.1)

======================
# 계산 중 graph 2개  --- jupyter notebook OK. spyder에서는 안됨
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from IPython.display import clear_output
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata1 = []
ydata2 = []
 
plt.show()
 
for i in range(100):
    clear_output(True)
    plt.figure(figsize=(30, 5))
    xdata.append(i)
    ydata1.append(ysample[i])
    ydata2.append(ysample[-i])
    
    
    plt.subplot(121)
    plt.plot(xdata,ydata1)
    
    plt.subplot(122)
    plt.plot(xdata,ydata2)    
    plt.pause(1e-17)
    plt.show()
    time.sleep(0.1)

##########################################################
"""
==================
Animated line plot
==================

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    return line,


def animate(i):
    line.set_ydata(np.sin(x + i / 100))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()


##########################################################
# grid, 2개의 data 각각의 점들을 선으로 연결
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    
    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    #grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    #grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # 양 끝점 포함.
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples
    
    ax.plot(samples[:, 0], samples[:, 1], 'o',markersize=3)  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's',markersize=3)  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange',linewidth=1))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])



low = [-1.0, -5.0]
high = [1.0, 5.0]
bins= [11,11]  # 양 끝점 포함해서 


grid = np.array([np.linspace(low[dim], high[dim], bins[dim])[1:-1] for dim in range(len(bins))])  # 양 끝점 제외



print(grid)

for l, h, b, splits in zip(low, high, bins, grid):
    print("    [{}, {}] / {} => {}".format(l, h, b, splits))


# data가 grid를 벗어나면 안된다.  ---> 벗어나면 error!!!!!
data = np.array([ np.clip(0.3*np.random.randn(4),low[0],high[0]), np.clip(2*np.random.randn(4),low[1],high[1])]).T

# np.digitize: np.digitize( data, gird) ---> data에는 여러개가 들어가도 됨.   
# data[i]가 grid의 어떤 index에 대응되는지 찾아준다. 정확히는 data[i]를 초과하는 grid값의 index
# grid의 index가 0..n-1(size n)이면 return 되는 index는 0~n까지이다. grid index 범위를 넘어간다.

discretized_data =  np.array([ [ np.digitize(s, g) for s, g in zip(d,grid) ]  for d in data])


####

print('data: ', data)
print('discretized_data: ', discretized_data)

visualize_samples(data, discretized_data, grid, low, high)
plt.show()



##########################################################
# heat map
data = np.random.randn(10,10)

plt.imshow(data, cmap='hot', interpolation='none')  # interpolation='nearest'
plt.show()


##########################################################
# 한글 title 사용
font_name = fm.FontProperties(fname=r'C:\Users\BRAIN\AppData\Local\Microsoft\Windows\Fonts\NanumGothic-Regular.ttf').get_name()
matplotlib.rc('font', family=font_name)
#fm._rebuild()   # cache 때문에 rebuild가 한번 필요함.


plt.rcParams["font.family"] = 'NanumGothic'   #'NanumBarunGothic', 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False  # 한글 설정후, '-' 마이너스 부호 깨짐 해결.  http://taewan.kim/post/matplotlib_hangul/

##########################################################
# 마진없이 저장.
def save_as_image():
    audio_filename = r"D:\SpeechRecognition\AudioSignalProcessingForML-musikalkemist\audio_resources\scale.wav"
    y, sr = librosa.load(audio_filename,duration=10) 
    print(y.shape,sr)     
    
    melspectrogram = librosa.feature.melspectrogram(y,n_mels=90)
    print(melspectrogram.shape)   # (90, 342)
    librosa.display.specshow(librosa.power_to_db(melspectrogram,ref=np.max),x_axis='time',y_axis='mel')
    plt.axis('off')  # 이게 있으면 좀 더 큰 사이즈로 저장.
    plt.savefig('mel_spec.png',bbox_inches='tight', pad_inches=0)  # (w,h) = (496 x 369) 크기로 저장


##########################################################
# scatter plot 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

num_class = 3
df=pd.DataFrame(np.random.rand(30,3))
grp=pd.DataFrame(np.random.randint(num_class,size=30))
df['grp']=grp

colors = ['salmon', 'orange', 'steelblue']
fig = plt.figure(figsize=(5, 4))
ax = Axes3D(fig)

for grp_name, grp_idx in df.groupby('grp').groups.items():
    y = df.iloc[grp_idx,1]
    x = df.iloc[grp_idx,0]
    z = df.iloc[grp_idx,2]
    ax.scatter(x,y,z, label=grp_name)
    #ax.scatter(x,y,z, label=grp_name,color=colors[grp_name])  # this way you can control color/marker/size of each group freely

ax.legend()

plt.show()

##########################################################
# word2vec tsne로 그릴때....
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))  # in inches

low_dim_embs = np.random.randn(5,2)
labels = ['a','h','Z','dd','pq']
for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,xy=(x, y),
        xytext=(5, 2),  # 점에서 text가 위치할 상대적 위치
        textcoords='offset points',fontsize=10)



plt.annotate('Median', xy=(low_dim_embs[0,0],low_dim_embs[0,1]), xytext=(0,-100),textcoords='offset points',
            fontsize=10, ha='center',
            arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=10))
##########################################################
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(7, 7))  # subplt 각각의 fig, axes
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    ax.set_title(commands[label_id.numpy()])   # label_id = 6 ---> commands[6] ---> 'up'
    ax.axis('off')

plt.show()

##########################################################
data = np.array([[-10,-2,3,4],[-2,6,7,-10]])

plt.subplot(2,1,1)
plt.imshow(data)
plt.colorbar()

plt.subplot(2,1,2)


x=[2,3,4]
y=[10,15,17,30,50]
plt.pcolormesh(y,x,data)  # y: 가로축(axis=1), x: 세로축(axis=0)
plt.colorbar()

##########################################################
N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
#y_err = x.std() * np.sqrt(1/len(x) +(x - x.mean())**2 / np.sum((x - x.mean())**2))
y_err = (y_est -y).std()

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.1)
ax.plot(x, y, 'o', color='tab:brown')

plt.show()


##########################################################



##########################################################



##########################################################



##########################################################



##########################################################



##########################################################



##########################################################





