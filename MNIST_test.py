# coding: utf-8
import urllib.request
import gzip
import numpy as np
import os.path
import pickle
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import time
import datetime
from collections import OrderedDict
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


class TwoLayerNet:  #sub_Layer를 없이 직접 구성

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    def predict_from_learning(self, x):
        
        return self.predict(x)

        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads



class TwoLayerNet2:   #sub_Layer를 활용하여 구성

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.sqrt(2/input_size) * np.random.randn(input_size, hidden_size)
        #self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt( 1/hidden_size ) * np.random.randn(hidden_size, output_size)
        #self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def predict_from_learning(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return softmax(x)
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict_from_learning(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

class MultiLayerNet:
    """완전연결 다층 신경망

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층 생성
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """가중치 초기화
        
        Parameters
        ----------
        weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
            'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
            'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU를 사용할 때의 권장 초깃값
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoid를 사용할 때의 권장 초깃값
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def predict_from_learning(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return softmax(x)


    def loss(self, x, t):
        """손실 함수를 구한다.
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블 
        
        Returns
        -------
        손실 함수의 값
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """기울기를 구한다(수치 미분).
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기를 담은 딕셔너리(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기를 담은 딕셔너리(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx    
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
        
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
        
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def download_mnist():
    for v in key_file.values():
       _download(v)


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def change_ont_hot_label(X,n_dim=10):
    T = np.zeros((X.size, n_dim))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T 

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    
    #pil_img.save("xxxx1.jpg")
    
    pil_img.show()

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = change_ont_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
         
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
              
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def load_imagefile(filename, ref=False):
    img = Image.open(filename)
    img = img.resize((28,28), Image.ANTIALIAS)  #    Image.ANTIALIAS, BICUBIC
    img = img.convert("L")   #RGB, CMYK, L(256단계 흑백 이미지), 1(단색)
    
    if ref==True:
        img = ImageOps.invert(img)
    
    all_pixels = []
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            cpixel = pixels[y, x]
            all_pixels.append(cpixel/255.0)
            
    img_data = np.array(all_pixels,np.float32)  
    return img_data


def load_imagefile2(filename, ref=False):
    img = Image.open(filename)
    
    if img.format != 'PNG':
        img.save('tmp.png')
        img = Image.open('tmp.png')    
    
    img = img.resize((28,28), Image.ANTIALIAS)  #    Image.ANTIALIAS, BICUBIC
    img = img.convert("L")   #RGB, CMYK, L(256단계 흑백 이미지), 1(단색)
    
    if ref==True:
        img = ImageOps.invert(img)
    

            
    img_data = np.array([img.getdata()],np.float32)  
    return img_data/255.0

def Get_Modified_Image(in_filename,out_filename, ref=False, clean = False):
    img = Image.open(in_filename)
    
    if img.format != 'PNG':
        img.save('tmp.png')
        img = Image.open('tmp.png')    
    
    img = img.resize((28,28), Image.ANTIALIAS)  #    Image.ANTIALIAS, BICUBIC
    img = img.convert("L")   #RGB, CMYK, L(256단계 흑백 이미지), 1(단색)
    
    if ref==True:
        img = ImageOps.invert(img)
    

            
    img_data = np.array(img.getdata(),np.float32)
    if clean==True:  
        img_data = [(lambda x: 0 if x <100 else x)(x) for x in img_data]
        img_data = np.array([(lambda x: 255 if x >230 else x)(x) for x in img_data],np.float32)
    img_data = img_data.reshape(28,28)
    mod_img = Image.fromarray(np.uint8(img_data))
    mod_img.save(out_filename)
   

def MNIST_Image_Test():
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    
    image_num = 0
    img = x_train[image_num]
    label = t_train[image_num]

    #Print_List(img)

    print ("Data Shape: ", img.shape,label.shape )
    print ("img: ", np.sum(img))
    img2 = img.reshape(28,28)

    #Print_List(img2)

    pil_img = Image.fromarray(np.uint8(img2))
    pil_img.show()
    
    
    
    img_data = list(pil_img.getdata())
    print ("img_data: ", np.sum(img_data))
 
    pil_img.save("MNIST" + str(image_num) +".png")
    
    
    
    
    img3 = Image.open("MNIST" + str(image_num) +".png")
    img_data2 = list(img3.getdata())
    print ("img_data2: ", np.sum(img_data2))
    
    #Print_List(img_data2)    

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
 
def Print_List(x):
    for i in range(len(x)):
        print (x[i])    

def main1():
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)

    data_num = 507
    img = x_train[data_num]
    label = t_train[data_num]
    img = img.reshape(28,28)
    
    pil_img = Image.fromarray(np.uint8(img))

    pil_img.show()
    
    pil_img.save("Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\tmp.png")
    img2 = Image.open("Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\tmp.png")
    
    img2_data = list(img2.getdata())
    img2.close()
    
    for i in range(len(x_train[300])):
        print(x_train[data_num][i], "\t", img2_data[i])

def main2():
    n_data = 100
    n_input = 784
    n_hidden = 100
    n_output = 10
    net = TwoLayerNet(input_size=n_input, hidden_size=n_hidden, output_size=n_output)
    
    x = np.random.rand(n_data,n_input)
    
    # y = net.predict(x)
    
    t = np.random.rand(n_data,n_output)
    
    grads = net.numerical_gradient(x, t)
    
    print(grads)
    

def MultiLayerNet_Test():
    start = time.time()
    print ((datetime.datetime.now()), " Start") 
    
    n_data = 100
    n_input = 784
    n_hidden = 50
    n_output = 10    
    
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    # 하이퍼파라미터
    iters_num = 10000  # 반복 횟수를 적절히 설정한다.
    train_size = x_train.shape[0]
    batch_size = 100   # 미니배치 크기
    learning_rate = 0.1    
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 1에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)    
    
    
    #net = TwoLayerNet(input_size=n_input, hidden_size=n_hidden, output_size=n_output)
    #net = TwoLayerNet2(input_size=n_input, hidden_size=n_hidden, output_size=n_output)
    net = MultiLayerNet(input_size=n_input,hidden_size_list =[30,100,100,30], output_size=n_output, activation='relu',weight_init_std='relu',weight_decay_lambda=0)
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        #grad = net.numerical_gradient(x_batch, t_batch)
        grad = net.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= learning_rate * grad[key]
    
        # 학습 경과 기록
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        
        # 1에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))        
    
    
    finish = time.time()
    print (int((finish - start)/60.0), "Min", (finish - start)%60, "Sec elapsed")     
    
        
    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()        
    plt.close()
    
    path = "Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\"    
    
#     REF = False
#     Test_Files = ["A23.png","A35.png","A501.png","A503.png","A507.png"]


#     REF = True
#     Test_Files = ["My1.png","My2.png","My3.png","My4.png","My5.png"]
    
    
#     REF = False
#     Test_Files = ["80.png","81.png","82.png","83.png"]
    
    REF = False
    Test_Files = ["40.png","41.png","50.png","51.png","60.png","61.png","70.png","71.png","800.png", "90.png","91.png"]  
  
    for i in range(len(Test_Files)):
        my_test_data = load_imagefile2(Test_Files[i],ref=REF)
        my_result = net.predict_from_learning(my_test_data)
        print(Test_Files[i], np.argmax(my_result), my_result)        
        img = Image.open(Test_Files[i])
        plt.figure()
        plt.title('Predicted: ' + str(np.argmax(my_result)))
        plt.imshow(img)  
        plt.show()
        plt.close()
    
#     print( np.argmax(t_train[215]), np.argmax([round(e,4) for  e in net.predict(x_train[215])]) )
#     
#     my_test_data = load_imagefile2("MY1.png",ref=False)
#     my_result = net.predict(my_test_data)
#     print( np.argmax(my_result), my_result )
    
    
    print("hi")     
    
if __name__ == '__main__':
    # init_mnist()
    #main1()
    #main2()
    MultiLayerNet_Test()
    #MNIST_Image_Test()
    #my_test_data = load_imagefile2("80.png",ref=False)
    #Get_Modified_Image("60.png","61.png",ref=True, clean=True)