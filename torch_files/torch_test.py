# coding: utf-8
'''
https://pytorch.org/tutorials/


'''



import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.datasets import load_digits
import tqdm

def test1():
    print(torch.__version__)
    
    # 중첩 list를 지정
    t = torch.tensor([[1, 2], [3, 4.]])

    # device를 지정하면 GPU로 Tensor를 만들 수 있다
    #t = torch.tensor([[1, 2], [3, 4.]], device="cuda:0")

    # dtype을 사용해 데이터형을 지정해 Tensor를 만들 수 있다
    t = torch.tensor([[1, 2], [3, 4.]], dtype=torch.float64)

    # 0부터 9까지의 수치로 초기화된 1차원 Tensor
    t = torch.arange(0, 10)

    
    #모든 값이 0인 100 x 10 의 Tensor를
    #작성해서 to메서드로 GPU에 전송
    #t = torch.zeros(100, 10).to("cuda:0")

    
    # 정규 난수로 100 x 10의 Tensor를 작성
    t = torch.randn(100, 10)
    
    # Tensor의 shape은 size 메서드로 취득 가능
    print(t.size(),t.shape)  # size(), shape은 alias
    
    a = torch.empty(5, 7, dtype=torch.float)
    a.fill_(3.5)
    b = a.add(4.0)
    
    x = torch.ones(5, 5)
    y = x.numpy()

def model1():
    # 참의 계수
    w_true = torch.Tensor([1, 2, 3])
    

    X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
    Y = torch.mv(X, w_true) + torch.randn(100) * 0.5
    w = torch.randn(3, requires_grad=True)
    
    # 학습률
    lr = 0.1    

    # 손실 함수의 로그
    losses = []
    
    # 100회 반복
    for epoc in range(100):
        # 전회의 backward 메서드로 계산된 경사 값을 초기화
        w.grad = None
        
        # 선형 모델으로 y 예측 값을 계산
        y_pred = torch.mv(X, w)
        
        # MSE loss와 w에 의한 미분을 계산
        loss = torch.mean((Y - y_pred)**2)
        loss.backward()
        
        # 경사를 갱신한다
        # w를 그대로 대입해서 갱신하면 다른 Tensor가 돼서
        # 계산 그래프가 망가진다. 따라서 data만 갱신한다
        w.data = w.data - lr * w.grad.data
        
        # 수렴 확인을 위한 loss를 기록해둔다
        losses.append(loss.item())
        
        if epoc%10==0:
            print('step: {}, loss = {:.4f}'.format(epoc,loss.item()))


    plt.plot(losses)
    plt.show()

def MultivariateRegression():
    # Regression 직접 구현
    device =  'cpu'  #'cuda:0'
    s = time.time()
    lr = 0.00002
    
    A = np.array([[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]])
    B = np.array([[152.],[185.],[180.],[196.],[142.]])



    AA = torch.tensor(A, dtype=torch.float32).to(device)
    BB = torch.tensor(B, dtype=torch.float32).to(device)
    w = torch.randn(3,1, requires_grad=True,device=device)
    b = torch.zeros(1,requires_grad=True,device=device)

    for step in range(2000):
        w.grad = None
        b.grad = None
        y = torch.mm(AA,w) + b

        loss = torch.mean((y-BB)**2)
        loss.backward()
        
        w.data = w.data - lr*w.grad.data
        b.data = b.data -lr*b.grad.data
        if step % 50 == 0:
            print('step: {}, loss = {:4f}'.format(step,loss))
        
    print(w,b,y)
    print('elapese: {} sec'.format(time.time()-s))
    
def MultivariateRegression2():
    # optim, nn.MSELoss() 이용
    device =  'cpu'  #'cuda:0'
    s = time.time()
    lr = 0.02
    
    A = np.array([[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]])
    B = np.array([[152.],[185.],[180.],[196.],[142.]])
    
    
    AA = torch.tensor(A, dtype=torch.float32).to(device)
    BB = torch.tensor(B, dtype=torch.float32).to(device)
    
    net = nn.Linear(in_features=3,out_features=1,bias=True)
    #torch.nn.init.normal(net.weight,0.0,0.01)
    
    
    optimizer = optim.Adam(net.parameters(),lr=lr)   # Adam은 learning rate이 너무 낮으면 안된다.
    #optimizer = optim.SGD(net.parameters(),lr=lr)
    
    loss_fn = nn.MSELoss()
    
    for step in range(2000):
        optimizer.zero_grad()
        y = net(AA)
        loss = loss_fn(y,BB)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('step: {}, loss = {:4f}'.format(step,loss))    
    
    print(y)
    print(net.weight,net.bias)
    print(list(net.parameters()))
    
    
def MultivariateRegression3():
    # MultivariateRegression2 + mini batch
    device =  'cpu'  #'cuda:0'
    s = time.time()
    lr = 0.2
    batch_size = 2
    
    
    A = np.array([[73., 80., 75.],
              [93., 88., 93.],
              [89., 91., 90.],
              [96., 98., 100.],
              [73., 66., 70.]])
    B = np.array([[152.],[185.],[180.],[196.],[142.]])
    nData = len(A)
    
    
    net = nn.Linear(in_features=3,out_features=1,bias=True)
    #torch.nn.init.normal(net.weight,0.0,0.01)
    
    
    optimizer = optim.Adam(net.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    
    net.to(device)
    for step in range(2000):
        choice = np.random.choice(nData,batch_size)
        x = torch.tensor(A[choice],dtype=torch.float32)
        y = torch.tensor(B[choice],dtype=torch.float32)
        
        
        optimizer.zero_grad()
        y_hat = net(x)
        loss = loss_fn(y_hat,y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('step: {}, loss = {:4f}'.format(step,loss))    
    
    print(y)
    print(net.weight,net.bias)
    print(list(net.parameters()))


def MNIST():
    # Multi Layer 모델, nn.Sequential() 이용
    # shuffle되어 있지 않다.
    digits = load_digits()     # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR']), (1797, 64), (1797,), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),(1797, 8, 8)

    shffule = np.random.choice(len(digits.data),len(digits.data))
    X = digits.data[shffule]
    Y = digits.target[shffule]
    
    # NumPy의 ndarray를 PyTorch의 Tensor로 변환
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.int64)
    
    
    net = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,10))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    
    for step in range(1000):
        optimizer.zero_grad()
        Y_hat = net(X)
        loss = loss_fn(Y_hat,Y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('step: {}, loss = {:4f}'.format(step,loss))    
    
    _, pred_ = torch.max(Y_hat,1)
    print(pred_[:15], Y[:15])
    print(len(list(net.parameters())))
    for a in net.parameters():
        print(a.shape)
    for a in net.named_parameters():
        print(a[0], a[1].shape)

    print('Done')
    


def MNIST2():
    # Network을 class로 구현
    batch_size = 8
    num_epoch = 1000
    
    # shuffle되어 있지 않다.
    digits = load_digits()     # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR']), (1797, 64), (1797,), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),(1797, 8, 8)
    X = torch.tensor(digits.data, dtype=torch.float32)
    Y = torch.tensor(digits.target, dtype=torch.int64)
    
    
    ds = TensorDataset(X,Y)  # tensor가 들어가야 한다.
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    
    
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            #self.net = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,16),nn.BatchNorm1d(16),nn.ReLU(),nn.Linear(16,10))
            self.net = nn.Sequential()
            self.net.add_module("L1", nn.Linear(64,32))
            self.net.add_module("L2", nn.ReLU())
            self.net.add_module("L3", nn.Dropout(0.3))
            self.net.add_module("L4", nn.Linear(32,16))
            self.net.add_module("L5", nn.BatchNorm1d(16))
            self.net.add_module("L6", nn.ReLU())
            self.net.add_module("L7", nn.Linear(16,10))
        def forward(self,x):
            return self.net(x)
    
    
    
    net = MyNet()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    net.train()  # train mode
    for epoch in range(num_epoch):
        for xx,yy in loader:
            optimizer.zero_grad()
            yy_hat = net(xx)
            loss = loss_fn(yy_hat,yy)
            loss.backward()
            optimizer.step()
        if epoch % 50 ==0:
            print('epoch: {}, loss = {:4f}'.format(epoch,loss))    
    
    net.eval()  # eval mode

    
    
    Y_hat = net(X)
    _, pred_ = torch.max(Y_hat,1)
    print(pred_[:15],'\n', Y[:15])
    print(len(list(net.parameters())))
    for a in net.named_parameters():
        print(a[0], a[1].shape)

    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)

    print('Done')

def conv_test():
    # pytorch는 convolution 연산후, image 크기 계산을 직접햐야 한다. tensorflow에서의 same padding이 없다.
    X = torch.randn(2, 3,100,100)
    # tensorflow의 same padding 같은 것은 업다.
    conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1,padding=[2,2])  # paddig(H,W)
    
    Y = conv(X)
    
    print(Y.shape)


def MNIST_conv():
    # Convolution 모델
    batch_size=128
    num_epoch=50
    
    # shuffle되어 있지 않다.
    digits = load_digits()     # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR']), (1797, 64), (1797,), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),(1797, 8, 8)

   
    X = torch.tensor(digits.images, dtype=torch.float32) # 8 x 8
    X = torch.unsqueeze(X,1)  # pytorch: N,C,H,W
    Y = torch.tensor(digits.target, dtype=torch.int64)
    
    
    
    ds = TensorDataset(X,Y)  # tensor가 들어가야 한다.
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    class FlattenLayer(nn.Module):
        def forward(self, x):
            sizes = x.size()
            return x.view(sizes[0], -1)
    class MyConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential()
            self.net.add_module("L1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1,padding=[2,2]))
            self.net.add_module("L2", nn.MaxPool2d(2))
            self.net.add_module("L3", nn.ReLU())
            self.net.add_module("L4", nn.Dropout(0.3))
            self.net.add_module("L5", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,padding=[2,2]))
            self.net.add_module("L6", nn.MaxPool2d(2))
            self.net.add_module("L7", nn.ReLU())
            self.net.add_module("L8", nn.Dropout(0.3))            
            self.net.add_module("L9", FlattenLayer())
            
            
            self.net.add_module("L10", nn.Linear(4*64,16))
            self.net.add_module("L11", nn.BatchNorm1d(16))
            self.net.add_module("L12", nn.ReLU())
            self.net.add_module("L13", nn.Linear(16,10))
        def forward(self,x):
            return self.net(x)
     
     
     
    net = MyConvNet()
     
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
     
    net.train()  # train mode
    
    

    for epoch in tqdm.tqdm(range(num_epoch)):
         
        for i,(xx,yy) in enumerate(loader):
 
            optimizer.zero_grad()
            Y_hat = net(xx)
            loss = loss_fn(Y_hat,yy)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print('step: {}, loss = {:4f}'.format(epoch,loss))    


    net.eval()  # eval mode
    Y_hat = net(X)
    _, pred_ = torch.max(Y_hat,1)
    print(pred_[:15],'\n', Y[:15])
    print(len(list(net.parameters())))
    for a in net.named_parameters():
        print(a[0], a[1].shape)
 
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
 
    print('Done')


def init_test():
    vocab_size=6
    embedding_dim = 8
    x_data = np.array([[0, 3, 1, 4, 3, 2],[0, 3, 4, 2, 3, 1],[0, 1, 3, 2, 2, 1]], dtype=np.int32)

    X = torch.tensor(x_data, dtype=torch.int64) #int64이어야 된다.
    
    emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
    
    #z = np.random.randn(6,8).astype(np.float32)
    z = np.arange(vocab_size*embedding_dim).reshape(-1,embedding_dim).astype(np.float32)
    
    emb.weight.data=torch.from_numpy(z)   # numpy array를 이용하여 초기화
    

    z = emb(X) # (3,6) -->(3,6,embedding_dim)
    print(emb.weight)
    print(z[0])
def RNN_test():
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    embedding_dim = 8
    hidden_dim =20
    num_layers = 2
    index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}
    x_data = np.array([[SOS_token, 1, 2, 3, 3, 4]], dtype=np.int32)
    y_data = np.array([[1, 2, 3, 3, 4,EOS_token]],dtype=np.int32)

    X = torch.tensor(x_data, dtype=torch.int64) #int64이어야 된다.
    Y = torch.tensor(y_data, dtype=torch.int64)
    
    class MyRNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.first_net=nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            
            self.lstm = nn.LSTM(embedding_dim,hidden_dim, num_layers,batch_first=True, dropout=0.2)
            
            self.last_net = nn.Sequential()
            self.last_net.add_module("L2", nn.Linear(hidden_dim,13))
            self.last_net.add_module("L3", nn.ReLU()) 
            self.last_net.add_module("L4", nn.Linear(13,vocab_size))
        
        def forward(self,x, h0=None):
            x = self.first_net(x)
            x,h = self.lstm(x,h0)
            
            x = self.last_net(x)
            return x,h
    
    
    net = MyRNN()
     
    loss_fn = nn.CrossEntropyLoss()  # 2dim에 대한 loss, seq loss는 안된다.
    optimizer = optim.Adam(net.parameters(),lr=0.01)
     
    net.train()  # train mode

    for epoch in range(500):
        optimizer.zero_grad()
        Y_hat,_ = net(X)
        loss = loss_fn(input=Y_hat.view(-1,vocab_size),target = Y.view(vocab_size))
        loss.backward()
        optimizer.step()
        
        if epoch %10 ==0:
            print('epoch: {}, loss = {:.4f}'.format(epoch,loss))




    net.eval()
    max_length = 20
    h=None
    input = np.array([[SOS_token]], dtype=np.int32)
    input = torch.tensor(input, dtype=torch.int64)
     
    result = []
    with torch.no_grad():
        for i in range(max_length):
            out,h=net(input,h)
             
            _,input = torch.max(out,2)
             
            ch = input.numpy()[0,0]
            if ch == EOS_token: break
            result.append(index_to_char[input.numpy()[0,0]])
     
     
        print(result)
 
    print(len(list(net.parameters())))
    for a in net.named_parameters():
        print(a[0], a[1].shape)
  
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
    print('Done')
    
    
    
    
    
if __name__ == '__main__':
    #test1()
    #model1()
    #MultivariateRegression()
    #MultivariateRegression2()
    #MultivariateRegression3()
    #MNIST()
    #MNIST2()
    
    #conv_test()
    #MNIST_conv()
    
    #init_test()
    RNN_test()


    print('Done')












