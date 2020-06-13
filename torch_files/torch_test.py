# coding: utf-8
'''
>pip install torch
>conda install pytorch torchvision cpuonly -c pytorch

https://pytorch.org/tutorials/
----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # --> device(type='cuda')
device = torch.device('cuda:0')                                        # --> device(type='cuda', index=0)

mynet.to(device) # 또는 mynet.cuda(), mynet.cpu()
y = x.to(device)  # inplace방식 아님. x.to(device)하면, x가 바뀌지는 않는다. 
------
x.cpu().detach().numpy()
x.cpu().data.numpy()
------------
변수 직접 선언
w = torch.randn(3,1, requires_grad=True,device=device)

from torch.autograd import Variable
w = Variable(torch.randn(3, 1).type(dtype), requires_grad=True)  ----> 이것보다 nn.Parameters 추천

----
nn.Parameter(torch.empty(3, 2))  # trainable Tensor


------
with torch.no_grad():
    traget = ...
y = forward(x)
loss = loss_fn(y, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

또는
with torch.no_grad():
    traget = ...
optimizer.zero_grad()
y = forward(x)
loss = loss_fn(y, target)
loss.backward()
optimizer.step()


또는

traget = ...
optimizer.zero_grad()
y = forward(x)
loss = loss_fn(y, target.detatch())
loss.backward()
optimizer.step()

-----
# save
torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))  # weights만 저장
또는 
torch.save(model, os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))   # 모델의 구조까지 저장
----
# restore
model.load_state_dict(torch.load('xxx.pth', map_location = device))  # model이 이미 정의된 상태
또는
model = torch.load('xxx.pth')  # 모델의 구조까지 복원
-----
network weights copy
net1.load_state_dict(net2.state_dict())
-----
optimizer = optim.Adam(net.parameters(),lr=lr) 
loss_fn = nn.MSELoss()

optimizer.zero_grad()  ----> net.parameters()에 있는 weight들의 grad 값을 0으로 만든다.
y = net(xx) ----> forward
loss = loss_fn(y,BB)
loss.backward()     -----> net.parameters()에 있는 weight들의 grad를 계산한다.
optimizer.step()   ------> net.parameters()에 있는 weight들의 값의 grad로 update한다.


-----
PyTorch에서는 모델을 저장할 때 .pt 또는 .pth 확장자를 사용하는 것이 일반적인 규칙입니다.  ---> pt, pth는 차이가 나지는 않고, 선택의 문제임.

-----
nn.CrossEntropyLoss: 2D 또는 3D logit이 넘어잘 수 있다. (N,C) 또는 (N,T,C)        target은 one-hot으로 변환하지 않은 것이 넘어간다.

-----
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # scheduler.step()이 30분 call하면 0.1을 곱한다.
# StepLR은 ExponentialLR을 좀 더 정교하게 .... 본질적으로 동일.
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)  # 매 step마다 gamma를 곱한다.StepLR과 다를게 뭔가????
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
for i in range(100):
    print(i, scheduler.get_lr(), scheduler.get_last_lr())
    scheduler.step()


-----
Attention Mask
http://juditacs.github.io/2018/12/27/masked-attention.html
'''



import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.datasets import load_digits
import tqdm,math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def show_weights(net):
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)    
    for a in net.named_parameters():
        print(a[0], a[1].shape)

def test0():
    z = np.random.randn(2,3)  # float64
    w = np.random.randint(5, size=[2,3])  # int32
    x1 = torch.from_numpy(z)  # torch.float64   --> numpy의 dtype을 유지.
    x2 = torch.Tensor(z)  # torch.float32  torch.FloatTensor와 동일
    x3 = torch.FloatTensor(z) # torch.float32

    y1 = torch.from_numpy(w)  # torch.int32
    y2 = torch.Tensor(w)  # torch.float32
    
    z = torch.Tensor(2,3)  # torch.empyt(2,3)과 동일. Returns a tensor filled with uninitialized data
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
    t = torch.zeros(100,10).data.normal_(0,1)
    t = torch.zeros_like(t).data.normal_(0,1)
    
    # Tensor의 shape은 size 메서드로 취득 가능
    print(t.size(),t.shape)  # size(), shape은 alias
    
    a = torch.empty(5, 7, dtype=torch.float)
    a.fill_(3.5)
    b = a.add(4.0)
    
    x = torch.ones(5, 5)
    y = x.numpy()


def test2():
    x = torch.Tensor(2,3)  # garbage로 초기화된 주어진 크기의 tensor를 만든다.  shape(2,3)
    print(x,x.shape)

    
    # 방법 1
    x.uniform_(5,7)  #5~7 사이의 uniform random값으로 최기화
    print(x)

    nn.init.normal_(x,0,1) # N(0,1)로 x를 초기화
    print(x)
    
    # x.data는 할당도 가능하다.
    print(x.data, x.numpy())  # x.item() x가 scalar인 경우에만

    print('\n.data -------------')
    x.data = torch.Tensor(1,2)
    print(x,x.shape)

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
        
    print("w,b,y",w,b,y)
    print('elapese: {} sec'.format(time.time()-s))

    
def MultivariateRegression1():
    from torch.autograd import Variable

    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor # GPU에서 실행하려면 이 주석을 제거하세요.

    # N은 배치 크기이며, D_in은 입력의 차원입니다;
    # H는 은닉 계층의 차원이며, D_out은 출력 차원입니다:
    N, D_in, H, D_out = 64, 1000, 100, 10

    # 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성하고, Variable로
    # 감쌉니다. requires_grade=False로 설정하여 역전파 중에 이 Variable들에 대한
    # 변화도를 계산할 필요가 없음을 나타냅니다.
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성하고, Variable로
    # 감쌉니다. requires_grad=True로 설정하여 역전파 중에 이 Variable들에 대한
    # 변화도를 계산할 필요가 있음을 나타냅니다.
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
    bias = Variable(torch.randn(D_out).type(dtype), requires_grad=True)
    learning_rate = 1e-6
    for t in range(500):
        # 순전파 단계: Variable 연산을 사용하여 y 값을 예측합니다. 이는 Tensor를 사용한
        # 순전파 단계와 완전히 동일하지만, 역전파 단계를 별도로 구현하지 않기 위해 중간
        # 값들(Intermediate Value)에 대한 참조(Reference)를 갖고 있을 필요가 없습니다.
        y_pred = x.mm(w1).clamp(min=0).mm(w2) + bias

        # Variable 연산을 사용하여 손실을 계산하고 출력합니다.
        # loss는 (1,) 모양을 갖는 Variable이며, loss.data는 (1,) 모양의 Tensor입니다;
        # loss.data[0]은 손실(loss)의 스칼라 값입니다.
        loss = (y_pred - y).pow(2).sum()
        if t % 10 ==0:
            print(t, loss.data)

        # autograde를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를
        # 갖는 모든 Variable에 대한 손실의 변화도를 계산합니다. 이후 w1.grad와 w2.grad는
        # w1과 w2 각각에 대한 손실의 변화도를 갖는 Variable이 됩니다.
        loss.backward()

        # 경사하강법(Gradient Descent)을 사용하여 가중치를 갱신합니다; w1.data와
        # w2.data는 Tensor이며, w1.grad와 w2.grad는 Variable이고, w1.grad.data와
        # w2.grad.data는 Tensor입니다.
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.data.zero_()
        w2.grad.data.zero_()    
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
    
    params_num = sum([np.prod(p.size()) for p in net.parameters()])
    print('params_num: ', params_num)
    
    
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
        x = torch.tensor(A[choice],dtype=torch.float32).to(device)
        y = torch.tensor(B[choice],dtype=torch.float32).to(device)
        
        
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
    shffle = np.arange(len(digits.data))
    np.random.shuffle(shffle)
    
    X = digits.data[shffule]
    Y = digits.target[shffule]
    
    # NumPy의 ndarray를 PyTorch의 Tensor로 변환
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.int64)
    
    
    net = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,10))
    
    loss_fn = nn.CrossEntropyLoss()  # 넘길 때, (N,C), (N,)  <--- one_hot으로 변환하지 않는 target을 넘긴다.
    optimizer = optim.Adam(net.parameters())
    
    
    for step in range(1000):
        optimizer.zero_grad()
        Y_hat = net(X)
        
        '''
        network을 loop로 통과시킬 수도 있다
        xx = X
        for l in net:
            xx= net(xx)
        '''
        
        loss = loss_fn(Y_hat,Y)  # Y_hat: (N, 10), Y: (N,)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('step: {}, loss = {:4f}'.format(step,loss))    
    
    
    
    with torch.no_grad():
        _, pred_ = torch.max(Y_hat,1)
    
    
    print(pred_[:15], Y[:15])
    print(len(list(net.parameters())))
    for a in net.parameters():
        print(a.shape)
    for a in net.named_parameters():
        print(a[0], a[1].shape)

    print("Net: ",net)
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
    
    # net.parameters() ---> train 가능한 weight가 들어있다. requires_grad을 False로 만들 수 있다.
    # net.state_dict() ---> bn의 moving mean 같은 것들도 들어 있다.
    
    for t in net.state_dict():
        print(t)
        
        
    for name, param in net.named_parameters():
        print (name, param.requires_grad, param.data)
    
    # 직접 접근할 수도 있다.
    net.net.L1.weight.data
    net.net.L1.weight.grad
        
        
    params_num = sum([np.prod(p.size()) for p in net.parameters()])
    print('params_num: ', params_num) 
        
    
#     for t in net.parameters():
#         t.requires_grad = False
    
    
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
def MNIST3():
    # nn.ModuleList를 이용하여, network을 구분.
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
            self.net1 = nn.Sequential()
            self.net1.add_module("L1", nn.Linear(64,32))
            self.net1.add_module("L2", nn.ReLU())
            self.net1.add_module("L3", nn.Dropout(0.3))
            
            self.net2 = nn.Sequential()
            self.net2.add_module("L4", nn.Linear(32,16))
            self.net2.add_module("L5", nn.BatchNorm1d(16))
            self.net2.add_module("L6", nn.ReLU())
            self.net2.add_module("L7", nn.Linear(16,10))
            
            self.net = nn.ModuleList([self.net1,self.net2])
        def forward(self,x):
            for l in self.net:
                x = l(x)
            
            return x
    
    
    
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
        print(a[0], a[1].shape)  # a[1].data (Tensor)  --> a[1].data.numpy() --> (numpy array)
    print('=='*10)
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
    
    print('Done')

def MNIST4():
    # data download
    # root로 지정한 디렉토리 아래에, MNIST\processed, MNIST\raw가 만들어진다.
    
    from torchvision import datasets, transforms
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    
    train_loader  = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=2, shuffle=True, num_workers=4)
    
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
    
    
    print('-----')
def conv_test():
    # pytorch는 convolution 연산후, image 크기 계산을 직접햐야 한다. tensorflow에서의 same padding이 없다.
    X = torch.randn(2, 3,100,100)  # pytorch는 channel_frist 방식
    # tensorflow의 same padding 같은 것은 업다.
    conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1,padding=[2,2])  # paddig(H,W)
    
    Y = conv(X)
    
    print(Y.shape)
    
    
    # Adaptive Pooling: output size로 설정되고, output 크기에 따라, kernel_size가 내부적으로 결정된다.
    m1 = nn.AdaptiveAvgPool1d(2)
    m2 = nn.AdaptiveAvgPool2d((None,7)) # None이면 크기가 변하지 않는다.
    input1 = torch.randn(1, 64, 8)
    output1 = m1(input1)
    
    
    input2 = torch.randn(1, 64, 20,20)
    input3 = torch.randn(1, 64, 10,10)
    output2 = m2(input2)
    output3 = m2(input3)
    print(output1.size(),output2.size(),output3.size())

def MNIST_conv():
    # Convolution 모델
    
    batch_size=128
    num_epoch=100
    
    # shuffle되어 있지 않다.
    digits = load_digits()     # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR']), (1797, 64), (1797,), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),(1797, 8, 8)

   
    X = torch.tensor(digits.images, dtype=torch.float32) # 8 x 8
    X = torch.unsqueeze(X,1)  # pytorch: N,C,H,W
    Y = torch.tensor(digits.target, dtype=torch.int64)
    
    
    
    ds = TensorDataset(X,Y)  # tensor가 들어가야 한다.
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    
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
     
     
     
    net = MyConvNet().to(device)
     
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
     
    net.train()  # train mode
    
    

    for epoch in tqdm.tqdm(range(num_epoch)):
         
        for i,(xx,yy) in enumerate(loader):
            xx = xx.to(device)
            yy = yy.to(device)
            optimizer.zero_grad()
            Y_hat = net(xx)
            loss = loss_fn(Y_hat,yy)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print('step: {}, loss = {:4f}'.format(epoch,loss))    


    net.eval()  # eval mode
    Y_hat = net(X.to(device))
    _, pred_ = torch.max(Y_hat,1)
    print(pred_[:15],'\n', Y[:15])
    print(len(list(net.parameters())))
    for a in net.named_parameters():
        print(a[0], a[1].shape)
 
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
 
    print('Done')


def init_test():
    # weight initialization from numpy array
    vocab_size=6
    embedding_dim = 8
    x_data = np.array([[0, 3, 1, 4, 3, 2],[0, 3, 4, 2, 3, 1],[0, 1, 3, 2, 2, 1],[2, 2, 2, 2, 2, 2]], dtype=np.int32)

    X = torch.tensor(x_data, dtype=torch.int64) #int64이어야 된다.
    
    emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
    
    z = np.random.randn(vocab_size,embedding_dim).astype(np.float32)
    #z = np.arange(vocab_size*embedding_dim).reshape(-1,embedding_dim).astype(np.float32)
    
    emb.weight.data=torch.from_numpy(z)   # numpy array를 이용하여 초기화
    

    z = emb(X) # (batch_size,T) -->(batch_size,T,embedding_dim)
    print(emb.weight)
    print(z[0])
    
    
    ##############################################
    ##############################################
    print('='*20)
    hidden_size=5
    
    rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size,num_layers=1,bias=True,nonlinearity='tanh', batch_first=True, dropout=0, bidirectional=False)
    
    y,h = rnn(z)  # stateful
    
    
    
    print(y.shape,h.shape)
    

def init_test2():
    
    
    net = nn.Linear(in_features=3,out_features=1,bias=True)
    print(net.state_dict())
    


    #torch.nn.init.xavier_uniform(net.weight)
    #torch.nn.init.xavier_normal_(net.weight)
    torch.nn.init.normal_(net.weight, mean=0.0, std=1.0)
    torch.nn.init.zeros_(net.bias)
    
    
    print(net.state_dict())
    
    
    print('Done')
    
    
def init_test3():  
    # weight initialization
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            #self.net = nn.Linear(in_features=3,out_features=1,bias=True)
            
            self.net = nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,2))
            
        def forward(self,x):
            return self.net(x)
    
    def weights_init(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)  # m.bias.data.fill_(0.)  
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias) 
    model = MyNet()
    print(model.state_dict())
    
    
    
    model.apply(weights_init)  # apply는 nn.Module로 부터 상속.
    print(model.state_dict())

    
    
def BCE_test():

    target = torch.tensor([[1,1,0,0,1],[0,0,1,1,1]],dtype=torch.float32)
    logit = torch.randn(2,5)
    weight = torch.tensor([0.5,2,0.5,0.5,10])  # All weights are equal to 1
    
    criterion1 = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='none')   # pos_weight는 positive label에만 부여하는 weight
    criterion2 = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='mean')  # (N,T) 전체 평균.
    
    print(criterion1(logit, target)) # logit, target: (N,T)
    print(criterion2(logit, target)) 
    
    p = torch.sigmoid(logit)
    
    a = -(target*torch.log(p) + (1-target)*torch.log(1-p))*weight
    print(a)
    print(torch.mean(a))
    
    
    
    
def RNN_test00():
    # nn.LSTM vs nn.LSTMCell
    batch_size = 2
    input_size=3 # embedding dim
    hidden_size = 4 # hidden size
    num_layers = 7 # LSTM을 몇단으로 쌓을 지...
    T= 5 # seq length
    
    h0 = torch.randn(num_layers,batch_size,hidden_size)
    c0 = torch.randn(num_layers,batch_size,hidden_size)
    rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True )
    
    input = torch.randn(batch_size, T, input_size)
    
    output, (hn, cn) = rnn(input, (h0, c0))  # (batch_size,T,hidden_dim), h(num_layers,batch_size,hidden_dim), c(num_layers,batch_size,hidden_dim)
    
    print(output.shape, hn.shape,cn.shape)  

def RNN_test11():
    # nn.LSTMCell
    batch_size = 2
    input_size=3 # embedding dim
    hidden_size = 4 # hidden size
    T= 5 # seq length

    h0 = torch.randn(batch_size,hidden_size)
    c0 = torch.randn(batch_size,hidden_size)
    rnn = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    input = torch.randn( T,batch_size, input_size)

    for i in range(T):
        hn, cn = rnn(input[i], (h0, c0))  # (batch_size,T,hidden_dim), h(num_layers,batch_size,hidden_dim), c(num_layers,batch_size,hidden_dim)

    print(hn.shape)



def RNN_test():
    mode = 1 #   1---> train mode, 0 ---> infer mode
    
    if mode==0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './saved_model/xxx.pt'
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    embedding_dim = 8
    hidden_dim =7
    num_layers = 2
    index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)

#     x_data = np.array([[SOS_token, 1, 2, 3, 3, 4]], dtype=np.int32)
#     y_data = np.array([[1, 2, 3, 3, 4,EOS_token]],dtype=np.int32)


    X = torch.tensor(x_data, dtype=torch.int64).to(device) #int64이어야 된다.  (N,T): embedding 전
    Y = torch.tensor(y_data, dtype=torch.int64).to(device)  # (N,T)
    
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
            x1 = self.first_net(x)  # x:(N,T), x1:(N,T,embedding_dim)
            x2,h = self.lstm(x1,h0)  # dropout이 있어, 같은 input에 대하여 값이 달라질 수 있다.
            # hidden state: ( h(num_layers, batch_size, hidden_dim), c(num_layers, batch_size, hidden_dim) )
            
            #loop를 이용한 계산
            if True:   #For loop를 돌릴려면, nn.LSTM보다는 nn.LSTMCell을 하는 것이 더 적절한다.   -----> 더 적절한게 아니고, Cell로 해야한다.
                max_len = x.size(1)
                lstm_output = []
                hh=h0
                for i in range(max_len):
                    tmp,hh = self.lstm(x1[:,i:i+1,:],hh)
                    lstm_output.append(tmp)
                    
                x22 = torch.cat(lstm_output,dim=1)  # x2와 numerical한 차이만 있다.
            #######################
            
            
            x3 = self.last_net(x2)
            return x3,h
    
    
    net = MyRNN()
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()  # 2dim에 대한 loss, seq loss는 안된다.  --->된다. 아래에 loss2
    optimizer = optim.Adam(net.parameters(),lr=0.01)
     

    
    if mode == 1: 
        net.train()  # train mode
        for epoch in range(500):
            optimizer.zero_grad()
            Y_hat,_ = net(X)  # X: (3,6)   Y_hat: (3,6,6)
            loss = loss_fn(input=Y_hat.view(-1,vocab_size),target = Y.view(-1))  # input: (18,6), target: (18)
            loss2 = loss_fn(input=torch.transpose(Y_hat,1,2),target = Y)  # (N,D,T), (N,T)로도 가능
            
            assert np.abs(loss.item() - loss2.item()) < 0.000001
            
            loss.backward()  # loss2.backward()
            optimizer.step()
            
            if epoch %100 ==0:
                print('epoch: {}, loss = {:.4f}'.format(epoch,loss))


    else:
        #net.load_state_dict(torch.load(save_path,map_location = device))
        net.load_state_dict(torch.load(save_path))  # gpu에서 train 되었는데, map_location 지정하지 않아도 OK.

    net.eval()
    max_length = 20
    h=None
    input = np.array([[SOS_token]], dtype=np.int32)
    input = torch.tensor(input, dtype=torch.int64).to(device)
     
    result = []
    with torch.no_grad():
        for i in range(max_length):
            out,h=net(input,h)
             
            _,input = torch.max(out,2)
             
            ch = input.cpu().numpy()[0,0]
            if ch == EOS_token: break
            result.append(index_to_char[input.cpu().numpy()[0,0]])
     
     
        print(result)



    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
#     print("Optimizer's state_dict:")
#     for var_name in optimizer.state_dict():   # dict_keys(['state', 'param_groups'])
#         print(var_name, "\t", optimizer.state_dict()[var_name])
 
    print("# of params: ", len(list(net.parameters())))
    
    
    for a in net.named_parameters():
        print(a[0], a[1].shape)
  
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
    print('Done')
    
    
    # Save. 디렉토리를 미리 만들어야 한다.
    if mode ==1:  # 1--> train mode
        torch.save(net.state_dict(), save_path)
    
    
def PackedSeq_test():
    '''
    https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
    pytorch의 rnn 모듈은 PackedSequence가 들어왔는지를 판단하여 처리한다.
    '''
    input_seq2idx= torch.tensor([[  1,  16,   7,  11,  13,   2],
        [  1,  16,   6,  15,   8,   0],
        [ 12,   9,   0,   0,   0,   0],
        [  5,  14,   3,  17,   0,   0],
        [ 10,   0,   0,   0,   0,   0]])
    
    

    input_lengths = torch.LongTensor([torch.max(input_seq2idx[i, :].data.nonzero())+1 for i in range(input_seq2idx.size(0))])
    input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
    input_seq2idx = input_seq2idx[sorted_idx]
    
    
    print(input_seq2idx)
    
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_seq2idx, input_lengths.tolist(), batch_first=True)
    print(packed_input)
    
    
    print('='*20)
    
    ###################################
    ###################################

    embedding_dim = 3
    hidden_size = 5
    emb = nn.Embedding(50, embedding_dim, padding_idx=0)
    rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    
    embed=emb(input_seq2idx)
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths.tolist(), batch_first=True)
    
    out,h=rnn(packed_input)  # h도 길이에 맞게 마지막 state가 들어있다.
    out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    
    out2,h2=rnn(embed)  # h2는 길이에 상관없이 제일 끝 hidden state가 들어가 있다.
    print(out,output_lengths)  # padded 부분에 garbage가 없다.
    print(out2)  # padded부분도 계산이되어, garbage가 들어 있다.
    
def bidirectional_test():
    batch_size = 2
    T = 5
    embedding_dim = 4
    vocab_size = 20
    hidden_size = 3
    
    input = torch.from_numpy(np.random.randn(0, vocab_size, size=(batch_size, T,embedding_dim)).astype(np.float32))
    rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,bidirectional=True)
    
    out,h = rnn(input) # out에는 forward, backward가 concat되어 있다.
    
    print(out.shape,h.shape) # out.shape = (batch_size,T,2*hidden_dim), 마지막 앞의 hidden_dim개는 forward, 뒤의 hidden_dim개는 backward
    '''
out([[[ 1.0000, -0.9972, -1.0000,  1.0000,  1.0000,  0.9347],
         [ 1.0000, -1.0000,  0.9931,  1.0000, -0.9455, -1.0000],
         [ 1.0000,  0.8011, -1.0000,  0.9968,  1.0000,  0.9996],
         [ 1.0000, -0.9998, -0.7761,  1.0000,  0.9998, -0.5244],
         [ 1.0000, -1.0000, -1.0000,  1.0000,  0.9515, -0.9956]],

        [[ 1.0000, -1.0000, -0.9850,  1.0000,  1.0000,  0.9973],
         [ 1.0000, -0.9974, -1.0000,  1.0000,  1.0000, -0.0949],
         [ 1.0000, -1.0000, -0.9984,  1.0000,  0.6022, -0.2628],
         [ 1.0000, -0.9973, -1.0000,  1.0000,  1.0000,  0.9879],
         [ 1.0000, -0.9606, -0.9993,  0.9962,  1.0000,  0.9973]]],


h([[[ 1.0000, -1.0000, -1.0000],
         [ 1.0000, -0.9606, -0.9993]],

        [[ 1.0000,  1.0000,  0.9347],
         [ 1.0000,  1.0000,  0.9973]]]  
    '''
    
    
def Loss_test():
    # NLLLoss: negative log likelihood loss
    # CrossEntropyLoss  == (logit -> softmax -> log -> NLLLoss)
    # NLLLoss는 넣어주는 값중에 lable에 해당하는 값에 마이너스 붙혀주는 역할.
    with torch.no_grad(): 
        loss1 = nn.NLLLoss()
        loss2 = nn.CrossEntropyLoss()
        
        logit = torch.tensor([[1.0,2.0,1.5],[3.0,1.0,4.0]], requires_grad=True, dtype=torch.float)
        
        target = torch.tensor([2,1])   # target label  ---> one hot 2 == (0,0,1)
        
        # logit(N, n_class), target: N
        loss1_ = loss1(logit, target)   # -torch.mean(logit[np.arange(2),target])
        loss2_ = loss2(logit, target)   # cross entropy loss
        
        print(loss1_,loss2_)
        
        
        softmax_val = torch.nn.functional.softmax(logit,1)
        log_softmax_val = torch.log(softmax_val)  # logit.log_softmax(1) 값과 같다.
        print("softmax: ", softmax_val)
        print("log-softmax: ", log_softmax_val)
        print("Cross Entropy loss: ", -torch.mean(log_softmax_val[np.arange(2),target]))
        
        print("Cross Entropy Loss by NLLLosss: ", loss1(log_softmax_val,target))
        
        print("NLLLost: ", -torch.mean(logit[np.arange(2),target]))
    
def Loss_Seq_test():
    # sequence loss를 구하기 위해서는, (N,T,n_class)가 아닌, (N,n_class,T)형태로 logit에 넣어 주어야 한다. target(N,T)
    with torch.no_grad(): 
        loss1 = nn.NLLLoss()
        loss2 = nn.CrossEntropyLoss()  # 평균은 N*T로 나누어서 취한다.
        
        batch_size = 3
        T = 4
        D = 5
        logit = torch.randn(batch_size, T, D)
        
        target = torch.randint(D,size=(batch_size,T))
        
        # logit(N, n_class), target: N
        loss1_ = loss1(torch.transpose(logit,1,2), target)   # (N,D,T), (N,T)
        loss2_ = loss2(torch.transpose(logit,1,2), target)   # cross entropy loss
        
        print("loss비교: ", loss1_,loss2_)
        
        softmax_val = torch.nn.functional.softmax(logit,2)
        log_softmax_val = torch.log(softmax_val)
        print("Cross Entropy loss before mean: ", log_softmax_val.view(-1,D)[np.arange(batch_size*T),target.view(-1)])
        print("Cross Entropy loss: ", -torch.mean(log_softmax_val.view(-1,D)[np.arange(batch_size*T),target.view(-1)]))
        
def Loss_Mask_test():
    with torch.no_grad(): 
        # sequence loss를 구하기 위해서는, (N,T,n_class)가 아닌, (N,n_class,T)형태로 logit에 넣어 주어야 한다. target(N,T)

        loss2 = nn.CrossEntropyLoss(reduction = 'none')  # keep dim
        # (N,T,n_class)= (2,3,3)
        logit = torch.tensor([[[1.0,2.0,1.5],[3.0,1.0,4.0],[1.0,1.2,1.5]],[[1.0,2.0,1.5],[3.0,1.0,2.5],[2.0,1.0,1.0]]], dtype=torch.float)
        
        target = torch.tensor([[0,1,0],[1,0,2]])   # target label  ---> one hot 2 == (0,0,1)
        
        # logit(N, n_class), target: N
        loss2_ = loss2(torch.transpose(logit,1,2), target)   # cross entropy loss
         
        print('loss2: ', loss2_)
        
        length = torch.tensor([3,2])
        maxlen = logit.size(1)
        mask = torch.arange(maxlen)[None, :] < length[:, None]   # tensor([[ True,  True,  True],[ True,  True, False]])
        
        print("mask: ",mask)
        mask= mask.type(torch.float)
        loss = torch.sum(loss2_*mask)/torch.sum(mask)
        print(loss)
        
def Attention_Mask():
    # mask test 1.
    X = torch.arange(12).view(4, 3)
    mask = torch.zeros((4, 3), dtype=torch.uint8)  # or dtype=torch.ByteTensor
    mask[0, 0] = 1; mask[1, 1] = 1; mask[3, 2] = 1;  # masking할 곳 지정하기.
    X[mask] = 100
    print(X)   

    # mask test 2. 
    mask = mask.bool() # bool 형식으로 변환해야 ~mask가 제대로 작동한다.
    X = torch.arange(12).view(4, 3)
    X[~mask] = -100
    print(X)


    # generating the actual toy example
    print('='*10)
    np.random.seed(1)
    X = np.random.random((4,6))  # (batch_size, max_sequence_length). attention score
    X = torch.from_numpy(X)
    X_len = torch.LongTensor([4, 1, 6, 3])  # length of each sequence


    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :] < X_len[:, None]
    
    print(mask)
    print('Bofore X: ', X)
    X[~mask] = float('-inf')
    print('After X: ', X)
    
    softmax_val = torch.nn.functional.softmax(X,1)  # score -> alignment
    print('masked softmax: ', softmax_val)
    
def network_copy():
    # network copy
    net1 = nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,2))
    net2 = nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,2))
    
    x = torch.from_numpy(np.random.randn(4,2).astype(np.float32))
    
    print('net1', net1(x))
    print('net2', net2(x))
    
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        p1.data.copy_(p2.data)  # net1 <--- net2
        #p1.data.copy_(0.3*p1.data+0.7*p2.data)
    
    print('net1', net1(x))
    print('net2', net2(x))

################################################################
def dropout_test():
    
    class MyTestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop_layer = nn.Dropout(0.5)

        def forward(self, x):
            return self.drop_layer(x)
    
    net = MyTestNet()
    
    x = torch.ones(2,3,5)
    
    print(net(x))
    
    net.eval()
    print(net(x))
    
###############################################################



###############################################################
    
###############################################################



###############################################################
def UserDenfedLayer_test():
    
    # 사용자 정의 Layer 만들기.
    class NoisyLinear(nn.Module):
        # Dense Layer를 변형하여 Noise Layer
        def __init__(self, in_features, out_features, std_init=0.4):
            super(NoisyLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            
            # register_buffer를 통해 tensor 생성. ---> non trainable.
            # register_buffer를 통해, state_dict에 추가된다.
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            
            
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
            self.reset_parameters()  # trainable variable 초기화
            self.sample_noise()      # non trainable variable인 noise update
    
        def reset_parameters(self):
            # trainable variable 초기화
            mu_range = 1.0 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
        def _scale_noise(self, size):
            x = torch.randn(size)
            return x.sign().mul_(x.abs().sqrt_())
    
        def sample_noise(self):
            # noise 자유도를 줄이기 위해.  ---> 논문 참조
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))   # ger: outer product
            self.bias_epsilon.copy_(epsilon_out)
    
        def forward(self, inp):
            if self.training:
                return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
            else:
                return F.linear(inp, self.weight_mu, self.bias_mu)
            
    L = NoisyLinear(3,2)
    x = torch.Tensor(5,3)
    y = L(x)
    
    for l in L.parameters():
        print(l)
    
    L.state_dict()
    L.state_dict()['weight_epsilon']
    L.state_dict()['weight_sigma']
###############################################################



###############################################################

###############################################################



###############################################################

###############################################################



###############################################################

###############################################################



###############################################################
    
if __name__ == '__main__':
    #test1()
    #test2() # tensor 생성과 초기화
    #model1()
    #MultivariateRegression()
    #MultivariateRegression2()
    #MultivariateRegression3()
    #MNIST()
    #MNIST2()
    #MNIST3()
    #MNIST4()
    #conv_test()
    #MNIST_conv()
    
    #init_test()
    #init_test2()
    #init_test3()
    #RNN_test00()
    #RNN_test11()
    #RNN_test()
    #PackedSeq_test()
    
    #bidirectional_test()


    #Loss_test()
    #Loss_Seq_test()
    #Loss_Mask_test()
    #Attention_Mask()
    dropout_test()
    print('Done')

