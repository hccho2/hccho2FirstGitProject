import torch
from torch import nn,optim
from matplotlib import pyplot as plt
import numpy as np
import time
def test1():
    print(torch.__version__)
    
    # 중첩 list를 지정
    t = torch.tensor([[1, 2], [3, 4.]])
    print(t)
    # device를 지정하면 GPU로 Tensor를 만들 수 있다
    t = torch.tensor([[1, 2], [3, 4.]], device="cuda:0")
    print(t)
    # dtype을 사용해 데이터형을 지정해 Tensor를 만들 수 있다
    t = torch.tensor([[1, 2], [3, 4.]], dtype=torch.float64)
    print(t)
    # 0부터 9까지의 수치로 초기화된 1차원 Tensor
    t = torch.arange(0, 10)
    print(t)
    
    #모든 값이 0인 100 x 10 의 Tensor를
    #작성해서 to메서드로 GPU에 전송
    t = torch.zeros(100, 10).to("cuda:0")
    print(t.size())
    
    
    # 정규 난수로 100 x 10의 Tensor를 작성
    t = torch.randn(100, 10)
    
    # Tensor의 shape은 size 메서드로 취득 가능
    print(t.size())

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
    
    
    optimizer = optim.Adam(net.parameters(),lr=lr)
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
if __name__ == '__main__':
    #test1()
    #model1()
    #MultivariateRegression()
    MultivariateRegression2()