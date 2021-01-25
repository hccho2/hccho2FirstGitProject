'''
http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html


"a guide to convolution arithmetic for deep learning"


https://github.com/vdumoulin/conv_arithmetic    ---> animation



https://www.slideshare.net/ssuserb208cc1/transposed-convolution


https://distill.pub/2016/deconv-checkerboard/   ---> artifact checkboard effect

이곳이 가장 잘 설명되어 있다.
https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967 


https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb   ---> 행렬곱으로 convolution

https://nbviewer.jupyter.org/github/metamath1/ml-simple-works/blob/master/CNN/transconv_fullconv.ipynb
'''


import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F



def transepose_conv1():

    H=3;W=3
    kernel_size=2
    pad = 0
    stride=1
    image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)
    
    image = torch.tensor(image)
    
    print(image)

    weight = torch.tensor([[[[0.1767, 0.1274], [1.0128, 0.4777]]]])
    
    out = F.conv_transpose2d(image,weight,stride=stride,padding=pad)
    
    print(f'output size: {(H-1)*stride - 2*pad+ + kernel_size} x {(W-1)*stride - 2*pad + kernel_size}')
    print(f'weight: {weight}')
    print(f'output: {out}')
    
    image_padded = F.pad(image,(kernel_size-1-pad,kernel_size-1-pad,kernel_size-1-pad,kernel_size-1-pad))
    
    print(image_padded)
    
    
    out2 = F.conv2d(image_padded,torch.flip(weight,[2,3]))
    print(f'output2: {out2}')


    weight_r =  np.zeros((16,9),dtype=np.float32)
    a = np.array([[0.1767,0,0],[0.1274, 0.1767,0],[0, 0.1274, 0.1767],[0,0,0.1274]])
    b = np.array([[1.0128,0,0],[0.4777, 1.0128,0],[0, 0.4777, 1.0128],[0,0,0.4777]])
    weight_r[0:4,0:3] =  a; weight_r[4:8,3:6] =  a; weight_r[8:12,6:9] =  a
    weight_r[4:8,0:3] =  b; weight_r[8:12,3:6] =  b; weight_r[12:16,6:9] =  b

    #print(weight_r)
    weight_r = torch.from_numpy(weight_r).reshape(1,1,16,9)
    out3 = image.reshape(1,1,1,-1).matmul(weight_r.transpose(2,3)).reshape(1,1,4,4)
    print(f'output3: {out3}')


def transepose_conv2():
    H=3;W=3
    pad = 0
    stride=1
    kernel_size = 2 
    
    image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)   random image
    
    image = torch.tensor(image)
    
    print("random image", image)

    weight = torch.randn(1,1,kernel_size,kernel_size)

    
    out = F.conv_transpose2d(image,weight,padding=pad,stride=stride)

    print(f'output size: {(H-1)*stride - 2*pad+ + kernel_size} x {(W-1)*stride - 2*pad + kernel_size}')
    print(f'weight: {weight}')
    print(f'output: {out}')
          
          
    #####################################################
    #####################################################
    # transpose convolution by normal convolution
    image_stride = np.zeros((1,1,(H-1)*stride+1,(W-1)*stride+1))

    image_stride[:,:,::stride,::stride] = image
    image_stride = torch.tensor(image_stride,dtype=torch.float32)

    print('new image', image_stride)
    
    image_padded = F.pad(image_stride,(kernel_size-1 - pad,kernel_size-1 - pad,kernel_size-1 - pad,kernel_size-1 - pad))
    
    print("padded image", image_padded)
    
    
    out2 = F.conv2d(image_padded,torch.flip(weight,[2,3]),padding=0,stride = 1)
    print(f'manual output: {out2}')



def backward_test():
    
    image = torch.randn(1,1,4,4,requires_grad=True)
    weight = torch.randn(1,1,3,3,requires_grad=True)
    
    z = F.conv2d(image,weight)
    print(f'image shape: {image.shape}, weight shape: {weight.shape}, output shape: {z.shape}')
    print(f'image: {image}, weight: {weight}')
    print(f'z: {z}')

    z.retain_grad()
    z.backward(z,retain_graph=True)  # retain_graph=True가 있어야, 아래의 image2.backward에서 error가 나지 않는다.
    
    print(f'image.grad: {image.grad}')



    image2 = F.conv_transpose2d(z,weight)
    print(f'transposed conv: {image2}')  # image.grad와 일치
    
    z.grad.zero_()
    image2.backward(image)
    print(f'z.grad: {z.grad}')

from scipy.linalg import toeplitz
def unfold_kernel(kernel, input_size):
    # kernel을 (HxW) x (OH x OW) 크기로 변형해 준다.
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz_list  = []
    for r in range(k_h):
        toeplitz_list.append(toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))) ) 

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz_list[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))  # OH,OW,H,W

    for i, B in enumerate(toeplitz_list):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv.T



def unfold_kernel_stride(kernel, input_size, stride=1):
    # kernel을 (HxW) x (OH x OW) 크기로 변형해 준다.
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = (i_h-k_h)//stride + 1, (i_w-k_w)//stride + 1  # stride=1인 경우

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz_list  = []
    for r in range(k_h):
        toe = toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w)))
        toeplitz_list.append( toe[::stride] ) 

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz_list[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz_list):
        for j in range(o_h):
            W_conv[j, :, i+j*stride, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv.T



def unfold_kernel_test():

    H=6;W=4
    kernel_size = 2
    
    w = np.random.randn(kernel_size,kernel_size)
    #w = np.arange(1,5).reshape(2,2)*10
    
    
    with np.printoptions(precision=3, suppress=True):
        print(f'kernel: \n{w}')
        w_unfold = unfold_kernel(w,(H,W))
        print(f'unfold kernel: {w_unfold.shape}\n{w_unfold}')
    
    
    OH = H-kernel_size + 1
    OW = W-kernel_size + 1
    image = np.random.rand(1,1,H,W)
    
    with np.printoptions(precision=3, suppress=True):
        out = F.conv2d(torch.Tensor(image),torch.Tensor(w)[None,None])
        print(f'pytoch conv: \n{out.numpy()}')
        out2 = np.matmul(image.reshape(1,-1),w_unfold).reshape(OH,OW)
        print(f'matmul conv: \n{out2}') 
    


    stride = 2
    OH = (H-kernel_size)//stride + 1
    OW = (W-kernel_size)//stride + 1    
    w_unfold = unfold_kernel_stride(w,(H,W),stride)

    with np.printoptions(precision=3, suppress=True):
        out = F.conv2d(torch.Tensor(image),torch.Tensor(w)[None,None],stride=stride)
        print(f'pytoch conv: \n{out.numpy()}')
        out2 = np.matmul(image.reshape(1,-1),w_unfold).reshape(OH,OW)
        print(f'matmul conv: \n{out2}') 








    
def test():
    # https://nbviewer.jupyter.org/github/metamath1/ml-simple-works/blob/master/CNN/transconv_fullconv.ipynb
    # transposed convolution이 convolution의 backward라는 걸 확인.  ---> transposed convolution과 직접 연관이 없다고도 할 수 있음.
    # 임의의 인풋과 필터에 대해서 포워드 패스를 수행한다.
    I = torch.randn(1 ,1, 4, 4, requires_grad=True)
    w = torch.randn(1, 1, 3, 3)
    z = F.conv2d(I, w, stride=1, padding=0)
    C = (torch.sigmoid(z)).sum() # f(z) = sum (sigmoid(z)) 로 정의
    
    print('z')
    print(z)
    
    print('C')
    print(C)
    
    
    w_unfold =  torch.tensor(unfold_kernel(w[0,0].numpy(),(4,4)),dtype=torch.float)
    
    
    ###########################################################
    # 파이토치의 autograd를 이용해 일단 dI를 구한다.
    ###########################################################
    dI_torch = torch.autograd.grad(C, I, retain_graph=True)[0]
    print('dI by torch')
    print(dI_torch)
    
    print('=='*10)
    
    ##########################################################
    # eq(3)으로 dI를 구한다.
    ##########################################################
    # delta = dC/dz를 구한다. 
    delta = torch.autograd.grad(C, z, retain_graph=True)[0]
    print('delta')
    print(delta)
    
    # delta를 180도 돌리고
    delta_flip  = torch.flip(delta, [2, 3])  # shape: (1, 1, 2, 2)
    w_flip  = torch.flip(w, [2, 3])
    print('delta flip')
    print(delta_flip)
    
    # w에 패딩을 주고 컨벌루션한다.
    print('dI')
    # convolution이 교환법칙이 성립하는 것을 보여주고 있다.   ---> padding이 고려되었을 때 성립하는 교환법칙.
    dI = F.conv2d(w, delta_flip, padding=1)  # F.conv_transpose2d(delta,w) = F.conv2d(delta,w_flip, padding=2)
    print(dI)
    
    
    ###########################################################
    
    
    
    print( "행렬곱 미분으로 계산: \n", torch.matmul(delta.reshape(1,-1),w_unfold.T).reshape(4,4) )    

if __name__ == '__main__':
    #transepose_conv1()
    #transepose_conv2()
    #backward_test()
    
    unfold_kernel_test()
    #test()












