'''
http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html


"a guide to convolution arithmetic for deep learning"


https://github.com/vdumoulin/conv_arithmetic    ---> animation


'''


import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F



def transepose_conv1():

    H=4;W=3
    kernel_size=2
    pad = 1
    stride=1
    image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)
    
    image = torch.tensor(image)
    
    print(image)

    #weight = torch.randn(1,1,kernel_size,kernel_size)
    weight = torch.tensor([[[[0.1767, 0.1274], [1.0128, 0.4777]]]])
    
    out = F.conv_transpose2d(image,weight,stride=stride,padding=pad)
    
    print(f'output size: {(H-1)*stride - 2*pad+ + kernel_size} x {(W-1)*stride - 2*pad + kernel_size}')
    print(f'weight: {weight}')
    print(f'output: {out}')
    
    image_padded = F.pad(image,(kernel_size-1-pad,kernel_size-1-pad,kernel_size-1-pad,kernel_size-1-pad))
    
    print(image_padded)
    
    
    out2 = F.conv2d(image_padded,torch.flip(weight,[2,3]))
    print(f'output2: {out2}')

def transepose_conv2():
    H=2;W=2
    pad = 1
    stride=2
    kernel_size = 3 
    
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
    
    

if __name__ == '__main__':
    #transepose_conv1()
    transepose_conv2()
    #backward_test()












