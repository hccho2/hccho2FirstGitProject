'''
http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

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
    stride=2
    image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)
    
    image = torch.tensor(image)
    
    print(image)
    
    #weight = torch.randn(1,1,kernel_size,kernel_size)
    weight = torch.tensor([[[[0.1767, 0.1274], [1.0128, 0.4777]]]])
    
    out = F.conv_transpose2d(image,weight)
    
    print(f'output size: {(H-1)*stride - 2*pad + kernel_size}')
    print(f'weight: {weight}')
    print(f'output: {out}')
    
    image_padded = F.pad(image,(pad,pad,pad,pad))
    
    print(image_padded)
    
    
    out2 = F.conv2d(image_padded,torch.flip(weight,[2,3]))
    print(f'output2: {out2}')

def transepose_conv2():
    H=4;W=3
    pad = 1
    stride=2
    kernel_size = 3 
    
    image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)   random image
    
    image = torch.tensor(image)
    
    print("random image", image)

    weight = torch.randn(1,1,kernel_size,kernel_size)
    
    
    out = F.conv_transpose2d(image,weight,padding=pad,stride=stride)

    print(f'output size: {(H-1)*stride - 2*pad+ + kernel_size} x {(W-1)*stride - 2*pad+ + kernel_size}')
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







if __name__ == '__main__':
    #transepose_conv1()
    transepose_conv2()












