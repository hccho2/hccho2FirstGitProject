'''
http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

'''


import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

H=4;W=3
kernel_size=2
pad = 0
stride=1
image = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)  # (N,C,H,W)

image = torch.tensor(image)

print(image)

#weight = torch.randn(1,1,kernel_size,kernel_size)
weight = torch.tensor([[[[0.1767, 0.1274], [1.0128, 0.4777]]]])

out = F.conv_transpose2d(image,weight)

print(f'output size: {(H-1)*stride - 2*pad + kernel_size}')
print(f'weight: {weight}')
print(f'output: {out}')

image_padded = F.pad(image,(1,1,1,1))

print(image_padded)


out2 = F.conv2d(image_padded,torch.flip(weight,[2,3]))
print(f'output2: {out2}')


