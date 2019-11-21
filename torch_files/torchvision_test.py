# coding: utf-8
'''
https://pytorch.org/docs/stable/torchvision/models.html


C:\Anaconda3\Lib\site-packages\torchvision\models   ---> 이곳에서 모델 구조를 봐야 한다.  ---> forward()를 살펴볼 것!!!


'''
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import os
from PIL import Image
def test1():
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet161()
    inception = models.inception_v3()
    googlenet = models.googlenet()
    shufflenet = models.shufflenet_v2_x1_0()
    mobilenet = models.mobilenet_v2()
    resnext50_32x4d = models.resnext50_32x4d()
    wide_resnet50_2 = models.wide_resnet50_2()
    mnasnet = models.mnasnet1_0()
    
class MyVGG(nn.Module):
    def __init__(self,original_model):
        super(MyVGG, self).__init__()
        # vgg16에는 features, avgpool, classifier로 나누어져 있다.
        self.features = nn.Sequential(*list(original_model.features.children())[:-3])
        #self.features = nn.Sequential(*list(original_model.children())[:-9])
    def forward(self, x):
        x = self.features(x)
        return x  
def test2():
    os.environ['TORCH_HOME'] = './pretrained'
    vgg16 = models.vgg16(pretrained=True, progress=True)  # 540M
 
    transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    img = Image.open('dog.jpg')  # (576, 768, 3)
    img = transform(img)  # torch.Size([3, 576, 768])
    
    imgs = torch.unsqueeze(img,0)
    
    feature1 = vgg16(imgs)  # ---> (N,1000)
    
    
    
    extractor = MyVGG(vgg16)
    
    featres2 = extractor(imgs)
    
    print('Done')
def test3():
    os.environ['TORCH_HOME'] = './pretrained'
    resnet50 = models.resnet50(pretrained=True, progress=True)  # 100M
 
    transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    img = Image.open('dog.jpg')  # (576, 768, 3)
    img = transform(img)  # torch.Size([3, 576, 768])
    
    imgs = torch.unsqueeze(img,0)
    
    feature1 = resnet50(imgs)  # ---> (N,1000)
    
    
    print('Done')

if __name__ == '__main__':
    #test1()
    #test2()
    test3()
    
    
    print('Done')

