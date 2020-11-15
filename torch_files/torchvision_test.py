# coding: utf-8
'''
https://pytorch.org/docs/stable/torchvision/models.html


C:\Anaconda3\Lib\site-packages\torchvision\models   ---> 이곳에서 모델 구조를 봐야 한다.  ---> forward()를 살펴볼 것!!!


- pretrained=True로 모델을 생성하면, trainable variable들의 값이 down받은 값으로 초기화 되어 있다.
- model의 일부를 단순히 교체하든지, 

'''
import torch
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import os
from PIL import Image
def test1():
    # https://pytorch.org/docs/stable/torchvision/models.html
    # default: pretrained=False, progress=True(If True, displays a progress bar of the download to stderr)
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

    print(vgg16)



class MyVGG(nn.Module):
    def __init__(self,original_model):
        super(MyVGG, self).__init__()
        # vgg16에는 features, avgpool, classifier로 나누어져 있다.
        self.features = nn.Sequential(*list(original_model.features.children())[:-2])
        #self.features = nn.Sequential(*list(original_model.children())[:-9])
        
        self.classifier = nn.Sequential(nn.Linear(self.features[-1].out_channels,100),nn.ReLU(),nn.Linear(100,5))
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, (x.shape[2:]))  #(N,512,14,14) --> (N,512,1,1)
        x = x.view(x.shape[:2])             # --> (N,512)
        x = self.classifier(x)
        
        return x  
def test2():
    # 책 "파이토치 첫걸음" pp 75
    os.environ['TORCH_HOME'] = './pretrained'
    vgg16 = models.vgg16(pretrained=True, progress=True)  # 540M
 
    #transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    img = Image.open('dog.jpg')  # (576, 768, 3)
    img = transform(img)  # torch.Size([3, 576, 768])
    
    imgs = torch.unsqueeze(img,0)
    feature1 = vgg16(imgs)  # ---> (N,1000)
    
    
    
    extractor = MyVGG(vgg16)
    print("Vgg16:",vgg16)
    print("Extractor:",extractor)
    
    feature2 = extractor(imgs)
    print(feature1.shape, feature2.shape)
    
    print('Done')
def test3():
    # https://github.com/pytorch/vision/issues/484   ---> imagenet_classes.txt, imagenet_synsets.txt

    with open('imagenet_synsets.txt', 'r') as f:
        synsets = f.readlines()
    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}
    with open('imagenet_classes.txt', 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]  # ['n01440764', 'n01443537', 'n01484850', ...]

    os.environ['TORCH_HOME'] = './pretrained'   # default: C:\Users\BRAIN/.cache\torch
    
    model = models.resnet50(pretrained=True, progress=True)  # 106M
    #model = models.inception_v3(pretrained=True, progress=True)  # 100M
 
    #transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),  #(H,W,C) --> (C,H,W)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    img = Image.open(r'D:\hccho\TF2\creative_commons_elephant.jpg')  # dog(576, 768, 3) elephant(600, 899, 3) ---> np.array(img)해 보면, uint8
    img = transform(img)  # torch.Size([3, 224, 224])   ---> eg. -1.91 ~ 2.36 사이 값
    
    imgs = torch.unsqueeze(img,0)
    
    model.eval()
    pred = model(imgs)  # ---> (N,1000)  ---> softmax 취하기 전.
    pred = pred[0]
    _,class_id = pred.max(-1)

    
    class_key = class_id_to_key[class_id]
    classname = key_to_classname[class_key]

    print("{}".format(classname))
    
    
    _, indices = torch.sort(pred, descending=True)
    percentage = torch.nn.functional.softmax(pred) * 100

    
    result =[(key_to_classname[class_id_to_key[idx]], percentage[idx].item()) for idx in indices[:5]]
    print(result)
    
    
    
    print('Done')
def test4():
    # 책 "파이토치 첫걸음" pp 74
    # Dataset: 상속하여 class로 작성. __len__(), __getitem__() 필수 
    
    from torch.utils.data.sampler import SubsetRandomSampler
    transform=transforms.Compose([transforms.Resize(size=(128,128)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    mydataset = datasets.ImageFolder('D:/hccho/CommonDataset/101_ObjectCategories',transform=transform)   # 이 디렉토리안에 sub-directory로 분류되어 있어야 한다.
    print(len(mydataset))
    print(mydataset.classes)
    print(mydataset.class_to_idx)
    
    print('sample: ', mydataset[-5][0].shape, mydataset[-5][1])
    
    
    
    dataloader = torch.utils.data.DataLoader(mydataset, batch_size=8,shuffle=True,num_workers=2)    
    for i, d in enumerate(dataloader):
        print(d[0].shape, d[1])
        if i>1: break
    
    
    print('='*20)
    # dataset 전체가 아닌 경우, SubsetRandomSampler가 유용하다. ---> train/valid로 누우어야 하는 경우.
    # balanced sampling:  https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    indices = list(range(len(mydataset)))
    np.random.shuffle(indices)
    split = int(0.7*len(mydataset))
    train_idx = indices[:split]
    sampler = SubsetRandomSampler(train_idx)    # torch.utils.data.WeightedRandomSampler
    dataloader2 = torch.utils.data.DataLoader(mydataset,sampler=sampler, batch_size=8)
    for i, d in enumerate(dataloader2):
        print(d[0].shape, d[1])
        if i>10: break    

if __name__ == '__main__':
    #test1()
    test2()
    #test3()
    #test4()
    
    
    print('Done')
