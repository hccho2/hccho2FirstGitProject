'''
ImageFolder로 읽은 data를 train/split으로 나누기:  ImageFolder를 만들면서, transformer를 적용하는데, train/valid 용 2가지가 적용되어야 한다.
따라서, ImageFolder를 만들면서는 transformer를 적용하지 않아야 한다.
1. torch.utils.data.random_split ---> train, valid 분리

2. map style로 dataset을 다시 만들어야 한다. https://pytorch.org/docs/stable/data.html#map-style-datasets
    --> 사용자 정의 Dataset


'''




import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os,glob,shutil,time
from tqdm import tqdm
print('pytorch version: ',torch.__version__)
print('torchvision version: ', torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['TORCH_HOME'] = './pretrained'
model_dir='./saved_model'

def move_files_to_subdirectory():
    # 하나의 디렉토리에 모여 있는 파일을 각각의 sub directory로 이동
    category =['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 
               'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 
               'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 
               'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 
               'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 
               'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

    
    base_dir = r'D:\hccho\CommonDataset\Pet-Dataset-Oxford\images'  
    
    # 디렉토리 만들기
    for c in category:
        dir_name = os.path.join(base_dir,c)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    
    all_files = glob.glob(os.path.join(base_dir,"*.jpg"))
    
    print("파일갯수: ", len(all_files))
    
    # file move
    for f in all_files:
        basename = os.path.basename(f)
        c = '_'.join(basename.split('_')[:-1])   # basset_hound_103.jpg --> ['basset', 'hound', '103.jpg']
        if os.path.exists(os.path.join(base_dir,c)):
            shutil.move(f,os.path.join(base_dir,c))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15,15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

def get_dataloader(batch_size=32):
    data_dir = r'D:\hccho\CommonDataset\Pet-Dataset-Oxford\images'   # 테스트를 위해, data몇개만 모아, 작은 dataset을 만듬.
     
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

     
    dataset  =  datasets.ImageFolder(data_dir,transform=None)
    class_names = dataset.classes
    n_data = len(dataset )
    print(len(dataset ),len(class_names), class_names)
    
    # fixed seed  generator=torch.Generator().manual_seed(42)
    train_dataset0, valid_dataset0 = torch.utils.data.random_split(dataset, [int(n_data*0.8), n_data - int(n_data*0.8)],generator=torch.Generator().manual_seed(42))
    

    train_dataset = MyDataset(train_dataset0,train_transforms)
    valid_dataset = MyDataset(valid_dataset0,valid_transforms)
        
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)    
    
    return train_dataloader, valid_dataloader,class_names


def test():
    train_dataloader, valid_dataloader,class_names = get_dataloader(batch_size=16)



    it = iter(valid_dataloader)
    for i in range(2):
        inputs, classes = next(it)
    
        out = torchvision.utils.make_grid(inputs)  # inputs: 5, 3, 224, 224  ---> out: 3, 228, 1132
        imshow(out, title=[class_names[x] for x in classes])


def train():
    n_epoch=10
    batch_size = 64
    
    train_dataloader, valid_dataloader,class_names = get_dataloader(batch_size)

    model = models.resnet18(pretrained=True, progress=True)
    model.fc = nn.Linear(model.fc.in_features,len(class_names))
    
    model.to(device)   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode = 'max',threshold_mode='abs',threshold=0.001, factor=0.6,patience=10, min_lr=0.0001,verbose=True)
    
    s_time = time.time()
    best_acc = 0
    best_duration=0
    n_step = len(train_dataloader)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
    
        running_loss = []
        acc = 0
        total = 0
        
        model.train()
        with tqdm(total=n_step,ncols=100) as pbar:
            for i, data in enumerate(train_dataloader):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, pred = outputs.max(axis=-1)
                acc  += (pred==labels).float().sum().item()
                total += len(labels)
        
        
                # print statistics
                running_loss.append(loss.item())
                pbar.set_description("epoch: {}, loss: {:.4f}, train acc:{:.4f}".format(epoch+1,loss.item(), acc/total))
                pbar.update(1)
        print('[epoch: %d] loss: %.3f, train acc: %.3f elapsed: %.2f' % (epoch + 1, np.mean(running_loss), acc/total , time.time()-s_time),end="\t" )
        #scheduler.step()
        acc = 0
        total=0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                _, pred = outputs.max(axis=-1)
                acc  += (pred.cpu()==labels).float().sum().item()
                total += len(labels)
        if acc/total > best_acc:
            best_acc = acc/total
            print('val acc: %43f, best: %.3f ===== new best ' % (best_acc,best_acc ))
            best_duration = 0
            
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}-{:.4f}.pth'.format(epoch,best_acc)))
        else:
            best_duration += 1
            print('val acc: %.4f, best: %.3f, best duration: %d ' % (acc/total,best_acc, best_duration ) )
        scheduler.step(best_acc)    

def evaluate():
    batch_size = 64
    
    train_dataloader, valid_dataloader,class_names = get_dataloader(batch_size)

    model = models.resnet18(pretrained=True, progress=True)
    model.fc = nn.Linear(model.fc.in_features,len(class_names))    
    
    model.load_state_dict(torch.load(os.path.join(model_dir,'epoch-6-0.9012.pth')))
    
    model.to(device)
    model.eval()
    acc = 0
    total=0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, pred = outputs.max(axis=-1)
            acc  += (pred.cpu()==labels).float().sum().item()
            total += len(labels)    
    print('val acc: %.4f' % (acc/total ))
    
    

if __name__ == '__main__':
    #move_files_to_subdirectory()
    
    #test()
    
    #train()
    
    evaluate()
