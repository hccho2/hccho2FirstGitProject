# coding: utf-8
'''
1. 모든 data를 memory에 올리는 못하는 경우에 Queue를 사용
2. data를 모두 memory에 올릴 수 있는 경우:  TensorDataset & DataLoader

3. data를 모두 memory에 올릴 수 없는 경우: Dataset을 상속받다, class를 만든 후, DataLoader

'''

import torch
import numpy as np
import os,glob,cv2
import threading,queue
from torch.multiprocessing import Process, Queue, Pool
import traceback
from torch.utils.data import TensorDataset,DataLoader,Dataset,Subset
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def prefetch_data(queue, batch_size,num_features):
    print("start prefetching data...")
    while True:
        try:
            data = torch.from_numpy(np.random.rand(batch_size,num_features))
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def init_jobs(queue, batch_size,num_features):
    # Process를 여러개 만들수도 있다. Peocess list로
    task = Process(target=prefetch_data, args=(queue, batch_size,num_features))

    task.daemon = True
    task.start()
    return task



def data_preprocessing(data_queue, processed_training_queue, sema):
    while True:
        data = data_queue.get()
        
        # 필요한 조작을 한다. (e.g. tensor를 pin momory로 올린다.)
        data = data*1000
        processed_training_queue.put(data)
    
        if sema.acquire(blocking=False):
            return

def train_by_queue():
    
    batch_size=2
    num_features=3
    
    training_queue   = Queue(3)
    training_tasks = init_jobs(training_queue, batch_size,num_features)
    
    processed_training_queue = queue.Queue(3)
    
    training_semaphore   = threading.Semaphore()
    training_semaphore.acquire()
    
    
    training_thread = threading.Thread(target=data_preprocessing, args=(training_queue,processed_training_queue,training_semaphore))
    training_thread.daemon = True
    training_thread.start()
    

    
    
    for i in range(100):
        data = processed_training_queue.get(block=True)
        data = data.to(DEVICE)
        
        print(data)
    
        del [data]
    
    
    training_tasks.terminate()
def train():
    # 간단한 data는 Dataset class를 만들지 않고 처리
    x_train1 = np.random.randn(100,3)
    x_train2 = np.random.randn(100,4)
    y_train = np.random.randn(100,1)
    
    x_train1 = torch.from_numpy(x_train1)
    x_train2 = torch.from_numpy(x_train2)
    y_train = torch.from_numpy(y_train)
    
    
    ds = TensorDataset(x_train1,x_train2,y_train)  # tensor를 넘겨야 한다.
    loader = DataLoader(ds,batch_size = 8, shuffle=True,drop_last=True,num_workers=4) # num_workers가 주어지면 훨씬 빨라진다.
    
    for epoch in range(1):
        for x1,x2,y in loader:  # enumerate(loader)
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            y = y.to(DEVICE)
            
            print(x2)
            
            del [x1,x2,y]
            
    print('Done')

def train_collate_fn():
    
    def my_collate(batch):
        # batch로 묶어 있는 data에 추가적인 작업을 해 줄 수 있다.
        # batch: batch_size 길이의 list
        # batch[0]: 길이 3짜리(x_train1, x_train2,y_train)
        
#         x1 = [item[0] for item in batch]
#         x2 = [(100*item[1]).int() for item in batch]
#         y = [item[2] for item in batch]

        x1,x2,y = zip(*batch)
        x2 = [(100*item).int() for item in x2 ]
        
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        y = torch.stack(y)
        print('---'*20)
        return x2,y
    
    
    
    # 간단한 data는 Dataset class를 만들지 않고 처리
    x_train1 = np.random.randn(100,3)
    x_train2 = np.random.randn(100,4)
    y_train = np.random.randn(100,1)
    
    x_train1 = torch.from_numpy(x_train1)
    x_train2 = torch.from_numpy(x_train2)
    y_train = torch.from_numpy(y_train)
    
    
    ds = TensorDataset(x_train1,x_train2,y_train)  # tensor를 넘겨야 한다.
    loader = DataLoader(ds,batch_size = 8, shuffle=True,drop_last=True,num_workers=0,collate_fn=my_collate) # num_workers가 주어지면 훨씬 빨라진다.
    
    for epoch in range(1):
        for x2,y in loader:  # enumerate(loader)
            x2 = x2.to(DEVICE)
            y = y.to(DEVICE)
            
            print(x2)
            
            del [x2,y]
            
            
    print('Done')
# data를 메모리에 올리지 못하는 경우, Dataset을 상속받아, class를 정의한다.
class myDataset(Dataset):
    '''
    __len__, __getitem__이 반드시 정의되어야 한다.
    '''
    def __init__(self, dir_path,imsize=(255,255)):
        self.image_filenames = glob.glob(os.path.join(dir_path, "*.jpg"))
        self.imsize = imsize
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self,idx):
        image = cv2.imread(self.image_filenames[idx])
        image = cv2.resize(image, self.imsize)  # resize하지 않으면, batch로 묶일 수 없어, error!!!
        return image # tensor로 변환하지 않아도 내부적으로 변환된다.
    
    @property
    def size(self):
        return len(self.image_filenames)
    
def train_with_Dataset():
        ds = myDataset('d:/hccho/CommonDataset/coco/images/val2017')
        print(ds.size)
        
        
        train_size = int(ds.size*0.9)
        test_size = ds.size - train_size
        train_ds, test_ds = torch.utils.data.random_split(ds, (train_size, test_size))   # 전제 dataset ds를 train, test로 분리
        #print(len(train_size))
        #print(len(test_ds))
        
        
        #loader = DataLoader(ds,batch_size = 8, shuffle=True,num_workers=8,drop_last=True,)  # num_workers가 주어지면 훨씬 빨라진다. 없을 땐 116초, num_workers=8이면, 8초
        
        train_loader = DataLoader(train_ds,batch_size = 8, shuffle=True,num_workers=8,drop_last=True,)
        test_loader = DataLoader(test_ds,batch_size = 8, shuffle=True,num_workers=8,drop_last=True,)




        for i, img in enumerate(train_loader):
            print(i, img.shape)
            
            
        
            
            
if __name__ == '__main__':
    s = time.time()
    #train_by_queue()

    #train()  # 모든 data를 메모리에 ....
    
    train_collate_fn()  # collate_fn
    
    
    #train_with_Dataset()

    print(time.time() -s , "sec elapsed")
