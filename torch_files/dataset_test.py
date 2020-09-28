# coding: utf-8
'''
from torch.utils.data import TensorDataset, Dataset DataLoader


1. numpy array 된 data는 TesnorDataset으로 변환하면 된다.   ---> DataLoader에 넘긴다.

2. Dataset을 상속하여 데이터 셋을 만든다.  ----> 이렇게 만든 데이터셋을 DataLoader에 넘기면 된다.



DataLoader에서 num_workers > 0일 때는 code에  __main__이 있어야 한다.   -----> drop_last의 default는 False이다.


'''



import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset,ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MyDataset(Dataset): 
    """ Diabetes dataset.""" 
    # Initialize your data, download, etc. 
    def __init__(self,type=0): 
        self.len = 9  # 전체 data 갯수
        if type==0:
            
            self.x_data = []
            for _ in range(self.len):
                random_length = np.random.randint(3,7)  # data의 길이가 random하다고 가정하는 역할.
                self.x_data.append(torch.from_numpy(np.random.randn(random_length))) 
            self.y_data = torch.from_numpy(np.random.rand(self.len,1)) 
        else:
            self.x_data = []
            for _ in range(self.len):
                random_length = np.random.randint(3,7)
                self.x_data.append(2*torch.from_numpy(np.ones(random_length))) 
            self.y_data = torch.from_numpy(np.zeros([self.len,1]))         
    
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.len


def Mycollate_fn(batch):
    # 후처리 작업: batch data 중에서 max길이를 보고, 그에 맞게 padding을 해주는 작업 같은 것은 collate_fn에서 해줄 수 밖에 없다.
    
    # test 목적으로 batch로 묶는 작업.  ----> 지금 여기서하는 batch를 묶는 작업은 collate_fn에서 해주지 않아도 자동으로 해준다.
    # 여기서 batch_size로 묶는 작업이 필요하다.
    # batch:  data 하나씩,  batch_size만큼의 list ----> 여기서는 mydataset1,mydataset2에서 random하게 추출되어서 mydataset1,2가 섞여 있다.
    x, y = zip(*batch)
    
    #return torch.cat([t.unsqueeze(0) for t in x], 0), torch.cat([t.unsqueeze(0) for t in y], 0)
    return pad_sequence(x,batch_first=True,padding_value=99), torch.stack(y)
    



def test1():
    mydataset1 = MyDataset(0)
    mydataset2 = MyDataset(1)
    mydataset = ConcatDataset([mydataset1,mydataset2])  # ConcatDataset은 data혼합(병렬 아님)
    
    train_loader = DataLoader(dataset=mydataset, batch_size=2, shuffle=True, num_workers=2,drop_last=True,collate_fn=Mycollate_fn)

    for i, data in enumerate(train_loader):
        print(data[0].size(), data[1].size(), data, '\n')
    
    
    
def test2():
    # iter을 이용해서 data를 계속 공급할 수 있게 한다.
    mydataset = MyDataset(0)
    train_loader = DataLoader(dataset=mydataset, batch_size=2, shuffle=True, num_workers=2,drop_last=True,collate_fn=Mycollate_fn)
    
    loader_iter = iter(train_loader)
    
    for i in range(10):
        try:
            a,b = loader_iter.next()  # data가 다 소진되면 error
            print(a,b)
        except StopIteration:
            print('='*20)
            loader_iter = iter(train_loader)  # data가 다 소진되었으니, 다시 reset.
            a,b = loader_iter.next()  # data가 다 소진되면 error
            print(a,b)        
        
        

if __name__ == '__main__':

    #test1()
    test2()
    
    print('Done')
    
