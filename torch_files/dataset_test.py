# coding: utf-8
'''
from torch.utils.data import TensorDataset, Dataset DataLoader


1. numpy array 된 data는 TesnorDataset으로 변환하면 된다.   ---> DataLoader에 넘긴다.

2. Dataset을 상속하여 데이터 셋을 만든다.  ----> 이렇게 만든 데이터셋을 DataLoader에 넘기면 된다.



DataLoader에서 num_workers > 0일 때는 code에  __main__이 있어야 한다.   -----> drop_last의 default는 False이다.


'''



import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class MyDataset(Dataset): 
    """ Diabetes dataset.""" 
    # Initialize your data, download, etc. 
    def __init__(self): 
        self.len = 9
        self.x_data = torch.from_numpy(np.random.randn(self.len,4)) 
        self.y_data = torch.from_numpy(np.random.rand(self.len,1)) 
    
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.len


def Mycollate_fn(batch):
    # 후처리 작업: batch data 중에서 max길이를 보고, 그에 맞게 padding을 해주는 작업 같은 것은 collate_fn에서 해줄 수 밖에 없다.
    
    # test 목적으로 batch로 묶는 작업.  ----> 지금 여기서하는 batch를 묶는 작업은 collate_fn에서 해주지 않아도 자동으로 해준다.
    # 여기서 batch_size로 묶는 작업이 필요하다.
    # batch:  data 하나씩,  batch_size만큼의 list 
    x, y = zip(*batch)
    
    #return torch.cat([t.unsqueeze(0) for t in x], 0), torch.cat([t.unsqueeze(0) for t in y], 0)
    return torch.stack(x), torch.stack(y)
    



def test1():
    mydataset = MyDataset()
    
    train_loader = DataLoader(dataset=mydataset, batch_size=8, shuffle=True, num_workers=2,drop_last=True,collate_fn=Mycollate_fn)


    for i, data in enumerate(train_loader):
        print(data[0].size(), data[1].size(), data)
if __name__ == '__main__':

    test1()
    
    print('Done')
    
