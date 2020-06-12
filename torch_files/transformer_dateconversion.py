# coding: utf-8

'''
https://discuss.pytorch.org/t/nn-transformer-explaination/53175


'''


import numpy as np
import torch
from torch import nn,optim
import torchtext
import pandas as pd
import time,os
from glob import glob
from natsort import natsorted
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # --> device(type='cuda')
vocab_filename = 'vocab.pickle'
model_save_dir = './saved_model'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

def get_latest_model(model_path):
    model_list = glob(os.path.join(model_path, '*.pth'))
    if len(model_list) <=0:
        print('No Model Found!!!')
        return None
    model_list = natsorted(model_list)

    return model_list[-1]


class DateDataset(torchtext.data.Dataset):
    def __init__(self, data_filename, fields, **kwargs):
        # fields: [('id', ID), ('text', TEXT), ('label', LABEL)]
        examples = []
        train_data = pd.read_csv(data_filename, header=None, delimiter='_' )
        inputs, targets = train_data[0].tolist(), train_data[1].tolist()

        for line in zip(inputs,targets):
        
            examples.append(torchtext.data.Example.fromlist(line, fields))
        
        
        super(DateDataset, self).__init__(examples, fields, **kwargs)

#TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=True,include_lengths=False,lower=True)
TEXT = torchtext.data.Field(sequential=True, tokenize=list,batch_first=False,include_lengths=False,init_token='<sos>',eos_token='<eos>',lower=True)  # src, target 모두에 sos,eos가 붇는다.
train_data = DateDataset('date.txt', [('src',TEXT),('target',TEXT)])


# for i, d in enumerate(train_data):
#     print(i, d.src,d.target)
#     if i>=2: break



TEXT.build_vocab(train_data, min_freq=1, max_size=100)   # build_vocab 단계를 거처야, 단어가 숫자로 mapping된다.
vocab_size = len(TEXT.vocab)
print('단어 집합의 크기 : {}'.format(vocab_size))
print(TEXT.vocab.stoi)  # 단어 dict   ----> inference를 대비해서 저장해 두어야 함.


batch_size = 32
train_loader = torchtext.data.Iterator(dataset=train_data, batch_size = batch_size,shuffle=True)

# for i, d in enumerate(train_loader):
#     print(i,d.src.shape, d.target.shape, d.src, d.target)   # d.text[0], d.text[1] ----> Field에서 include_lengths=True로 설정.
#     if i>=2: break


#################
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        hidden_dim = 64
        decoder_length = 11
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_model = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=128,dropout=0.3)
        self.out_linear = nn.Linear(hidden_dim, vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.tgt_mask = self.transformer_model.generate_square_subsequent_mask(decoder_length).to(device)
        
    def forward(self,ecoder_inputs, decoder_inputs):
        x = self.embedding(ecoder_inputs)  # (29,N)  ---> (29,N,hidden_dim)
        y = self.embedding(decoder_inputs) # (11,N)  ---> (11,N,hidden_dim)
        
        outputs = self.transformer_model(x,y,src_mask=None, tgt_mask=self.tgt_mask)  # tgt_mask필요
        outputs = self.out_linear(outputs)
        return outputs
    
    

def train():   

    model = Transformer().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    num_epoch = 10
    start_epoch=0
    saved_model = get_latest_model(model_save_dir)
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        start_epoch = int(os.path.basename(saved_model).split('-')[-1].split('.')[0])  # 특수한 상황....
    model.train()
    
    s_time = time.time()
    step_count = 0
    for epoch in range(start_epoch, num_epoch):
        for i, d in enumerate(train_loader):
            step_count += 1
            optimizer.zero_grad()
            target = d.target.to(device)  #(T,N)
            encoder_inputs = d.src[1:-1,:].to(device)
            decoder_inputs = target[:-1,:]
            outputs = model(encoder_inputs,decoder_inputs)  # (T,N,D)
            
            loss = model.loss_fn(outputs.permute(1,2,0), target[1:,:].T)  # (T,N,D)  ---> CrossEntropyLoss에는 (N,D,T)를 넘겨야 한다. target에는 (N,T)
            loss.backward()
            optimizer.step()
            
            if step_count % 200 == 0:
                print('epoch: {}, setp: {}, loss: {}, elapse: {}'.format(epoch, step_count, loss.item(), int(time.time()-s_time)))
                predict = torch.argmax(outputs,-1).detach() # predict: (T,N), target: (T,N)
                print('accuracy: ', np.mean((target[1:,:]==predict).prod(dim=0).to('cpu').numpy()))
                print(''.join([TEXT.vocab.itos[x] for x in encoder_inputs[:,0].to('cpu').numpy()]),'|', ''.join([TEXT.vocab.itos[x] for x in target[:,0].to('cpu').numpy()]),'-->',''.join([TEXT.vocab.itos[x] for x in predict[:,0].to('cpu').numpy()]))
            
        if (epoch+1)%5==0:         
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'epoch-{}.pth'.format(epoch)))  # weights만 저장


def test():
    model = Transformer().to(device)

    saved_model = get_latest_model(model_save_dir)
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location = device))
        print('Model loaded!!!', saved_model)
    else:
        exit()
        
        
    test_data = ['TUESDAY, SEPTEMBER 10, 1991','December 1, 2013']
    
    
    
    
    

if __name__ == '__main__':
    #train()
    test()