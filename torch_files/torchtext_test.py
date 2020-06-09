# coding: utf-8

'''
tf.keras.preprocessing vs torchtext


https://wikidocs.net/64517   ---> NLP 책


영어의 경우 토큰화를 사용하는 도구로서 대표적으로 spaCy와 NLTK가 있습니다

1. spacy: pip install spacy
관리자 권한: > python -m spacy download en   ----> 이걸 해주어여 sapcy.load('en')을 통해 tokenize할  수 있다.    python -m spacy download de

pip install torchtext


1. data file을 pandas로 읽어들인 후, preprocessing 작업을 한 후, 필요한 column만으로 csv파일을 만든다.
2. data.Field를 만든다.   ---> build_vocab    ---> 0: unknown, 1: pad, ....
3. data.TabularDataset을 만든다.

여러 파일로 분리되어 있는 경우: TabularDataset.splits
train_data = TabularDataset.splits(path='./data/',
                    train='train_path',
                    valid='valid_path',
                    test='test_path',
                    format='tsv', 
                    fields=[('text', TEXT), ('label', LABEL)])


하나의 파일인 경우: 읽은 후, spliti할 수 있다.
train_data = TabularDataset(path='./data/examples.tsv', 
                format='tsv', 
                fields=[('text', TEXT), ('label', LABEL)])
train_data.split(...)


4. Iterator, BucketIterator를 이용해서 DataLoader를 만든다.
========================
** 보통 torch.utils.data.Dataset을 상속받아 custom Dataset을 만든다.
여기서는 torchtext.data.Dataset을 상속받아 custom Dataset을 만들 수 있다.

========================
imdb.tar.gz  ---> 1줄로된 파일이 모여있다. 12,500 x 4 = 5만 ----> 전체를 하나의 csv로 모아 놓을 것이   IMDb_Reviews.csv (5만 lines)   1은 긍정, 0은 부정

========================





'''

import pandas as pd
from konlpy.tag import Mecab
import torchtext
import spacy  # 
import nltk  # nltk.download('punkt')
from nltk.tokenize import word_tokenize
from konlpy.tag import Kkma
import random

def test1():
    en_text = "The Dogs Run back corner near spare bedrooms?"
    
    
    spacy_en = spacy.load('en')
    
    tokens = [tok.text for tok in spacy_en.tokenizer(en_text)]
    print('spacy:', tokens)
    
    print('nltk:', word_tokenize(en_text))
    print('공백분리:', en_text.split())
    
    
    data = pd.read_table('D:/BookCodes/tensorflow-ml-nlp-master/4.TEXT_CLASSIFICATION/data_in/ratings.txt')
    print(data.head(10))
    
    
    print('전체 샘플의 수 : {}'.format(len(data)))
    sample_data = data[:100] # 임의로 100개만 저장
    
    sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    print(sample_data.head(10))
    
    sample_data.to_csv("hccho.csv",columns =['document', 'label'], index=False)






def test2_english():
    
    if False:  # 한번만 하면 됨   ---> csv파일 만들기

        df = pd.read_csv('D:/hccho/CommonDataset/IMDb_Reviews.csv', encoding='latin1')
        print(df.head())
        
        print('전체 샘플의 개수 : {}'.format(len(df)))
        
        train_df = df[:25000]
        test_df = df[25000:]
        
        train_df.to_csv("imdb_train_data.csv", index=False)
        test_df.to_csv("imdb_test_data.csv", index=False)

    TEXT = torchtext.data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=str.split,
                      lower=True,
                      batch_first=True,
                      fix_length=100)   # fix_length를 넘어가면, pad(1)이 붙는다.
    
    LABEL = torchtext.data.Field(sequential=False,
                       use_vocab=False,
                       batch_first=False,
                       is_target=True)

    train_data, test_data = torchtext.data.TabularDataset.splits(
            path='.', train='imdb_train_data.csv', test='imdb_test_data.csv', format='csv',
            fields=[('text', TEXT), ('label', LABEL)], skip_header=True)


    print('훈련 샘플의 개수 : {}'.format(len(train_data)))
    print('테스트 샘플의 개수 : {}'.format(len(test_data)))
    
    
    
    print(vars(train_data[0]))  # python 내장함수 vars() ---> dict로 변환해준다.


    # 필드 구성 확인.
    print(train_data.fields.items())

    
    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
    print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))   # {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, ....}
    print(TEXT.vocab.stoi)
    
    # DataLoader 만들기
    batch_size = 2
    train_loader = torchtext.data.Iterator(dataset=train_data, batch_size = batch_size)
    test_loader = torchtext.data.Iterator(dataset=test_data, batch_size = batch_size)
    
    
    for i, d in enumerate(train_loader):
        print(i,d.text, d.label)
        if i>=2: break
    
    
    
    print('='*20)
    for i in range(2):
        batch = next(iter(train_loader))
        print(batch.text, batch.label)


def test2_kor():
    train_df = pd.read_table('D:/BookCodes/tensorflow-ml-nlp-master/4.TEXT_CLASSIFICATION/data_in/ratings_train.txt')
    test_df = pd.read_table('D:/BookCodes/tensorflow-ml-nlp-master/4.TEXT_CLASSIFICATION/data_in/ratings_test.txt')  

    print(train_df.head())
 
    print('훈련 데이터 샘플의 개수 : {}'.format(len(train_df)))
    print('테스트 데이터 샘플의 개수 : {}'.format(len(test_df)))

    tokenizer = Kkma()  # .morphs() ---> 너무 느리다.

    ID = torchtext.data.Field(sequential = False,
                    use_vocab = False) # 실제 사용은 하지 않을 예정   ---> txt파일에 ID column이 있어서...
    TEXT = torchtext.data.Field(sequential=True,include_lengths=True,
                      use_vocab=True,
                      tokenize=tokenizer.morphs, # 토크나이저로는 Kkma 사용.
                      lower=True,
                      batch_first=True,  # batch_firt=True ---> (N,fix_length)   False ---> (fix_length,N)
                      fix_length=20)
    
    LABEL = torchtext.data.Field(sequential=False,
                       use_vocab=False,
                       is_target=True)


    if False:
        train_data, test_data = torchtext.data.TabularDataset.splits(
                path='D:/BookCodes/tensorflow-ml-nlp-master/4.TEXT_CLASSIFICATION/data_in', train='ratings_train.txt', 
                test='ratings_test.txt', format='tsv',
                fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)
    else:
        train_data = torchtext.data.TabularDataset(
                path='D:/BookCodes/tensorflow-ml-nlp-master/4.TEXT_CLASSIFICATION/data_in/ratings_train_small.txt', 
                format='tsv', fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)     
        
        train_data, test_data = train_data.split(split_ratio=0.7,random_state=random.seed(100)) 

        

    print('훈련 샘플의 개수 : {}'.format(len(train_data)))
    print('테스트 샘플의 개수 : {}'.format(len(test_data)))
    
    print(vars(train_data[0]))
    
    
    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
    print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
    print(TEXT.vocab.stoi)
    
    
    # DataLoader 만들기
    batch_size = 2
    
    if False:
        train_loader = torchtext.data.Iterator(dataset=train_data, batch_size = batch_size,shuffle=True)  # shuffle=True epoch 사이에 shuffle 여부.
        test_loader = torchtext.data.Iterator(dataset=test_data, batch_size = batch_size)
        

    else:
        # data.BucketIterator ----> padding이 최소화 되도록 한다.
        train_loader, test_loader = torchtext.data.BucketIterator.splits((train_data, test_data),
                                                                       batch_size=batch_size,
                                                                       device='cpu',
                                                                       sort_key=lambda x: len(x.text))
    
    for i, d in enumerate(train_loader):
        print(i,d.text, d.label)   # d.text[0], d.text[1] ----> Field에서 include_lengths=True로 설정.
        if i>=2: break
    
    
    
    print('='*20)
    for i in range(2):
        batch = next(iter(train_loader))
        print(batch.text, batch.label)    






if __name__ == '__main__':
    #test1()
    #test2_english()
    test2_kor()

    print('Done')








