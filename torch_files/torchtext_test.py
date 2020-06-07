# coding: utf-8

'''
tf.keras.preprocessing vs torchtext


https://wikidocs.net/64517


영어의 경우 토큰화를 사용하는 도구로서 대표적으로 spaCy와 NLTK가 있습니다

1. spacy: pip install spacy
관리자 권한: > python -m spacy download en   ----> 이걸 해주어여 sapcy.load('en')을 통해 tokenize할  수 있다.    python -m spacy download de

pip install torchtext


1. data file을 pandas로 읽어들인 후, preprocessing 작업을 한 후, 필요한 column만으로 csv파일을 만든다.
2. data.Field를 만든다.
3. data.TabularDataset을 만든다.



'''

import pandas as pd
from konlpy.tag import Mecab
import torchtext

from torchtext.data import Field
import spacy  # 
import nltk  # nltk.download('punkt')
from nltk.tokenize import word_tokenize

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
