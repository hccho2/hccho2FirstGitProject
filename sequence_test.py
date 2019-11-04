# coding: utf-8

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import re
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.text import Tokenizer

'''
X = ['sa','bb']
np.savetxt('aa.txt', X, delimiter=" ", fmt="%s") 
'''

def test1():

    text_sequences = [[2,3,4],[1,5,6,8,9]]
    train_inputs = pad_sequences(text_sequences, maxlen=10, padding='post')  # maxlen보다 길면 잘라 준다.
    """
    array([[2, 3, 4, 0, 0, 0, 0, 0, 0, 0],
           [1, 5, 6, 8, 9, 0, 0, 0, 0, 0]])
    
    """


###############################################
###############################################
def test2():
    FILTERS = "([~.,!?\"':;)(])"
    CHANGE_FILTER = re.compile(FILTERS)
    
    sequence = ['안녕? 뭐 먹을까?', '반가워~ 또 봐요?']
    sequence = [re.sub(CHANGE_FILTER, "", s) for s in sequence]  # re.sub는 1개씩만 처리
    """
    ['안녕 뭐 먹을까', '반가워 또 봐요']
"""


###############################################
# 영문 + tokenize
###############################################
def test3():
    sentences= ['and twenty men could hold it with spears and arrows?', 'but, all my dreams violated this law']
    
    tokenizer = Tokenizer(lower=True,char_level=False)  # filter 기능 있음. ? , ... 등 제외
    tokenizer.fit_on_texts(sentences)
    print(tokenizer.word_index)  # {'and': 1, 'twenty': 2, 'men': 3, 'could': 4, 'hold': 5, 'it': 6, 'with': 7, 'spears': 8, 'arrows': 9, 'but': 10, 'all': 11, 'my': 12, 'dreams': 13, 'violated': 14, 'this': 15, 'law': 16}
    
    sentences_id = tokenizer.texts_to_sequences(sentences) #[[1, 2, 3, 4, 5, 6, 7, 8, 1, 9], [10, 11, 12, 13, 14, 15, 16]]
    
    
    #####
    sentences_merged = [s.split(' ') for s in sentences]   # [['and', 'twenty','men','could','hold','it','with','spears','and','arrows'], ['but', 'all', 'my', 'dreams', 'violated', 'this', 'law']]
    sentences_merged = [[c for c in x.strip().lower()] for x in sentences] # alphabet 단위. strip() 양 끝의 공백 제거.
    ####


###############################################
# 형태소 분석 + tokenize
###############################################

def test4():
    sentences= ['안녕 어제는 뭐 했어?', '안녕, 반가워? 어제도 봤는데...']
    morph_analyzer = Okt()
    sentences = [morph_analyzer.morphs(s) for s in sentences] # 1개 문장씩 처리. --> [['안녕', '어제', '는', '뭐', '했어', '?'], ['안녕', ',', '반가워', '?', '어제', '도', '봤는데', '...']]
    sentences_merged =[" ".join(s) for s in sentences]   # ['안녕 어제 는 뭐 했어 ?', '안녕 , 반가워 ? 어제 도 봤는데 ...']
    
    
    tokenizer = Tokenizer(lower=True)  # filter 기능 있음. ? , ... 등 제외
    tokenizer.fit_on_texts(sentences_merged)
    print(tokenizer.word_index)  # {'안녕': 1, '어제는': 2, '뭐': 3, '했어': 4, '반가워': 5, '또': 6, '봐요': 7}
    
    sentences_id = tokenizer.texts_to_sequences(sentences_merged) # [[1, 2, 3, 4, 5], [1, 6, 2, 7, 8]]



###############################################
# 단어 빈도--> 일부를 unknown으로 처리하려면, nltk가 낫다.
###############################################
def test5():
    import itertools
    import nltk
    x = ["This is a foobar-like sentence.","Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."]
    y= [nltk.word_tokenize(sent) for sent in x]  # [['This', 'is', 'a', 'foobar-like', 'sentence', '.'], ['Good',  'muffins',  'cost',  '$',  '3.88',  'in',  'New',  'York',  '.',  'Please',  'buy',  'me',  'two',  'of',  'them',  '.',  'Thanks',  '.']]
    w = list(itertools.chain(*y))  # list 하나로 합치기
    z = nltk.FreqDist(w)    # zz=collections.Counter(w) 같은 결과
    vocab = z.most_common(5) # zz.most_common(5)


def test6():
    filename = 'D:/OCR/ocr_kor-master/data/generator/TextRecognitionDataGenerator/dicts/ko.txt'
    ko_dic = open(filename, encoding='utf-8')
    words = ko_dic.readlines()  # list: ['가게\n', '가격\n', '가구\n', ....]

    tokenizer = Tokenizer(lower=True,char_level=True)  # filter 기능 있음. ? , ... 등 제외
    tokenizer.fit_on_texts(words)
    kor_dic_char = ''.join(sorted(tokenizer.word_index.keys()))
    print(len(kor_dic_char), kor_dic_char[:10])
    
    print('Done')
    
if __name__ == '__main__':
    test6()





