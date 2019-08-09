# coding: utf-8

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import re
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.text import Tokenizer


text_sequences = [[2,3,4],[1,5,6,8,9]]
train_inputs = pad_sequences(text_sequences, maxlen=10, padding='post')  # maxlen보다 길면 잘라 준다.
"""
array([[2, 3, 4, 0, 0, 0, 0, 0, 0, 0],
       [1, 5, 6, 8, 9, 0, 0, 0, 0, 0]])

"""


###############################################
###############################################

FILTERS = "([~.,!?\"':;)(])"
CHANGE_FILTER = re.compile(FILTERS)

sequence = ['안녕? 뭐 먹을까?', '반가워~ 또 봐요?']
sequence = [re.sub(CHANGE_FILTER, "", s) for s in sequence]  # re.sub는 1개씩만 처리
"""
['안녕 뭐 먹을까', '반가워 또 봐요']
"""

###############################################
# 형태소 분석 + tokenize
###############################################


sentences= ['안녕 어제는 뭐 했어?', '안녕, 반가워? 어제도 봤는데...']
morph_analyzer = Okt()
sentences = [morph_analyzer.morphs(s) for s in sentences] # 1개 문장씩 처리. --> [['안녕', '어제', '는', '뭐', '했어', '?'], ['안녕', ',', '반가워', '?', '어제', '도', '봤는데', '...']]
sentences_merged =[" ".join(s) for s in sentences]   # ['안녕 어제 는 뭐 했어 ?', '안녕 , 반가워 ? 어제 도 봤는데 ...']


tokenizer = Tokenizer(lower=True)  # filter 기능 있음. ? , ... 등 제외
tokenizer.fit_on_texts(sentences_merged)
print(tokenizer.word_index)  # {'안녕': 1, '어제는': 2, '뭐': 3, '했어': 4, '반가워': 5, '또': 6, '봐요': 7}

sentences_id = tokenizer.texts_to_sequences(sentences_merged) # [[1, 2, 3, 4, 5], [1, 6, 2, 7, 8]]

###############################################
# 단어 빈도
###############################################

import itertools
import nltk
x = ["This is a foobar-like sentence.","Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."]
y= [nltk.word_tokenize(sent) for sent in x]  # [['This', 'is', 'a', 'foobar-like', 'sentence', '.'], ['Good',  'muffins',  'cost',  '$',  '3.88',  'in',  'New',  'York',  '.',  'Please',  'buy',  'me',  'two',  'of',  'them',  '.',  'Thanks',  '.']]
w = list(itertools.chain(*y))  # list 하나로 합치기
z = nltk.FreqDist(w)    # zz=collections.Counter(w) 같은 결과
vocab = z.most_common(5) # zz.most_common(5)
