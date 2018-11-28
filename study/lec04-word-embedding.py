# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import nltk,re
import csv, itertools
import matplotlib.pyplot as plt
import pickle
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(threshold=np.nan)


def generate_data():
    vocabulary_size = 250
    unknown_token = "UNKNOWN_TOKEN"
    word_dim=3
    
    print( "Reading CSV file...")
    with open('raw_sentences.txt', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True,delimiter='\n')
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = [' '.join(x.split()) for x in sentences]  # ['no , he says now .', 'and what did he do ?',, ...]
    print( "Parsed %d sentences." % (len(sentences)))
    
    
    # Tokenize the sentences into words(문장을 각각의 단어로 분할)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]  # [['no', ',', 'he', 'says', 'now', '.'],  ['and', 'what', 'did', 'he', 'do', '?'], ...]
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print ("Found %d unique words tokens." % len(word_freq.items()))
    
    
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)  # vocab <--- list  [('.', 80974), ('it', 29200), (',', 24583), ...]
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    print ("Using vocabulary size %d." % vocabulary_size)
    print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    
    
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
    print ("\nExample sentence: '%s'" % sentences[0])
    print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
    
    
    indexed_sentences = [[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences]
    
    all_data = []
    for sen in indexed_sentences:
        if len(sen) >= word_dim+1:
            for i in range(len(sen)-word_dim):
                all_data.append(sen[i:i+word_dim+1])   
    all_data = np.array(all_data,dtype=np.int16)
    np.random.shuffle(all_data)
    data ={'data': all_data,'word_to_index': word_to_index, 'index_to_word':index_to_word, 'vocab':vocab}
    with open('data.pickle', 'wb') as outfile:
        pickle.dump(data, outfile)
generate_data()
with open('data.pickle','rb') as f:
    data1=pickle.load(f)