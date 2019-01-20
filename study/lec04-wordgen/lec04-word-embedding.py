# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import nltk,re
import csv, itertools
import matplotlib.pyplot as plt
import pickle,os,time
from datetime import datetime
from glob import glob
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(threshold=np.nan)

unknown_token = "UNKNOWN_TOKEN"
class TextData():
    def __init__(self,txt_filename,pickle_filename=None,vocabulary_size=250,word_dim=3):
        self.txt_filename = txt_filename
        self.vocabulary_size= vocabulary_size
        self.word_dim = word_dim
        if pickle_filename is None:
            self.pickle_filename = os.path.splitext(txt_filename)[0] +'.pickle'
        else:
            self.pickle_filename = pickle_filename
   
        if not os.path.exists(self.pickle_filename): 
            self.generate_data()
        else:
            with open(self.pickle_filename,'rb') as f:
                self.data=pickle.load(f)                 
        self.idx = 0
        print('data size: ', self.data['data'].shape)
    def generate_data(self):
        print('Generating pickle file...')
        print( "Reading CSV file...")
        with open(self.txt_filename, 'r') as f:
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
        vocab = word_freq.most_common(self.vocabulary_size-1)  # vocab <--- list  [('.', 80974), ('it', 29200), (',', 24583), ...]
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        
        print ("Using vocabulary size %d." % self.vocabulary_size)
        print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
        
        
        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
        
        print ("\nExample sentence: '%s'" % sentences[0])
        print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
        
        
        indexed_sentences = [[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences]
        
        all_data = []
        for sen in indexed_sentences:
            if len(sen) >= self.word_dim+1:
                for i in range(len(sen)-self.word_dim):
                    all_data.append(sen[i:i+self.word_dim+1])   
        all_data = np.array(all_data,dtype=np.int32)
        np.random.shuffle(all_data)
        self.data ={'data': all_data,'word_to_index': word_to_index, 'index_to_word':index_to_word, 'vocab':vocab}
        with open(self.pickle_filename, 'wb') as outfile:
            pickle.dump(self.data, outfile)

    def next_batch(self,batch_size):
        # self.idx는 이번에 읽어야 될 위치
        if self.idx + batch_size <= len(self.data['data']):
            self.idx += batch_size
            return self.data['data'][self.idx-batch_size:self.idx]
        else:
            np.random.shuffle(self.data['data'])
            self.idx = batch_size
            return self.data['data'][:batch_size]

    def convert_to_words(self,words):
        w = []
        for x in words:
            w.append(self.data['index_to_word'][x])
        return ' '.join(w)
    def convert_to_words_batch(self,words):
        w = []
        for x in words:
            w.append(self.convert_to_words(x))
        return w
    
    def convert_to_indexs(self,str):
        result = []
        words = str.lower().split()
        for w in words:
            result.append(self.data['word_to_index'][w])
        return result    
    
class Word_Gen():
    def __init__(self,dataset,word_embedding_dim=50,name='model'):
        # dataset과 모델을 분리하고, hyper parameter를 따로 두어야 하는데, 결합하다 보니, inference 단계에서도 dataset을 받아야 하는 문제가 있음.
        self.dataset = dataset
        self.name=name
        self.word_embedding_dim = word_embedding_dim
        self.build()

    
    def build(self):
        with tf.variable_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.x = tf.placeholder(tf.int32,shape=[None,self.dataset.word_dim])
            self.y = tf.placeholder(tf.int32,shape=[None,1])
            
            self.embedding_table = tf.get_variable('word_embedding',shape=[self.dataset.vocabulary_size,self.word_embedding_dim],dtype=tf.float32)
            embedding = tf.reshape(tf.nn.embedding_lookup(self.embedding_table,self.x),shape=(-1,self.word_embedding_dim*self.dataset.word_dim))
            
            layer1 = tf.layers.dense(embedding,units=200,activation=tf.nn.relu)
            layer2 = tf.layers.dense(layer1,units=200,activation=tf.nn.relu)
            logits = tf.layers.dense(layer2,units=self.dataset.vocabulary_size,activation=None)
            self.predict = tf.nn.softmax(logits)
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y,self.dataset.vocabulary_size),logits=logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss,global_step=self.global_step)
                        
    def display_nearest_words(self,word_ind,k=1):
        # word_ind와 가장 가까운 k개를 return한다.
        word_vec = tf.nn.embedding_lookup(self.embedding_table,word_ind)
        word_vec_all = tf.nn.embedding_lookup(self.embedding_table,tf.range(self.dataset.vocabulary_size))
        
        distance = tf.reduce_sum(tf.square(word_vec_all-word_vec),axis=-1)               
        return tf.nn.top_k(-distance,k)  # values, indices



def get_most_recent_checkpoint(checkpoint_dir,model_name):
    checkpoint_paths = [path for path in glob("{}/{}-*.data-*".format(checkpoint_dir,model_name))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]
    if len(idxes) > 0:
        max_idx = max(idxes)
        lastest_checkpoint = os.path.join(checkpoint_dir, "{}-{}".format(model_name,max_idx))
    
        print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
        return lastest_checkpoint
    else:
        return None
def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def train():
    logroot = './logdir'
    
    #logdir = None
    logdir = './logdir/2018-11-29_17-44-41'
    
    
    epochs = 1000
    batch_size = 100
    SAVE_EVERY = 20
        
    #dataset = TextData('raw_sentences-simple.txt')
    dataset = TextData('raw_sentences.txt')
    
    """
    x = dataset.next_batch(5)
    y = dataset.convert_to_words(x[0])
    print(y)
    y = dataset.convert_to_words_batch(x)
    print(y)
    """
    

    my_model = Word_Gen(dataset,word_embedding_dim=50,name='my_model')

    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        if logdir is None:
            logdir = os.path.join(logroot, get_time())
            os.makedirs(logdir)
        else:
            restore_path = get_most_recent_checkpoint(logdir,my_model.name)
            if restore_path:
                saver.restore(sess, restore_path)       
        
        checkpoint_path = os.path.join(logdir, my_model.name)
        s = time.time()
        n_iter = len(my_model.dataset.data['data']) // batch_size
        initial_epoch = sess.run(my_model.global_step) // n_iter
        for step in range(initial_epoch+1,epochs+1):
            for _ in range(n_iter):
                next_batch = my_model.dataset.next_batch(batch_size)
                sess.run(my_model.optimizer,feed_dict={my_model.x: next_batch[:,:my_model.dataset.word_dim], my_model.y: next_batch[:, -1:]})
        
            loss_ = sess.run(my_model.loss,feed_dict={my_model.x: next_batch[:,:my_model.dataset.word_dim], my_model.y: next_batch[:, -1:]})
            print('epoch: {}, loss = {:.4f}, elapsed = {:.2f}'.format(step,loss_,time.time()-s))       
        
            if step % SAVE_EVERY == 0:
                print('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                saver.save(sess, checkpoint_path, global_step=step)


def generate():
    logdir = './logdir/2018-11-29_17-44-41'
    vocabulary_size=250
    
    dataset = TextData('raw_sentences.txt',vocabulary_size=250) # dataset내에 있는 함수 때문에....
    my_model = Word_Gen(dataset,word_embedding_dim=50,name='my_model')

    saver = tf.train.Saver()
    
    
    max_step=20
    input_string = 'she has a'
    input_index = dataset.convert_to_indexs(input_string)
    
    stop_token = '. ?'
    stop_token = dataset.convert_to_indexs(stop_token)    
    
    
    with tf.Session() as sess:
        restore_path = get_most_recent_checkpoint(logdir,my_model.name)
        saver.restore(sess, restore_path)
        
        i = 0
        while True:
            if input_index[-1] in stop_token or i>=max_step:
                break
            result = sess.run(my_model.predict,feed_dict={my_model.x: [input_index[-3:]]})
            choice = np.random.choice(vocabulary_size,p=result[0])
            input_index.append(choice)
            i = i+1
        print(dataset.convert_to_words(input_index))        
def display_nearest_words():
    logdir = './logdir/2018-11-29_17-44-41'
    vocabulary_size = 250
    word_dim = 3
    embedding_dim = 50


    
    word = 'can'
    top = 5

    
    dataset = TextData('raw_sentences.txt',vocabulary_size=250) # dataset내에 있는 함수 때문에....
    my_model = Word_Gen(dataset,word_embedding_dim=50,name='my_model')
    
    saver = tf.train.Saver()
    

    word_ind = dataset.data['word_to_index'][word]

    with tf.Session() as sess:
        
        restore_path = get_most_recent_checkpoint(logdir,my_model.name)
        if restore_path:
            saver.restore(sess, restore_path)     

            
            
        top_k = my_model.display_nearest_words(word_ind,top)
        result = sess.run(top_k)
        print(word,": ", dataset.convert_to_words(result.indices))
        print("distance: ", -result.values)        
if __name__ == "__main__":    
    s=time.time()
    #train()
    #generate()
    display_nearest_words()
    
    e=time.time()
    
    print('done: {} sec'.format(e-s))