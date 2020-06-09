# coding: utf-8
'''
easy_seq2seq-master에 있는 코드를 수정한 것

train.dec.ids20000  파일은 없으면 자동으로 만든다.


'''
import numpy as np
import random,os,re,sys
import tensorflow as tf
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")




_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


class SeqDataFeeder():
    def __init__(self,data,bucket):
        self.data = data
        self.bucket = bucket
        
        train_bucket_sizes = [len(self.data[b]) for b in range(len(self.bucket))]
        train_total_size = float(sum(train_bucket_sizes))
        self.train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size  for i in range(len(train_bucket_sizes))]    


    def get_batch(self,data,batch_size):
        """Get a random batch of data from the specified bucket, prepare for step.
    
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
    
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
    
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in range(len(self.train_buckets_scale)) if self.train_buckets_scale[i] > random_number_01])
        
        
        
        encoder_size, decoder_size = _buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
    
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
    
            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))
    
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)
    
    
        return encoder_inputs, decoder_inputs

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        if isinstance(space_separated_fragment, str):
            word = str.encode(space_separated_fragment)
        else:
            word = space_separated_fragment
        words.extend(re.split(_WORD_SPLIT, word))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,tokenizer=None, normalize_digits=True):

    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            print('>> Full Vocabulary Size :',len(vocab_list))
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):

    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

        # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(enc_vocab_path, train_enc, enc_vocabulary_size, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec, dec_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)
def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

if __name__ =='__main__':
    # 정수로 변환
    enc_train, dec_train, enc_dev, dec_dev, _, _ = prepare_custom_data('./data',
                                                                                  './data/train.enc',
                                                                                  './data/train.dec',
                                                                                  './data/test.enc',
                                                                                  './data/test.dec',
                                                                                  20000,
                                                                                  20000)
    
    train_set = read_data(enc_train, dec_train)
    
    
    data = SeqDataFeeder(train_set,_buckets)
    
    
    encoder_inputs, decoder_inputs = data.get_batch(train_set, 2)
    
    
    
    
    enc_vocab_path = os.path.join('./data',"vocab%d.enc" % 20000)
    dec_vocab_path = os.path.join('./data',"vocab%d.dec" % 20000)
    enc_vocab, rev_enc_vocab = initialize_vocabulary(enc_vocab_path)
    dec_vocab, rev_dec_vocab = initialize_vocabulary(dec_vocab_path)
    
    
    print(encoder_inputs[0])
    print(" ".join([tf.compat.as_str(rev_enc_vocab[output]) for output in encoder_inputs[0]]))
    
    
    print(decoder_inputs[0])
    print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in decoder_inputs[0]]))
    
    ##########
