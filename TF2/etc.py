import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def bernoulli_sample_test():
    # tfa.seq2seq.sampler.bernoulli_sample   --> probs 또는 logits를 보고 단순 sampling만 해준다.
    batch_size = 5
    
    probs = tf.random.uniform(shape=[batch_size])
    logits = tf.random.normal(shape=(batch_size,7))
    
    # tfa.seq2seq.sampler.bernoulli_sample    ---> probs 또는 logits가 넘어가면 된다. logits는 내부에서 sigmoid를 취해 확률이 된다.
    # output은 항상 0 또는 1로 되어 있다.
    x = tfa.seq2seq.sampler.bernoulli_sample(probs = probs)
    y = tfa.seq2seq.sampler.bernoulli_sample(logits = logits)  # logits가 (2,5) shape이면 2x5개의 sampling이 이루어 진다.
    
    
    z = tfa.seq2seq.sampler.categorical_sample(logits)  # logits: (batch_size, N)  ---> batch_size만큼 sampling이 된다.
    
    
    
    print(probs)
    print(logits)
    print(x)
    print(y)
    print(z)


if __name__ == '__main__':
    bernoulli_sample_test()




