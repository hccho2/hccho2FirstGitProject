
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
print(tf.__version__)



embedding_dim =5
vocab_size =3


init = np.random.randn(vocab_size,embedding_dim)
print('init: ',init)
#embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True,name='my_embedding') 
embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=Constant(init),trainable=True) 
print('embedding.trainable_variables', embedding.trainable_variables)


input = np.array([[1,0,2,2,0,1],[1,1,1,2,2,0]])

output = embedding(input)




print('='*10)
print(input,output)
print('done')


model = tf.keras.Sequential()
model.add(embedding)
print('trainable: ',model.trainable_variables)


