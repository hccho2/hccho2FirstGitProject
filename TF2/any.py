# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.seq2seq import Sampler



# BeamSearchDecoder
vocab_size = 6
SOS_token = 0
EOS_token = 5

# x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
# y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
# print("data shape: ", x_data.shape)


index_to_char = {SOS_token: '<S>', 1: 'h', 2: 'e', 3: 'l', 4: 'o', EOS_token: '<E>'}
x_data = np.array([[SOS_token, 1, 2, 3, 3, 4]], dtype=np.int32)
y_data = np.array([[1, 2, 3, 3, 4,EOS_token]],dtype=np.int32)



output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =7

seq_length = x_data.shape[1]
embedding_dim = 8


embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True) 
##### embedding.weights, embedding.trainable_variables, embedding.trainable_weights --> 모두 같은 결과 


target = tf.convert_to_tensor(y_data)

# Decoder
method = 1
if method==1:
    # single layer RNN
    decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)
    # decoder init state:
    
    #init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
    init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
    
else:
    # multi layer RNN
    decoder_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_dim),tf.keras.layers.LSTMCell(2*hidden_dim)])
    init_state = decoder_cell.get_initial_state(inputs=tf.zeros_like(x_data,dtype=tf.float32))  # inputs의 batch_size만 참조하기 때문에


projection_layer = tf.keras.layers.Dense(output_dim)


# train용 Sampler로 TrainingSampler 또는 ScheduledEmbeddingTrainingSampler 선택.
sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()
#sampler = tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler(sampling_probability=0.2)

decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)



optimizer = tf.keras.optimizers.Adam(lr=0.01)

inputs = tf.keras.Input(shape=(seq_length))

embedded = embedding(inputs)
embedded = tf.reshape(embedded,[batch_size,seq_length,embedding_dim])


if isinstance(sampler, tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler):
    outputs, last_state, last_sequence_lengths = decoder(embedded,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True,embedding=embedding.weights)
else: outputs, last_state, last_sequence_lengths = decoder(embedded,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)

outputs, last_state, last_sequence_lengths = decoder(embedded,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)

model = tf.keras.Model(inputs,outputs.rnn_output)
print(model.summary())


train_mode = False

if train_mode:

    for step in range(500):
        with tf.GradientTape() as tape:
    
            logits = model(x_data)
            
            weights = tf.ones(shape=[batch_size,seq_length])
            loss = tfa.seq2seq.sequence_loss(logits,target,weights)
        
        trainable_variables = embedding.trainable_variables + decoder.trainable_variables   # 매번 update되어야 한다.
        grads = tape.gradient(loss,trainable_variables)
        optimizer.apply_gradients(zip(grads,trainable_variables))
        
        if step%10==0:
            print(step, loss.numpy())
    
    tf.saved_model.save(model,'./saved_model/')  # model구조까지 저장

else:
    model = tf.saved_model.load('./saved_model/')
    
    sample_batch_size = 5
    
    decoder_type = 1
    if decoder_type==1:
        # GreedyEmbeddingSampler or SampleEmbeddingSampler()
        
        # sampler 선택 가능.
        sampler = tfa.seq2seq.GreedyEmbeddingSampler()  # alias ---> sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler
        #sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_fn=lambda ids: tf.nn.embedding_lookup(embedding.weights, ids)) # embedding_fn을 넘겨줄 수도 ㅣ있다.
        #sampler = tfa.seq2seq.SampleEmbeddingSampler()
        
        decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer,maximum_iterations=seq_length)
        if method==1:
            # single layer
            init_state = decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size, dtype=tf.float32)
        else:
            # multi layer
            init_state = decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size,hidden_dim],dtype=tf.float32))
        
    else:
        # Beam Search
        beam_width=2
        decoder = tfa.seq2seq.BeamSearchDecoder(decoder_cell,beam_width,output_layer=projection_layer,maximum_iterations=seq_length)
        
        # 2가지 방법은 같은 결과를 준다.
        if method==1:
            #init_state = decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size*beam_width, dtype=tf.float32)
            init_state = tfa.seq2seq.tile_batch(decoder_cell.get_initial_state(inputs=None, batch_size=sample_batch_size, dtype=tf.float32),multiplier=beam_width)
        else:
            #init_state = decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size*beam_width,hidden_dim],dtype=tf.float32))
            init_state = tfa.seq2seq.tile_batch(decoder_cell.get_initial_state(inputs=tf.zeros([sample_batch_size,hidden_dim],dtype=tf.float32)),multiplier=beam_width)
        
    outputs, last_state, last_sequence_lengths = decoder(embedding.weights,initial_state=init_state,
                                                         start_tokens=tf.tile([SOS_token], [sample_batch_size]), end_token=EOS_token,training=False) 
    
    
    if decoder_type==1:
        result = tf.argmax(outputs.rnn_output,axis=-1).numpy()
        
        print(result)
        for i in range(sample_batch_size):
            print(''.join( index_to_char[a] for a in result[i] if a != EOS_token))

    else:
        result = outputs.predicted_ids.numpy()
        print(result.shape)
        for i in range(sample_batch_size):
            print(i,)
            for j in range(beam_width):
                print(''.join( index_to_char[a] for a in result[i,:,j] if a != EOS_token))








