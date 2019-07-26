# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.contrib.seq2seq import AttentionWrapperState
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention
from tensorflow.python.layers.core import Dense
tf.reset_default_graph()


class MyAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    """
    tf.contrib.seq2seq.AttentionWrapper의 call 함수만 변경. 
     이전 attention과 input을 concat하지 않고, attention 만 넣어준다.
    """
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None):
        super(MyAttentionWrapper, self).__init__(cell=cell,
                 attention_mechanism=attention_mechanism,
                 attention_layer_size=attention_layer_size,
                 alignment_history=alignment_history,
                 cell_input_fn=cell_input_fn,
                 output_attention=output_attention,
                 initial_cell_state=initial_cell_state,
                 name=name,
                 attention_layer=attention_layer)

    def call(self, inputs, state):

        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        #cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_inputs = state.attention  # 이 한줄 고치기 위해....
        
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

def attention_test():
    # BasicRNNCell을 single로 쌓아 attention 적용
    vocab_size = 6
    SOS_token = 0
    EOS_token = 5
    
    x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
    y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
    print("data shape: ", x_data.shape)
    sess = tf.InteractiveSession()
    
    output_dim = vocab_size
    batch_size = len(x_data)
    hidden_dim =7
    seq_length = x_data.shape[1]
    embedding_dim = 8
    state_tuple_mode = True

    init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1)
    
    train_mode = True
    alignment_history_flag = True   # True이면 initial_state나 last state를 sess.run 하면 안됨. alignment_history가 function이기 때문에...
    with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
        # Make rnn cell
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
    
        embedding = tf.get_variable("embedding", initializer=init.astype(np.float32),dtype = tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
    
        Y = tf.convert_to_tensor(y_data)
    
        #encoder_outputs = tf.ones([batch_size,20,30])
        encoder_outputs = tf.convert_to_tensor(np.random.normal(0,1,[batch_size,20,30]).astype(np.float32)) # 20: encoder sequence length, 30: encoder hidden dim
        
        #input_lengths = [20]*batch_size
        input_lengths = [5,10,20]  # encoder에 padding 같은 것이 있을 경우, attention을 주지 않기 위해
        
        # attention mechanism  # num_units = Na = 11
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths,normalize=False)
        #attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(num_units=11, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        # LuongAttention에서는 num_units이 임의로 들어가면 안되고, decoder의 hidden_dim과 일치해야 한다
        #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hidden_dim, memory=encoder_outputs,memory_sequence_length=input_lengths)
        
        
        # output_attention = True(default) ==> 이면 output으로 attention이 나가고, False이면 cell의 output이 나간다
        # attention_layer_size = N_l
        
        attention_initial_state = cell.zero_state(batch_size, tf.float32)
        
        
        
        
        ###############################
        ###############################
        
        cell = MyAttentionWrapper(cell, attention_mechanism, attention_layer_size=13,initial_cell_state=attention_initial_state,
                                                   alignment_history=alignment_history_flag,output_attention=True)


        ###############################
        ###############################


        # 여기서 zero_state를 부르면, 위의 attentionwrapper에서 넝허준 attention_initial_state를 가져온다. 즉, AttentionWrapperState.cell_state에는 넣어준 값이 들어있다.
        initial_state = cell.zero_state(batch_size, tf.float32) # AttentionWrapperState
 
        if train_mode:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
     
        output_layer = Dense(output_dim, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
        # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
        outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)
     
        weights = tf.ones(shape=[batch_size,seq_length])
        loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
     
     
        
        
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)
        
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            loss_,_ =sess.run([loss,opt])
            print("{} loss: = {}".format(i,loss_))
        
        if alignment_history_flag ==False:
            print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
     
        print("\n\nlast_state: ",last_state)
        if alignment_history_flag == False:
            print(sess.run(last_state)) # batch_size, hidden_dim
        else:
            print("alignment_history: ", last_state.alignment_history.stack())
            alignment_history_ = sess.run(last_state.alignment_history.stack())
            print(alignment_history_)
            print("alignment_history sum: ",np.sum(alignment_history_,axis=-1))
            
            print("cell_state: ", sess.run(last_state.cell_state))
            print("attention: ", sess.run(last_state.attention))
            print("time: ", sess.run(last_state.time))
            
            alignments_ = sess.run(last_state.alignments)
            print("alignments: ", alignments_)
            print('alignments sum: ', np.sum(alignments_,axis=1))   # alignments의 합이 1인지 확인
            print("attention_state: ", sess.run(last_state.attention_state))

     
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
         
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
     
        if train_mode:
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )            




if __name__ == '__main__':

    attention_test()

    
    print('Done')
















