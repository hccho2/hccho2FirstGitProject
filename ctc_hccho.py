# coding: utf-8

'''
conv2d: weight(kernel_size,kernel_size,in_channel,out_channel)
conv2d_transpose: weight(kernel_size,kernel_size,out_channel,in_channel)

'''


import numpy as np
import tensorflow as tf
import torch
from torch import nn
import re
import string
import codecs
import datetime
from tensorflow.python.ops.parallel_for.gradients import jacobian
tf.reset_default_graph()
def CTC_Loss():
    # ctc_loss v1에서는 sparse matrix가 들어가기 때문에, gt(label)에 0번 character가 포함되어 있으면, 0번에 대한 loss를 계산못한다.
    # v2에서도 sparse를 넣어주면 같은 결과가 나온다. 
    # 이는 0번을 padding으로 인식하는 문제가 있기 때문이다.
    # 따라서, 0번에는 의미 있는 charcter를 부여하면 안된다.
    # v2에서 label에 sparse가 아닌, dense를 넣어주어야 한다.


    batch_size=2
    output_T=5
    target_T=3 # target의 길이. Model이 만들어 내는 out_T는 target보다 길다.
    num_class = 4 # 0, 1, 2는 character이고, 마지막 3은 blank이다.

    x = np.arange(40).reshape(batch_size,output_T,num_class).astype(np.float32)
    x = np.random.randn(batch_size,output_T,num_class)
    x = np.array([[[ 0.74273746,  0.07847633, -0.89669566,  0.87111101],
            [ 0.35377891,  0.87161664,  0.45004634, -0.01664156],
            [-0.4019564 ,  0.59862392, -0.90470981, -0.16236736],
            [ 0.28194173,  0.82136263,  0.06700599, -0.43223688],
            [ 0.1487472 ,  1.04652007, -0.51399114, -0.4759599 ]],

           [[-0.53616811, -2.025543  , -0.06641838, -1.88901458],
            [-0.75484499,  0.24393693, -0.08489008, -1.79244747],
            [ 0.36912486,  0.93965647,  0.42183299,  0.89334628],
            [-0.6257366 , -2.25099419, -0.59857886,  0.35591563],
            [ 0.72191422,  0.37786281,  1.70582983,  0.90937337]]]).astype(np.float32)

    xx = tf.convert_to_tensor(x)
    xx = tf.Variable(xx)
    logits = tf.transpose(xx,[1,0,2])

    yy = np.random.randint(0,num_class-1,size=(batch_size,target_T))  # low=0, high=3 ==> 0,1,2
    yy = np.array([[1, 2, 2],[1, 0, 1]]).astype(np.int32)
    #yy = np.array([[1, 2, 2,0,0,0],[1,0,2,0,0,0]]).astype(np.int32)  # 끝에 붙은 0은 pad로 간주한다. 중간에 있는 0은 character로 간주

    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(yy, zero)
    indices = tf.where(where)
    values = tf.gather_nd(yy, indices)
    targets = tf.SparseTensor(indices, values, yy.shape)


    # preprocess_collapse_repeated=False  ---> label은 반복되는 character가 있을 수 있으니, 당연히 False
    # ctc_merge_repeated=False  ---> 모델이 예측한 반복된 character를 merge하지 않는다. 이것은 ctc loss의 취지와 다르다.
    loss0 = tf.nn.ctc_loss(labels=targets,inputs=logits,sequence_length=[output_T]*batch_size,ctc_merge_repeated=False) 
    # 이 loss0는 의미 없음.

    loss1 = tf.nn.ctc_loss(labels=targets,inputs=logits,sequence_length=[output_T]*batch_size)
    loss2 = tf.nn.ctc_loss_v2(labels=yy,logits=logits,label_length =[target_T]*batch_size,
                              logit_length=[output_T]*batch_size,logits_time_major=True,blank_index=num_class-1)


    # lables에 sparse tensor를 넣으면, v1과 결과가 같다. 
    loss3 = tf.nn.ctc_loss_v2(labels=targets,logits=logits,label_length =[3,3],
                              logit_length=[output_T]*batch_size,logits_time_major=True,blank_index=num_class-1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    gradient = optimizer.compute_gradients(loss1)[0]  # return (gradient, variable)


    prob = tf.nn.softmax(xx,axis=-1)
    # jacobian을 이용해서 logits에 대한 softmax값의 미분을 구한다.
    a = xx[0,1]
    b = tf.nn.softmax(a)
    grad = jacobian(b,a) # logit에 대한 softmax 미분.


    # logit에 대한 미분을 softmax에 대한 미분으로 변환하기 위해 grad의 inverse를 곱한다.
    # grad의 역행렬이 존재하지 않는다.


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    l0 = sess.run(loss0)
    l1 = sess.run(loss1)
    l2 = sess.run(loss2)
    l3 = sess.run(loss3)
    print('loss: ',l0, l1,l2,l3)  # l1=l2=l3
    g = sess.run(gradient)  # g[1]은 x값과 같고, g[0]이 gradient이다.  g[0][0][1] == 엑셀에서 계산한 값
    p = sess.run(prob)  # logit을 softmax취한 확률값.
    gg = sess.run(grad)  # logit에 대한 softmax 미분.

    print('엑셀값과 비교:', g[0][0][1])  # g[0][첫번째 batch][두번째 time step]


def CTC_Loss2():
    # tensorflow, pytorch에서의 ctc loss 비교 --> 같은 결과
    batch_size=2
    output_T=5
    target_T=3 # target의 길이. Model이 만들어 내는 out_T는 target보다 길다.
    num_class = 4 # 0(pad), 1(a), 2(b)는 character이고, 마지막 3(blank)이다.
    
    x = np.arange(40).reshape(batch_size,output_T,num_class).astype(np.float32)
    x = np.random.randn(batch_size,output_T,num_class)
    x = np.array([[[ 0.74273746,  0.07847633, -0.89669566,  0.87111101],
            [ 0.35377891,  0.87161664,  0.45004634, -0.01664156],
            [-0.4019564 ,  0.59862392, -0.90470981, -0.16236736],
            [ 0.28194173,  0.82136263,  0.06700599, -0.43223688],
            [ 0.1487472 ,  1.04652007, -0.51399114, -0.4759599 ]],
    
           [[-0.53616811, -2.025543  , -0.06641838, -1.88901458],
            [-0.75484499,  0.24393693, -0.08489008, -1.79244747],
            [ 0.36912486,  0.93965647,  0.42183299,  0.89334628],
            [-0.6257366 , -2.25099419, -0.59857886,  0.35591563],
            [ 0.72191422,  0.37786281,  1.70582983,  0.90937337]]]).astype(np.float32)
    
    # tensorflow
    xx = tf.convert_to_tensor(x)
    xx = tf.Variable(xx)
    logits = tf.transpose(xx,[1,0,2])
    #yy = np.array([[1, 2, 2],[1, 2, 1]]).astype(np.int32)
    yy = np.array([[1, 2, 2,0,0],[1, 2, 1,0,0]]).astype(np.int32)  # 0: padding
    label_length=[target_T]*batch_size   # label_length는 다양한 방법으로 구할 수 있다.
    logit_length = [output_T]*batch_size
    
    
    loss_tf = tf.nn.ctc_loss_v2(labels=yy,logits=logits,label_length =label_length,
                              logit_length=logit_length,logits_time_major=True,blank_index=num_class-1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loss_tf_ = sess.run(loss_tf)
    print('loss_tf: ', loss_tf_)


    # pytorch
    CTCLoss = torch.nn.CTCLoss(blank=num_class-1, reduction='none', zero_infinity=True)
    logits = torch.from_numpy(x).type(torch.float32).log_softmax(2)
    logits = logits.permute(1, 0, 2)  # torch.nn.CTCLoss에 넣기 위해서, (T,N,C)로 변환
    
    Input_lengths = torch.IntTensor(logit_length)
    Target_lengths = torch.IntTensor(label_length)
    target = torch.from_numpy(yy[yy>0]).type(torch.int32).view(-1)
    
    
    loss_torch = CTCLoss(logits,target,Input_lengths,Target_lengths)

    
    print('loss_torch: ', loss_torch.numpy())  # tensorflow, pytorch의 결과가 일치한다.








    print('Done')


if __name__ == '__main__':
    #CTC_Loss()
    CTC_Loss2()
