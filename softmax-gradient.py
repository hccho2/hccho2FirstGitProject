# coding: utf-8
#https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy

import numpy as np


np.set_printoptions(suppress=True, formatter={'float_kind':'{:5.5f}'.format}, linewidth=130)


z = np.array([[-1.0, -1.0, 1.0],[-2.0, 0.0, 1.0]])

v = np.array([[-1.0, -1.0, 1.0],[-2.0, 0.0, 1.0]]) # unscaled logits
t = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]])     # target probability distribution


def softmax(v):
    exps = np.exp(v)
    sum  = np.sum(exps)
    return exps/sum

def cross_entropy(inps,targets):
    return np.sum(-targets*np.log(inps),axis=-1)

def cross_entropy_derivatives(inps,targets):
    return -targets/inps

# Fixed softmax derivative which returns the jacobian instead
# see https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function 

def softmax_derivatives(softmax_):
    if softmax_.ndim < 2:
        s = softmax_.reshape(-1,1)
        
        return np.diagflat(s) - np.dot(s, s.T)
    else:
        s = np.expand_dims(softmax_,2)
        dia = np.array([np.diagflat(x) for x in softmax_])
        return dia - np.matmul(s,np.transpose(s,[0,2,1]))



soft = softmax(v)           # [0.10650698, 0.10650698, 0.78698604]


L=cross_entropy(soft,t)       # 2.2395447662218846

cross_der = cross_entropy_derivatives(soft,t) 
                            # [-0.       , -9.3890561, -0.       ]

soft_der = softmax_derivatives(soft)

#[[ 0.09516324, -0.01134374, -0.08381951],
#[-0.01134374,  0.09516324, -0.08381951],
#[-0.08381951, -0.08381951,  0.16763901]]


# derivative using chain rule 
print(np.squeeze(np.matmul(np.expand_dims(cross_der,1), soft_der),axis=1))      # [[ 0.10650698, -0.89349302,  0.78698604]]

# Derivative using analytical derivation 
print(soft - t)                    # [ 0.10650698, -0.89349302,  0.78698604]



