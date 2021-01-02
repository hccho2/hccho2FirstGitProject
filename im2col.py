
import numpy as np
import tensorflow as tf
import torch
print(f'Tensorflow Version: {tf.__version__}')

HCCHO = True  # cn231n code에서 im2col의 2-dim의 행벡터가 batch data가 섞여서 나오는 방식이라, 이를 batch data간에 섞이지 않는 방식으로 변환.

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride) + 1
    out_width = int((W + 2 * padding - field_width) / stride) + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]  # channel_in
    
    if HCCHO:
        cols = np.concatenate(cols,axis=-1).reshape(field_height * field_width * C, -1)  # ---> 이렇게 하면, column에 batch가 섞이지 않는데...    
    else:
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)  # 왜 batch간에 섞이게 coding했을까?
    
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    
    if HCCHO:
        cols_reshaped = np.stack(np.split(cols,N,axis=-1))  # hccho ---> im2col_indices 에서 내가 사용한 방식을 사용했다면...
    else:
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1) # im2col_indices에서 cols.transpose(1, 2, 0)....  에서 바꿔놓은 거 되돌리기.
    
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    #x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)  # cython으로 만든 pyd
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    if HCCHO:
        out = res.reshape(w.shape[0], x.shape[0], out.shape[2], out.shape[3])  # (c_out,N, OH,OW)
        out = out.transpose(1, 0, 2, 3)
    else:
        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])  # (c_out,OH,OW, N)
        out = out.transpose(3, 0, 1, 2)  # 

    cache = (x, w, b, conv_param, x_cols)
    return out, cache
def conv_backward_im2col(dout, cache):
    """
    A fast implementation of the backward pass for a convolutional layer
    based on im2col and col2im.
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    
    
    if HCCHO:
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(num_filters, -1)
    else:
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    
    
    
    
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    #dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3], filter_height, filter_width, pad, stride)

    return dx, dw, db

def forward_test():
    # tensorflow, pytorch와 convolution 결과 비교
    
    
    
    data_case = 0
    
    if data_case==1:
        stride=2;pad=2;
        conv_param = {'stride': stride, 'pad': pad}
        c_in = 3; c_out=2
        FH=5; FW=5
        H=11; W=11
        batch_size=2
    else:
        stride=1;pad=0;
        conv_param = {'stride': 1, 'pad': 0}
        c_in = 3; c_out=2
        FH=3; FW=3
        H=4; W=5
        batch_size=2
    
    
    
    OH = int((H-FH+2*pad)/stride) + 1
    OW = int((W-FW+2*pad)/stride) + 1
    print(f'OH: {OH}, OW: {OW}')
    
      
    x = np.arange(batch_size*c_in*H*W,dtype=np.float32).reshape(batch_size,c_in,H,W)  # x: (N,C,H,W)
    y = im2col_indices(x,FH,FW,conv_param['pad'],conv_param['stride'])
    print(y.shape)  # (c_in*FH*FW,batch_size*OH*OW)
    
    
    
    w = np.random.normal(size=(c_out,c_in,FH,FW))
    b = np.random.normal(size=c_out)
    
    # numpy
    z,_ = conv_forward_im2col(x, w, b, conv_param)
    
    # tensorflow
    #zz = tf.nn.conv2d(x.transpose(0,2,3,1),w.transpose(2,3,1,0),strides=conv_param['stride'],padding='VALID',data_format='NHWC') + b
    
    if pad > 0:
        zz = tf.nn.conv2d(x,w.transpose(2,3,1,0),strides=conv_param['stride'],padding=[[0,0],[0,0],[pad,pad],[pad,pad]],data_format='NCHW') + b.reshape(1,-1,1,1)
    else:
        zz = tf.nn.conv2d(x,w.transpose(2,3,1,0),strides=conv_param['stride'],padding='VALID',data_format='NCHW') + b.reshape(1,-1,1,1)
    
    # pytorch
    zzz = torch.nn.functional.conv2d(torch.Tensor(x),torch.Tensor(w),torch.Tensor(b),conv_param['stride'],conv_param['pad'])
    
    
    
    print(z.shape, zz.shape, zzz.shape)
    print(z,'\n', zz,'\n', zzz)  # ==> 결과가 같다.


    


def im2col_test():
    stride=1;pad=0;
    c_in = 3
    FH=3; FW=3
    H=4; W=5   
    batch_size = 2 
    
    OH = int((H-FH+2*pad)/stride) + 1
    OW = int((W-FW+2*pad)/stride) + 1
    print(f'OH: {OH}, OW: {OW}')    
    
    x = np.arange(batch_size*c_in*H*W,dtype=np.float32).reshape(batch_size,c_in,H,W)  # x: (N,C,H,W)
    
    y = im2col_indices(x,FH,FW,padding=0, stride=1)
    print(x.shape, y.shape)  # y: (FH*FW*c_in,OH*OW*N)   
    
    print(x)
    print(y) 
    
    
    k, i, j = get_im2col_indices(x.shape, FH,FW,padding=0, stride=1)

    print("k",k.shape, k)
    print("i",i.shape, i)
    print("j",j.shape, j)

def col2im_test():
    conv_param = {'stride': 1, 'pad': 0}
    c_in = 1; c_out=1
    FH=2; FW=2
    H=3; W=4   
    batch_size = 2 
    x = np.arange(batch_size*c_in*H*W,dtype=np.float32).reshape(batch_size,c_in,H,W)  # x: (N,C,H,W)
    x = np.random.normal(size=(batch_size,c_in,H,W))

    y = im2col_indices(x,FH,FW,padding=0, stride=1)
    print(x.shape, y.shape)  # (75,9)   
    
    print(f'image: {x}')
    print(f'im2col: {y}') 
    
    
    k, i, j = get_im2col_indices(x.shape, FH,FW,padding=0, stride=1)

    print("k",k.shape, k)
    print("i",i.shape, i)
    print("j",j.shape, j)



    w = np.random.normal(size=(c_out,c_in,FH,FW))
    b = np.random.normal(size=c_out)
    print(f'w: {w}')
    print(f'b: {b}')
    # numpy
    z,cache = conv_forward_im2col(x, w, b, conv_param)    
    
    print(f'z shape: {z.shape}, z: {z}')
    
    dz = np.random.normal(size=z.shape)
    print(f'dz(random): {dz}')
    
    dx, dw, db = conv_backward_im2col(dz,cache)  # col2im은 dx 계산에 사용된다.

    print(f'dx: {dx}')
    print(f'dw: {dw}')
    print(f'db: {db}')


def col2im_test2():
    # batch_size=1, stride=1, pad=0 인 경우에만...
    # 이 조건이 아니면, 원래 코드대로 월씬 복잡해진다.
    random_data=False
    
    #FH=2; FW=2; H=3; W=4;batch_size = 1; stride=1; pad=0
    FH=5; FW=5; H=11; W=11;batch_size = 2; stride=2;pad=1

    conv_param = {'stride': stride, 'pad': pad}
    c_in = 1; c_out=2    
     
    OH = int((H-FH+2*pad)/stride) + 1
    OW = int((W-FW+2*pad)/stride) + 1
    print(f'OH: {OH}, OW: {OW}')  
    if random_data:
        #x = np.random.normal(size=(batch_size,c_in,H,W))  # x: (N,C,H,W)
        x = np.arange(batch_size*c_in*H*W,dtype=np.float32).reshape(batch_size,c_in,H,W)
    else:
        stride=1; pad=0
        c_in = 1; c_out=1
        FH=2; FW=2
        H=3; W=4
        batch_size = 1 
        conv_param = {'stride': stride, 'pad': pad}
        
        OH = int((H-FH+2*pad)/stride) + 1
        OW = int((W-FW+2*pad)/stride) + 1
    
        x = np.array([[[[ 1.83408274, -0.25310561, -0.77140011, -1.15510552],
           [ 0.41389487, -0.30680423,  1.14462378,  0.37406678],
           [ 1.24669999,  0.33290709, -0.99042362, -0.51790133]]]])
    
    
    y = im2col_indices(x,FH,FW,padding=0, stride=1)
    print(x.shape, y.shape)  # (75,9)   
    
    print(f'image: {x}')
    print(f'im2col: {y}') 
    
    
    k, i, j = get_im2col_indices(x.shape, FH,FW,padding=0, stride=1)
    
    print("k",k.shape, k)
    print("i",i.shape, i)
    print("j",j.shape, j)
    
    
    if random_data:
        w = np.random.normal(size=(c_out,c_in,FH,FW))
        b = np.random.normal(size=c_out)
    else:
        w = np.array([[[[-0.51151987,  0.11880424],
            [-0.86983925,  1.38963489]]]])
        b = np.array([1.41023464])
    
    print(f'w: {w}')
    print(f'b: {b}')
    # numpy
    z,cache = conv_forward_im2col(x, w, b, conv_param) #(N,c_out,OH,OW)
    
    print(f'z shape: {z.shape}, z: {z}')
    
    if random_data:
        dz = np.random.normal(size=z.shape)
    else:
        dz = np.array([[[[ 2.65539899, -0.61304065, -0.99982636],
            [-1.70924638,  1.37006551,  0.03781655]]]])
    
    
    print(f'dz(random): {dz}')
    
    dx, dw, db = conv_backward_im2col(dz,cache)  # col2im은 dx 계산에 사용된다.
    
    print(f'dx: {dx}')
    print(f'dw: {dw}')
    print(f'db: {db}')
    
    print('='*10)
    
    if batch_size==1 and pad==0 and stride==1:
        # HCCHO = True/False를 구분하지 않았기 때문....
        print('manual compuation: ')
        # z확인
        print(f'im2col(manual, z와 일치): { (w.reshape(c_out,-1).dot(y) + b.reshape(-1,1)).reshape(batch_size,c_out,OH,OW)  }')
        
        # dw확인
        print(f'dw(manual): {dz.reshape(c_out,-1).dot(y.T).reshape(w.shape)}')
        
        # db확인
        print(f'db(manual): {dz.sum(axis=(0, 2, 3))}')
        
        
        # dx확인
        dx_ = np.zeros_like(x)
        dy = w.reshape(c_out,-1).T.dot(dz.reshape(c_out,-1))[None]
        np.add.at(dx_,(slice(None),k,i,j),dy)
        print(f'dx(manual): {dx_}')
    
    
    #########
    # pytorch로 확인
    print('='*10)
    tensor_x = torch.tensor(x,requires_grad=True,dtype=np.float)
    tensor_w = torch.tensor(w,requires_grad=True,dtype=np.float)
    tensor_b = torch.tensor(b,requires_grad=True,dtype=np.float)
    
    zz = torch.nn.functional.conv2d(tensor_x,tensor_w,tensor_b,conv_param['stride'],conv_param['pad'])
                                    
    zz.backward(torch.Tensor(dz))
    
    
    print(f'torch dw: {tensor_w.grad}')
    print(f'torch db: {tensor_b.grad}')
    print(f'torch dx: {tensor_x.grad}')
    


if __name__ == '__main__':
    #forward_test()  # tensorflow, pytorch와 convolution 결과 비교
    #im2col_test()
    #col2im_test()
    col2im_test2()
    


