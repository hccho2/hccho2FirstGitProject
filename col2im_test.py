#  coding: utf-8
import numpy as np
np.set_printoptions(threshold=np.nan)
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
def col2im_back(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    밑바닥 교재의 col2im의 이름을 col2im_back으로 바꿈
    아래의 col2im 함수는 수정된 버전으로 있음.
    img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]  <---원교재
    img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]  <--- hccho수정    
    
    
    
    (im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]    
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    밑바닥 교재의 col2im의 이름을 col2im_back으로 바뀜. 
    이함수 col2im은 수정된 버전.
    img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]  <---원교재
    img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]  <--- hccho수정    
    
    
    (im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]     
X = np.array([[[[9, 9, 0, 7, 8, 4, 6],
         [8, 1, 5, 0, 2, 3, 5],
         [8, 9, 1, 7, 3, 9, 2],
         [2, 2, 9, 1, 3, 0, 5],
         [1, 4, 2, 7, 5, 0, 4],
         [0, 2, 6, 9, 3, 2, 6],
         [8, 1, 1, 6, 9, 1, 5]],

        [[7, 7, 2, 1, 5, 5, 4],
         [5, 9, 4, 6, 4, 4, 4],
         [2, 9, 4, 2, 4, 2, 5],
         [2, 0, 5, 6, 0, 8, 5],
         [3, 9, 4, 9, 7, 5, 4],
         [8, 7, 2, 0, 2, 5, 3],
         [8, 3, 5, 7, 5, 4, 7]],

        [[7, 5, 4, 4, 6, 9, 0],
         [4, 9, 0, 3, 9, 1, 0],
         [6, 5, 5, 3, 6, 3, 7],
         [7, 1, 9, 8, 2, 7, 4],
         [5, 1, 3, 8, 2, 7, 7],
         [4, 6, 2, 8, 3, 3, 8],
         [1, 5, 6, 5, 5, 7, 6]]]])


W = np.array([[[[-1.71,  1.37, -0.35, -0.08,  1.54],
         [ 0.76,  1.64, -1.35,  0.53, -2.43],
         [-0.61,  0.8 , -1.08,  1.65,  0.67],
         [-1.22,  0.31,  1.89, -1.18, -0.64],
         [ 0.79, -0.29,  0.11, -0.49,  2.2 ]],

        [[ 1.23,  0.76,  0.97,  0.62,  0.73],
         [ 0.16,  0.1 , -1.1 ,  0.46, -1.51],
         [-0.08, -0.67, -0.81, -0.35,  0.83],
         [-1.41,  0.25, -0.58, -0.11, -1.66],
         [-0.19, -0.4 , -0.32,  0.74,  0.87]],

        [[-0.19,  0.8 , -1.09,  1.56,  0.04],
         [ 0.73,  1.29, -1.58,  0.47, -1.08],
         [ 0.38,  0.71, -0.65,  0.57, -0.51],
         [ 1.01,  0.44,  1.85,  1.42, -0.08],
         [-1.23,  1.24,  0.71, -0.34, -1.34]]],


       [[[ 1.34, -0.43, -1.56,  0.54, -0.52],
         [ 2.  , -0.93, -0.56,  0.01,  0.14],
         [-0.48,  2.43, -0.73,  0.28, -2.  ],
         [ 0.22, -0.28, -0.47,  0.66,  0.72],
         [ 1.45, -0.7 ,  1.4 , -0.02, -0.53]],

        [[-1.28, -0.65,  0.33,  1.75,  2.57],
         [-0.24,  0.7 ,  0.3 , -0.09, -0.26],
         [ 0.1 , -0.43, -1.39, -1.48,  0.14],
         [ 0.31,  0.53,  0.9 , -0.52,  0.47],
         [ 0.37, -0.21, -1.31, -0.17,  0.44]],

        [[ 0.25, -0.26,  0.5 , -0.09,  1.01],
         [ 2.06, -0.06, -0.85, -0.57, -0.76],
         [-0.57, -0.01,  0.18, -0.04,  0.01],
         [-0.18,  0.66,  0.59, -0.02,  0.55],
         [-0.84, -0.85, -0.33, -0.12,  0.04]]]])

# N = 1, C = 3, H=7, W=7, FN=2, FH=5,FW=5, stride = 1, pad=0    
X1 = im2col(X,5,5,1,0)
W1 = W.reshape(2,-1).T

out = np.dot(X1,W1)
print(X.shape,W.shape,X1.shape,W1.shape)
out_h = int((7-2*0-5)/1) +1
out_w = int((7-2*0-5)/1) +1
out1 = out.reshape(1,out_h,out_w,-1).transpose(0,3,1,2)

X2 = col2im(X1,X.shape,5,5,1,0)
