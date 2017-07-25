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
    
X0 = np.array([[-0.15, -0.12,  1.29, -1.64, -0.02,  0.62, -0.81, -0.21,  0.7 ,
        -0.66, -1.26, -1.99,  0.31,  0.64, -2.38,  0.25, -0.53, -0.04,
        -1.16, -0.55,  2.45, -0.65, -2.08,  1.84, -0.11, -0.03, -0.97,
        -1.01,  0.87, -0.2 , -0.09, -0.09, -1.26,  0.8 , -1.19, -0.13,
        -0.81,  0.04,  0.86, -0.71,  0.48,  2.2 , -0.32, -0.83, -0.37,
        -1.14,  2.41,  1.26, -0.11, -0.9 ,  0.47, -0.09,  0.74, -0.61,
        -0.3 ,  0.18, -0.44,  1.98, -0.86, -0.36,  0.65,  0.27,  0.83,
         0.58, -0.58, -1.1 ,  1.24, -0.77, -1.01, -0.6 , -1.59, -0.9 ,
         1.79, -0.83,  0.57],
       [ 1.67,  1.5 ,  0.11,  0.64,  0.24, -1.42,  0.66, -0.72, -0.09,
        -0.23, -1.27, -0.65,  0.88, -1.57, -1.63,  0.77, -0.31,  0.92,
        -0.28,  1.41,  0.58, -1.21, -2.26,  0.63,  1.36,  0.82, -1.88,
         0.47, -0.63, -0.25,  0.07,  0.59,  0.07,  0.39, -0.84,  0.05,
         0.8 , -1.1 , -0.91, -1.03, -0.3 ,  1.22, -1.02, -1.32,  0.88,
         0.84, -0.38, -0.78,  1.1 ,  0.26, -0.38,  1.14,  2.32, -0.7 ,
        -0.32,  0.22, -0.82, -1.4 ,  0.16, -0.27,  0.49,  0.56, -0.08,
        -0.64,  0.37, -1.18, -0.64, -0.42,  1.13, -0.18, -0.27, -0.55,
        -0.91,  1.13, -1.07],
       [ 0.35,  0.62,  1.07, -0.35, -1.08, -0.72, -0.57,  1.17,  0.35,
         0.02, -0.74,  0.2 ,  1.57,  1.2 , -0.41,  1.33, -0.68,  0.83,
         0.39, -1.9 ,  0.61, -0.51,  0.1 ,  0.12,  0.45,  0.15, -0.61,
         1.27, -0.44, -0.37,  0.57,  1.75, -0.47,  1.76,  0.59, -0.36,
        -0.02, -0.1 ,  1.67,  0.34,  0.09,  2.02, -0.25, -0.1 , -1.31,
        -1.42, -0.6 ,  0.1 , -1.32,  1.09,  0.1 ,  1.06,  0.94, -0.23,
         1.49, -1.8 ,  0.9 , -0.58,  1.77, -0.87,  0.75,  0.86,  0.99,
        -1.14, -2.41, -0.25,  0.83, -0.31, -0.39,  0.65,  0.67, -0.49,
        -2.03,  0.83,  1.79],
       [ 1.52,  1.76,  1.38,  0.33,  0.73, -0.58,  0.45, -0.57, -0.15,
        -0.37,  1.14,  0.05, -0.94,  0.39,  1.22, -0.91, -1.02, -1.52,
        -1.03, -0.64, -1.23, -0.53,  0.79, -1.84, -0.5 , -0.09,  0.65,
         0.53, -0.09, -0.06, -0.03, -0.59, -1.05,  0.63, -0.41,  1.42,
        -0.33, -1.73,  0.13,  0.61,  2.56, -1.8 ,  1.22, -0.62,  0.62,
         0.52,  1.83, -0.21, -1.09,  1.49, -0.15, -0.7 ,  1.28,  0.3 ,
        -1.15, -1.73,  0.18,  0.9 , -1.83, -0.4 , -0.58, -0.33, -1.36,
         0.05, -0.48, -0.05, -0.43, -2.04,  0.39,  1.55,  0.26,  0.82,
        -0.65,  0.42, -0.55],
       [ 0.16,  2.69, -1.02, -0.93,  0.09, -0.7 ,  1.45, -0.97, -1.02,
         1.09,  0.03,  1.11,  1.17,  0.17,  0.11, -1.21,  0.61, -0.88,
         0.94,  0.14, -1.02,  0.89, -0.83, -0.52, -0.43,  0.62,  0.22,
         0.68, -0.71, -1.58,  0.25,  0.84, -0.13, -0.15,  0.28, -2.68,
        -1.18, -1.44,  0.75,  0.11, -0.14, -0.41, -0.91,  1.53, -0.49,
         0.74,  0.28, -1.24,  0.54, -1.92, -1.58,  0.58,  1.01,  1.07,
         0.52, -0.25,  0.63, -0.8 , -0.62,  1.32, -0.72,  0.61,  1.8 ,
        -0.31,  0.73, -0.08,  0.28, -0.37,  0.64,  0.36,  1.79,  0.92,
         0.93,  1.18,  0.27],
       [-0.27, -1.04, -0.1 , -0.32,  0.65, -2.08, -0.99, -0.8 , -0.11,
         0.08, -1.06,  0.58, -0.66,  0.11, -1.42, -1.28,  0.71, -0.67,
        -0.1 , -1.63,  0.5 , -0.69, -1.65, -0.62, -0.35,  2.05,  0.22,
         0.75, -0.69,  1.02,  0.91,  0.2 ,  0.89,  0.51, -0.15,  0.81,
         0.26,  0.6 , -0.54, -0.14, -0.04,  0.46, -0.5 , -0.37, -1.3 ,
        -0.39,  0.23, -0.16, -1.13,  0.04, -0.75, -1.13, -0.04, -1.13,
        -0.98,  1.03,  1.26,  1.09,  1.68, -1.02, -0.57,  0.64,  0.  ,
        -0.49, -0.34, -1.17,  0.25,  1.27,  0.95,  0.18, -1.63, -0.75,
        -0.39,  1.4 ,  0.66],
       [-0.99, -0.3 , -2.08,  0.31,  0.3 ,  0.  ,  0.29, -0.21,  1.18,
        -1.24, -0.48, -1.62,  0.54, -1.34,  1.44, -0.81, -0.96,  3.76,
         1.01, -1.55,  0.13,  0.68, -1.74,  0.19,  0.05, -1.28, -0.03,
        -0.01,  2.58, -1.  ,  0.  , -1.09, -1.58,  1.54, -0.71, -0.16,
         0.18,  0.99, -0.27,  1.05, -0.12, -0.35,  0.48,  1.74,  0.46,
         0.67, -0.08, -0.6 , -0.68, -0.53,  0.24,  1.94,  1.15, -0.71,
        -2.07,  0.34, -0.42, -1.87, -0.64, -0.02, -2.13,  0.43,  0.99,
        -0.05,  2.17, -0.77,  0.94, -0.53,  0.99,  1.75, -1.37,  1.35,
         0.26,  0.41, -0.36],
       [-0.04,  0.47, -0.57, -0.46,  0.45, -0.29, -0.55,  0.08, -1.07,
         0.45, -0.94,  0.62,  0.16,  1.17,  0.36,  0.45,  1.31,  0.4 ,
        -0.45, -0.83,  1.85,  0.9 ,  0.08,  0.34, -1.97, -0.99,  0.62,
         0.89, -0.39, -0.06, -0.64, -0.46,  0.23,  0.91,  0.89,  2.09,
        -1.02,  0.15, -2.57,  1.5 ,  1.  ,  0.55,  0.93,  0.14, -0.45,
         0.86, -0.72, -1.35, -0.71, -0.49, -0.45,  0.11, -1.31,  1.38,
        -0.75,  0.36,  1.4 , -0.26,  0.14,  0.09, -1.58, -0.09, -0.32,
         0.46,  0.4 ,  0.85, -0.07, -1.71, -0.03, -0.55, -0.98, -0.37,
         1.47,  0.79, -0.5 ],
       [ 0.78,  1.06,  0.2 , -0.39, -1.77, -1.34, -1.19, -0.6 , -0.33,
        -0.23, -0.13,  0.73, -1.63, -0.45,  0.07,  1.1 ,  0.02, -0.35,
        -0.71, -0.16, -0.68,  0.59,  0.19, -1.07,  0.47, -0.14,  0.26,
        -0.09, -0.92,  0.82,  1.  , -0.94, -0.3 ,  0.16,  0.64, -1.13,
         0.18, -0.64,  0.1 ,  1.74,  0.57,  1.1 ,  1.84, -0.67, -0.94,
         0.82, -1.03,  0.77, -2.24, -0.06,  0.34, -2.33, -1.7 ,  0.3 ,
        -0.23,  0.9 , -0.25, -0.45, -0.45,  0.72, -0.6 ,  0.11, -0.15,
         0.13, -1.27, -0.75,  1.8 , -0.53,  1.78, -0.21, -0.8 , -0.38,
        -0.6 ,  0.07, -1.61]])    
    

Z=np.array([[[[  0,   1,   2,   3,   4,   5,   6],
         [  7,   8,   9,  10,  11,  12,  13],
         [ 14,  15,  16,  17,  18,  19,  20],
         [ 21,  22,  23,  24,  25,  26,  27],
         [ 28,  29,  30,  31,  32,  33,  34],
         [ 35,  36,  37,  38,  39,  40,  41],
         [ 42,  43,  44,  45,  46,  47,  48]],

        [[ 49,  50,  51,  52,  53,  54,  55],
         [ 56,  57,  58,  59,  60,  61,  62],
         [ 63,  64,  65,  66,  67,  68,  69],
         [ 70,  71,  72,  73,  74,  75,  76],
         [ 77,  78,  79,  80,  81,  82,  83],
         [ 84,  85,  86,  87,  88,  89,  90],
         [ 91,  92,  93,  94,  95,  96,  97]],

        [[ 98,  99, 100, 101, 102, 103, 104],
         [105, 106, 107, 108, 109, 110, 111],
         [112, 113, 114, 115, 116, 117, 118],
         [119, 120, 121, 122, 123, 124, 125],
         [126, 127, 128, 129, 130, 131, 132],
         [133, 134, 135, 136, 137, 138, 139],
         [140, 141, 142, 143, 144, 145, 146]]]])
    
    
    
# N = 1, C = 3, H=7, W=7, FN=2, FH=5,FW=5, stride = 1, pad=0    
X1 = im2col(X,5,5,1,0)
W1 = W.reshape(2,-1).T

out = np.dot(X1,W1)
print(X.shape,W.shape,X1.shape,W1.shape)
out_h = int((7-2*0-5)/1) +1
out_w = int((7-2*0-5)/1) +1
out1 = out.reshape(1,out_h,out_w,-1).transpose(0,3,1,2)

X2 = col2im_back(X0,X.shape,5,5,1,0)


