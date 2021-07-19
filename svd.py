# coding: utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from scipy.fftpack import fft
import librosa
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import sys

# singular value of a ~ sqrt(eigenvalue) of (a.T)a  or a(a.T)
# data matrix a의 평균을 0으로 맞추고 나면, SVD는 PCA의 일반화. 
# SVD는 a.T.dot(a)를 계산하는 과정에서의 수치적인 손실을 줄일 수 있어 더 안정적이다.



# np.linalg의 첫번째  return  U의 열vector가 eigenvector에 해당함.

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


np.set_printoptions(threshold=sys.maxsize)
#a = np.random.randint(10, size=(9, 6))
#a =np.array([[5,-7,7],[4,-3,4],[4,-1,2]])
a =np.array([[0, 9, 7, 9, 1, 6, 7],
       [2, 3, 7, 8, 9, 6, 15],
       [2, 3, 4, 5, 9, 4, 13],
       [5, 8, 0, 3, 9, 5, 14],
       [7, 9, 9, 8, 4, 1, 5],
       [2, 7, 8, 2, 0, 0, 0],
       [8, 7, 0, 4, 8, 3, 12],
       [9, 2, 3, 8, 3, 0, 3],
       [6, 6, 2, 0, 9, 0, 9]])

def test1():
    
    aa = a.T.dot(a)  # covariance matrix of a
    w,z = np.linalg.eig(aa)  # z의 열벡터가 eigenvector
    
    bb = a.dot(a.T)
    ww,zz = np.linalg.eig(bb)  # zz의 열벡터가 eigenvector
    
    U, s, V = np.linalg.svd(a, full_matrices=True) # V의 행벡터가 aa의 eigenvector. vector들 간의 부호가 뒤죽박죽.
    # V의 행벡터가 aa의 eigenvector. vector들 간의 부호가 뒤죽박죽. 즉 z.T ~ V, 행들의 부호 때문에 잘 봐야함. rank 갯수 까지만 유효
    # U의 열벡터가 bb의 eigenvector. zz.T ~ U.T. rank 갯수까지 유효
    
    
    UU, ss, VV = np.linalg.svd(a, full_matrices=False) # "numerical recipes" 는 False에 해당하는 것을 구현
    
    print(U.shape,s.shape,V.shape)     # (9, 9) (7,) (7, 7)
    print(UU.shape,ss.shape,VV.shape)  # (9, 7) (7,) (7, 7)
    
    
    
    with printoptions(precision=3, suppress=True):
        
        
        # spectral decomposition은 eigenvalue의 정의에 의해 쉽게 증명가능
        print('eigen decomposition = spectral decomposition', z.dot(np.diag(w)).dot(z.T) - aa)  
        
        # enginvalue 비교
        print('eigenvalue 비교',np.sqrt(w) - s)
        print('U.dot(U.T)',U.dot(U.T))
        print('UU.dot(UU.T)',UU.dot(UU.T))   # Identity 안됨
        print('UU.T.dot(UU)',UU.T.dot(UU))
        print('V.dot(V.T)', V.dot(V.T))
        print('VV.dot(VV.T)', VV.dot(VV.T))
        
        # full matrix는 padding을 해야 복원
        print("복원(full): ", U.dot(np.pad(np.diag(s),[(0,2),(0,0)],'constant')).dot(V))
        # reduced form에서는 그냥 곱하면 됨
        print("복원(reduced)", UU.dot(np.diag(ss)).dot(VV))
        
        
        r = np.min(a.shape)
        print('thin SVD')
        print(U[:,:r].dot(np.diag(s[:r])).dot(V[:r,:]))
        print(UU[:,:r].dot(np.diag(ss[:r])).dot(VV[:r,:]))   
        
        
        # rank을 이용하여 ...
        r= np.linalg.matrix_rank(a)
        print('compact SVD')
        print(U[:,:r].dot(np.diag(s[:r])).dot(V[:r,:]))
        print(UU[:,:r].dot(np.diag(ss[:r])).dot(VV[:r,:]))
        
        
        r = 5 # rank 보다 작게 잡아, demension reduction을 할 수 있다.
        print('truncated SVD')
        print(U[:,:r].dot(np.diag(s[:r])).dot(V[:r,:]))
        print(UU[:,:r].dot(np.diag(ss[:r])).dot(VV[:r,:]))    
    
        # 행 축소
        r=4
        print('row reduction')
        print(U[:,:r].T.dot(a))
    
        # 열 축소
        r=3
        print('col reduction')
        print(a.dot(V[:r,:].T))


    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=3)
    vecs = svd.fit_transform(a) # svd.transform(a)와 같은 결과
    print("sklearn TruncatedSVD: ", vecs)  # col reduction 결과와 동일
    # explained_variance_, explained_variance_ratio_  --> 이것에 대한 설명은 TruncatedSVD 소스 코드를 보면 된다.
    print("explained_variance/explained_variance_ratio: ", svd.explained_variance_ , "/", svd.explained_variance_ratio_)



if __name__ == "__main__":
    test1()