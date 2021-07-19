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

def covariance_test():
    N=20; M=4
    X = np.concatenate([np.random.randn(N,M-1), np.random.rand(N,1)],axis=-1) + np.random.randn(1,M)*5.0

    m = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    print("mean, std: ", m, std)
    X0 = X-m
    X00 = (X-m)/std
    
    C = (1/N)*np.matmul(X.T,X)
    C0 = (1/N)*np.matmul(X0.T,X0)  # np.cov(X.T,bias=True)와 동일
    C00 = (1/N)*np.matmul(X00.T,X00)
    
    
    print('C: ', C)
    print('C0: ', C0)
    print('np.cov: ', np.cov(X.T,bias=True))
    print('C00: ', C00)
    
    print(np.linalg.eig(C))
    p,v = np.linalg.eig(C0) # pca.fit_transform(X) 결과와 일치
    print(p,v)   
    print(np.linalg.eig(C00))
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    print("eigenvectors(행벡터): ",pca.components_)
    
    print('Done')

def PCA1():
    # 2차원 data의 main 축 찾기

    theta = np.arctan(1.5)   # np.pi/4
    N=5000

    X = np.array([np.random.normal(0,4,N),np.random.normal(0,1,N)])   # (2, N)
    X = np.matmul([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],X).T   # X를 회전...
    
    m = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    X0 = X-m
    print("mean: ", m)
    plt.scatter(X[:,0],X[:,1],s=1)
    plt.axis([-10, 10, -10, 10])
    #plt.show()
    

    C = (1/N)*np.matmul(X.T,X)  # normalize가 되지 않았기 때문에, 정확한 covariance matrix가 아니다.
    C0 = (1/N)*np.matmul(X0.T,X0)  # np.cov(X.T,bias=True)와 동일  --> 이것이 covariance matrix!!!
    
    p,v = np.linalg.eig(C0)  # v의 column vector가 eigenvector
    
    q=np.argsort(-p)
    w = v[:,q]
    
    
    Y=np.matmul(X0,w)  # 각 column별로 부호 차이가 날 수 있다. 감안하면 principalComponents와 일치.
    
    plt.scatter(Y[:,0],Y[:,1],s=1)
    plt.axis([-10, 10, -10, 10])
    plt.show()
    
    
    ##### sklearn의 PCA
    pca = PCA(n_components=2)  # 뽑아낼 component 갯수 지정
    #pca = PCA(0.8)  # 80%까지의 component를 추출
    
    # fit을 통홰, normalization(평균 차감)
    principalComponents = pca.fit_transform(X)  # 저장된 Scaler로 scale이 조정된다. pac.mean_
    
    # pca.explained_variance_ ---> p와 동일
    # pca.components_  ---> 행벡터가 eigenvector
    print(pca.explained_variance_ratio_)  # p/np.sum(p)와 동일
    print(pca.explained_variance_)  # p*N/(N-1)
    
    
    print('Done')


def PCA2():
    #from sklearn import datasets   # iris = datasets.load_iris()
    from mpl_toolkits.mplot3d import Axes3D

    labels=['setosa','versicolor','virginica']
    csv_filenamme = r"D:\hccho\CommonDataset\iris\iris_training.csv"
    data = pd.read_csv(csv_filenamme,header=None,skiprows=1,names=['feaure1','feaure2','feaure3','features4','label'])
    
    print(data.head(5))

    y = data['label'] # genre variable.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels


    #### NORMALIZE X ####
    cols = X.columns



    #### PCA 3 COMPONENTS ####
    n_components = 2
    
    if n_components==2:

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    
    
        # concatenate with target label
        finalDf = pd.concat([principalDf, y], axis = 1)
        
        print(pca.explained_variance_ratio_)
    
        fig = plt.figure(figsize = (8, 4))
        sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7, s = 100);
        
        plt.title('PCA on Genres', fontsize = 25)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 10);
        plt.xlabel("Principal Component 1", fontsize = 15)
        plt.ylabel("Principal Component 2", fontsize = 15)
        
        
    else: 
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3'])
    
    
        # concatenate with target label
        finalDf = pd.concat([principalDf, y], axis = 1)
        
        print(pca.explained_variance_ratio_)
    
        fig = plt.figure(figsize = (8, 4))
        ax = Axes3D(fig)
    
    
        for grp_name, grp_idx in finalDf.groupby('label').groups.items():
            y = finalDf.iloc[grp_idx,1]
            x = finalDf.iloc[grp_idx,0]
            z = finalDf.iloc[grp_idx,2]
            ax.scatter(x,y,z, label=labels[grp_name])  # this way you can control color/marker/size of each group freely
        ax.legend()
        plt.title('PCA on IRIS', fontsize = 10)
        plt.xticks(fontsize = 7)
        plt.yticks(fontsize = 5);
        ax.set_xlabel("Principal Component 1", fontsize = 7)
        ax.set_ylabel("Principal Component 2", fontsize = 7)
        ax.set_zlabel("Principal Component 3", fontsize = 7)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    #covariance_test()
    PCA1()
    #PCA2()

