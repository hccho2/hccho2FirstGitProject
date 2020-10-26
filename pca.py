# coding: utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from scipy.fftpack import fft
import librosa
from sklearn.decomposition import PCA
import pandas as pd

def PCA1():
    # 2차원 data의 main 축 찾기

    theta = np.arctan(1.5)   # np.pi/4
    N=5000

    X = np.array([np.random.normal(0,4,N),np.random.normal(0,1,N)])   # (2, N)
    X = np.matmul([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],X).T   # X를 회전...
    
    
    np.mean(X,axis=0)
    plt.scatter(X[:,0],X[:,1],s=1)
    plt.axis([-10, 10, -10, 10])
    #plt.show()
    

    C = (1/N)*np.matmul(X.T,X)
    
    p,v = np.linalg.eig(C)  # column vector가 eigenvector
    
    q=np.argsort(-p)
    w = v[:,q]
    
    
    Y=np.matmul(w.T,X.T).T
    
    plt.scatter(Y[:,0],Y[:,1],s=1)
    plt.axis([-10, 10, -10, 10])
    plt.show()
    
    
    ##### sklearn의 PCA
    pca = PCA(n_components=2)  # 뽑아낼 component 갯수 지정
    #pca = PCA(0.8)  # 80%까지의 component를 추출
    
    # 먼저 fit을 하게 되면, normalization을 수행한다.
    pca.fit(X)  # StandardScaler의 내부적으로 저장해 둔다.
    principalComponents = pca.fit_transform(X)  # 저장된 Scaler로 scale이 조정된다.
    
    # pca.explained_variance_ ---> p와 동일
    # pca.components_  ---> 행벡터가 eigenvector
    print(pca.explained_variance_ratio_)  # p/np.sum(p)와 동일
    
    
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
    #PCA1()
    PCA2()


