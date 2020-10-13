

'''
Data: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification   ---> 여기서 다운로드

https://www.kaggle.com/andradaolteanu/work-w-audio-data-visualise-classify-recommend
https://www.kaggle.com/manoramamaharana/music-genre-classification-18bce2516


xgboost gpu버전 설치:
https://medium.com/@SeoJaeDuk/how-to-install-xgboost-gpu-no-gpu-on-window-10-x64-for-python-3-8c2ffa00fb29

'''

# Usual Libraries
import pandas as pd   # pip install pandas
import numpy as np
import seaborn as sns # pip install seaborn 
import matplotlib.pyplot as plt

import sklearn  # pip install scikit-learn

# Librosa (the mother of audio files)
import librosa
import librosa.display
import warnings
import os
import pickle
warnings.filterwarnings('ignore')


general_path = r'D:\SpeechRecognition\DeepLearningForAudioWithPython-musikalkemist\Data'
print(list(os.listdir(f'{general_path}/genres_original/')))

def EDA():
    data = pd.read_csv(f'{general_path}/features_30_sec.csv')
    print(data.head())
    
    
    
    
    # Computing the Correlation Matrix
    spike_cols = [col for col in data.columns if 'mean' in col]
    corr = data[spike_cols].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 9));
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 25)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.tight_layout()
    plt.savefig("Corr Heatmap.jpg")
    plt.show()


def PCA_analysis():
    from sklearn.decomposition import PCA
    from sklearn import preprocessing

    data = pd.read_csv(f'{general_path}/features_30_sec.csv')
    data = data.iloc[0:, 1:] 
    print(data.head(5))
    
    y = data['label'] # genre variable.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels


    #### NORMALIZE X ####
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)


    #### PCA 2 COMPONENTS ####
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


    # concatenate with target label
    finalDf = pd.concat([principalDf, y], axis = 1)
    
    print(pca.explained_variance_ratio_)

    plt.figure(figsize = (16, 9))
    sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7, s = 100);
    
    plt.title('PCA on Genres', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Principal Component 1", fontsize = 15)
    plt.ylabel("Principal Component 2", fontsize = 15)
    plt.savefig("PCA Scattert.jpg")
    plt.show()
    
    

def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')


def xgboost_train():

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier   # GPU 지원 안 함
    from xgboost import XGBClassifier, XGBRFClassifier   # pip install xgboost
    from xgboost import plot_tree, plot_importance
    
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import RFE





    data = pd.read_csv(f'{general_path}/features_3_sec.csv')
    data = data.iloc[0:, 1:] 
    print(data.head(5))
    
    y = data['label'] # genre variable.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
    
    #### NORMALIZE X ####
    
    # Normalize so everything is on the same scale. 
    
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)   # return numpy array (9990,58)
    
    # new data frame with the new scaled data. 
    X = pd.DataFrame(np_scaled, columns = cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # Frame in Frame out
    
    # Final model
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    
    
    preds = xgb.predict(X_test)   # array(['hiphop', 'jazz', 'blues', ....],dtype=object)
    
    print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')
    
    # Confusion Matrix
    confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'  ---> confusion_matr: numpy array
    plt.figure(figsize = (14, 7))
    sns.heatmap(confusion_matr, cmap="Blues", annot=True, 
                xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
               yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
    plt.savefig("conf matrix")
    plt.show()
    
    
    
    #### model save ######
    #### 그냥 pickle로 object 저장
     
    pickle.dump(xgb, open("my_xgb_model.pkl", "wb"))
    
    

    
    
    
    print('Done')

def load_and_feature_analysis():
    
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
    import xgboost
    
    ################### eli5
    import eli5  # pip install eli5
    from eli5.sklearn import PermutationImportance
    
    data = pd.read_csv(f'{general_path}/features_3_sec.csv')
    data = data.iloc[0:, 1:] 
    print(data.head(5))
    
    y = data['label'] # genre variable.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
    
    #### NORMALIZE X ####
    
    # Normalize so everything is on the same scale. 
    
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)   # return numpy array (9990,58)
    
    # new data frame with the new scaled data. 
    X = pd.DataFrame(np_scaled, columns = cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # Frame in Frame out
    
    ########## model load ##############
    xgb_classifier = pickle.load(open("my_xgb_model.pkl", "rb"))
    preds = xgb_classifier.predict(X_test)   # array(['hiphop', 'jazz', 'blues', ....],dtype=object)
    
    print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')    
    
    
    # feature F score. pandas dataframe으로 train했기 때문에, featrue이름이 표시된다. numpy array로 data를 넣었다면, feature이름이 표시되지 않는다.
    xgboost.plot_importance(xgb_classifier)
    plt.show()
    
    
    #######   eli5 PermutationImportance  #################
    perm = PermutationImportance(estimator=xgb_classifier, random_state=1)
    perm.fit(X_test, y_test)
    
    # return  되는 값은 정확도의 변화이다. 한번만 simulation하는 것이 아니므로, +/-가 있다.
    weights = eli5.show_weights(estimator=perm, feature_names = X_test.columns.tolist())    #### weights.data가 string인데, 내용은 html형식.
    with open('weights.htm','wb') as f:
        f.write(weights.data.encode("UTF-8"))




def lightgbm_test():
    import lightgbm as lgb
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
    
    
    data = pd.read_csv(f'{general_path}/features_3_sec.csv')
    data = data.iloc[0:, 1:] 
    print(data.head(5))
    
    y = data['label'] # genre variable.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
    
    #### NORMALIZE X ####
    
    # Normalize so everything is on the same scale. 
    
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)   # return numpy array (9990,58)
    
    # new data frame with the new scaled data. 
    X = pd.DataFrame(np_scaled, columns = cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # Frame in Frame out

    
    #boost_from_average=True  ==> 평균에서 시작. data가 뷸균형할 경우 False로 주는 것이 좋다.
    lgb_classifier = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05)  # n_estimators = 1000, 2000  ==>0.92192, 0.92392
    lgb_classifier.fit(X_train, y_train)
    
    
    preds = lgb_classifier.predict(X_test)
    
    print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')



    lgb.plot_importance(lgb_classifier,max_num_features=10)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #EDA()
    #PCA_analysis()
    #xgboost_train()
    #load_and_feature_analysis()
    lightgbm_test()
    
    print('Done')



