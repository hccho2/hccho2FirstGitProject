
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')


# In[2]:


train_test_data = [train, test] # combining train and test dataset
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }




# 1. Title 처리
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Title 단순화
    dataset['Title'] =dataset['Title'].apply(lambda x: x if x in [ 'Mr', 'Miss', 'Mrs'] else 'etc')
    
    # Age Missing Data 처리
    dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    
    # Fare Missing을 Pclass 중간값으로
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    
    # Embarked Missing을 가장 많은 S로
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    # Cabin 정보를 첫번째 alphabet만
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    # test data에 T class가 없기 때문에 이렇게 해야, get_dummies가 잘 작동
    dataset['Cabin'] = dataset['Cabin'].astype('category',categories=["A", "B", "C", "D", "E", "F", "G", "T"])
    
    
    # Cabin Missing을 Pclass 빈도가 가장 많은 data로
    dataset['Cabin'] = dataset.groupby('Pclass').Cabin.transform(lambda x: x.fillna(x.mode()[0]))
    
    # 가족 data합치기
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    
    # 불필요한 자료 제거
    dataset.drop( ['Name','Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)

 
train = train.drop(['PassengerId'], axis=1)    

# one hot encoding
train = pd.get_dummies(train,columns=['Title', 'Sex','Pclass','Cabin','Embarked'])
test = pd.get_dummies(test,columns=['Title', 'Sex','Pclass','Cabin','Embarked'])





train_data = train.drop('Survived', axis=1)
target_data = train['Survived']



# In[3]:


train_data.head(7)


# In[4]:


inputs = train_data.values
targets = target_data.values.reshape(-1,1)


scaler = StandardScaler()
inputs_temp = scaler.fit_transform(inputs[:,:3] )
inputs=np.concatenate([inputs_temp,inputs[:,3:]],axis=1)
print(inputs.shape, targets.shape)


# In[179]:


# class MyTitanic():
#     def __init__(self,is_training=True,name=None):
#         self.name=name
#         self.is_training = is_training
        
#         self.build()
#     def build(self):
#         with tf.variable_scope(self.name):
#             self.X = tf.placeholder(tf.float32, shape=[None, 23])
#             self.Y = tf.placeholder(tf.float32, shape=[None,1])
        
            
#             x = tf.layers.dense(self.X,units=256,activation=tf.nn.relu) 
#             x = tf.layers.dropout(x,0.5,self.is_training)
#             x = tf.layers.dense(self.X,units=256,activation=tf.nn.relu)  
#             x = tf.layers.dropout(x,0.8,self.is_training)
#             x = tf.layers.dense(x,units=10,activation=tf.nn.relu)  
#             x = tf.layers.dense(x,units=10,activation=tf.nn.relu) 
#             logits = tf.layers.dense(x,units=1,activation=None)
           
#             self.predict = (tf.nn.sigmoid(logits) >=0.5)
#             self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y,logits=logits))
            
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)
            

# [32, dropout 0.5, 3, 1]  ==> test acc = 76.79
# [32, dropout 0.5, 3, 1]  epoch 5000 ==> test acc = 77.51
class MyTitanic():
    def __init__(self,is_training=True,name=None):
        self.name=name
        self.is_training = is_training
        
        self.build()
    def build(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape=[None, 23])
            self.Y = tf.placeholder(tf.float32, shape=[None,1])
        
            
            x = tf.layers.dense(self.X,units=32,activation=tf.nn.relu) 
            x = tf.layers.dropout(x,0.5,self.is_training)
            x = tf.layers.dense(x,units=3,activation=tf.nn.relu) 
            logits = tf.layers.dense(x,units=1,activation=None)
           
            self.predict = (tf.nn.sigmoid(logits) >=0.5)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y,logits=logits))
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)


# In[186]:


tf.reset_default_graph()
tf.set_random_seed(1234)


with tf.variable_scope('model') as scope:
    model = MyTitanic(is_training=True,name="titanic")
    
with tf.variable_scope('model',reuse=True) as scope:
    model_test = MyTitanic(is_training=False,name="titanic")

    
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(5000):
    sess.run(model.optimizer,feed_dict={model.X: inputs,model.Y: targets})
    if i% 100 ==0:
        loss = sess.run(model.loss,feed_dict={model.X: inputs,model.Y: targets})
        print('step: {}, loss = {:.4f}'.format(i, loss))
        
        
predict = sess.run(model.predict,feed_dict={model.X: inputs,model.Y: targets}).astype(np.int32)
acc = np.mean(1*(predict==targets))

print('train acc: ', acc)


# In[187]:


test_temp = test.drop(['PassengerId'], axis=1).values
test_temp = np.concatenate([scaler.transform( test_temp[:, :3]),test_temp[:,3:]],axis=1)

test_predict = sess.run(model_test.predict,feed_dict={model_test.X: test_temp}).astype(np.int32).reshape(-1)


# In[188]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_predict
    })

submission.to_csv('submission.csv', index=False)

