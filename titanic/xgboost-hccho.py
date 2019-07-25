# https://www.kaggle.com/datacanary/xgboost-example-python/

# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?


"""
- train, test data를 합쳐서, missing 처리 같은 전처리를 하고, 다시 분리한다.
- 결측값대체(imputation)
"""


import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
train_df = pd.read_csv('./input/train.csv', header=0)
test_df = pd.read_csv('./input/test.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])  # female, male을 각각 0,1로 바꾼다.

##### hccho 수정: one-hot 변환   ---> accuracy는 더 낮게 나온다. why???   female, male 2개 category라서 그런가?
# one_hot_encoded = OneHotEncoder().fit_transform(big_X_imputed[nonnumeric_columns]).toarray()
# big_X_imputed.drop(nonnumeric_columns, axis=1, inplace=True)
# big_X_imputed=big_X_imputed.join(pd.DataFrame(one_hot_encoded))
####

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].values
test_X = big_X_imputed[train_df.shape[0]::].values
train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.05).fit(train_X, train_y)


y_pred = gbm.predict(train_X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(train_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


predictions = gbm.predict(test_X)
# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)












