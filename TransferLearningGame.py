# cody by chenchiwei
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
import TrAdaBoost as tr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import numpy as np


def append_feature(dataframe, istest):
    lack_num = np.asarray(dataframe.isnull().sum(axis=1))
    # lack_num = np.asarray(dataframe..sum(axis=1))
    if istest:
        X = dataframe.values
        X = X[:, 1:X.shape[1]]
    else:
        X = dataframe.values
        X = X[:, 1:X.shape[1] - 1]
    total_S = np.sum(X, axis=1)
    var_S = np.var(X, axis=1)
    X = np.c_[X, total_S]
    X = np.c_[X, var_S]
    X = np.c_[X, lack_num]

    return X


train_df = pd.DataFrame(pd.read_csv("/Users/zoom/Documents/迁移学习/new-data/A_train.csv"))
train_df.fillna(value=-999999)
train_df1 = pd.DataFrame(pd.read_csv("/Users/zoom/Documents/迁移学习/new-data/B_train.csv"))
train_df1.fillna(value=-999999)
test_df = pd.DataFrame(pd.read_csv("/Users/zoom/Documents/迁移学习/new-data/B_test.csv"))
test_df.fillna(value=-999999)

train_data_T = train_df.values
train_data_S = train_df1.values
test_data_S = test_df.values

print 'data loaded.'

label_T = train_data_T[:, train_data_T.shape[1] - 1]
# trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
trans_T = append_feature(train_df, istest=False)

label_S = train_data_S[:, train_data_S.shape[1] - 1]
# trans_S = train_data_S[:, 1:train_data_S.shape[1] - 1]
trans_S = append_feature(train_df1, istest=False)

test_data_no = test_data_S[:, 0]
# test_data_S = test_data_S[:, 1:test_data_S.shape[1]]
test_data_S = append_feature(test_df, istest=True)

print 'data split end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape

# # 加上和、方差、缺失值数量的特征，效果有所提升
# trans_T = append_feature(trans_T, train_df)
# trans_S = append_feature(trans_S, train_df1)
# test_data_S = append_feature(test_data_S, test_df)
#
# print 'append feature end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape

imputer_T = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_S = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# imputer_T.fit(trans_T,label_T)
imputer_S.fit(trans_S, label_S)

trans_T = imputer_S.transform(trans_T)
trans_S = imputer_S.transform(trans_S)

test_data_S = imputer_S.transform(test_data_S)

# pca_T = decomposition.PCA(n_components=50)
# pca_S = decomposition.PCA(n_components=50)
#
# trans_T = pca_T.fit_transform(trans_T)
# trans_S = pca_S.fit_transform(trans_S)
# test_data_S = pca_S.transform(test_data_S)

print 'data preprocessed.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape

X_train, X_test, y_train, y_test = model_selection.train_test_split(trans_S, label_S, test_size=0.33, random_state=42)

# feature scale
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train, y_train)
# X_test = scaler.transform(X_test)
# print 'feature scaled end.'

pred = tr.tradaboost(X_train, trans_T, y_train, label_T, X_test, 10)
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=pred, pos_label=1)
print 'auc:', metrics.auc(fpr, tpr)
