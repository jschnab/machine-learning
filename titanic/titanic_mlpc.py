# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 08:46:06 2018

@author: thejo
"""

#script which implements a multilayer perceptron classifier with sklearn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#import train data
data = pd.read_csv("train_feat_engin.csv", header=None)
X_df = data.iloc[0:data.shape[0], 1:data.shape[1]]
Y_df = data.iloc[0:data.shape[0], 0]

X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.25)

#import test data
test = pd.read_csv("test_feat_engin.csv", header=None)

#import non-engineered test data to get passenger Id
test_original = pd.read_csv("test.csv")

#train the model
mlp = MLPClassifier(hidden_layer_sizes=(14, 14, 14),
                    learning_rate_init=0.001,
                    max_iter=5000,
                    solver="adam")
mlp.fit(X_train, Y_train)

#prediction and evaluation
predictions = mlp.predict(X_test)
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

#save prediction results
#prediction in column with passenger ID
results = np.hstack((test_original['PassengerId'][:, np.newaxis], 
                        mlp.predict(test)[:, np.newaxis]))
    
#save data as csv file
np.savetxt("titanic_pred_nn.csv", results, delimiter=',', fmt='%d')