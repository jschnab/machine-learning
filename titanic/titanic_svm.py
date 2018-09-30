#script to perform prediction of titanic passenger survival
#with a support vector machine algorithm

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

#import data
train = pd.read_csv("train_feat_engin.csv", header=None)
X = train.iloc[:, 1:]
Y = train.iloc[:, 0]
test = pd.read_csv("test_feat_engin.csv", header=None)
test_original = pd.read_csv("test.csv") #to get passenger IDs

#split train data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

#perform grid search to find best C and gamma parameters
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
kernels = ['linear', 'rbf', 'sigmoid']
#degrees = [2, 3, 4, 5, 6]

#variables to store results
best_score = 0
best_C = None
best_gamma = None
best_kernel = None
best_degree = None

#loop over grid
def grid_search():
    for C in C_values:
        for gamma in gamma_values:
            for kernel in kernels:
                svc = svm.SVC(C=C, gamma=gamma, kernel=kernel)
                svc.fit(X_train, Y_train)
                score = svc.score(X_test, Y_test)

                if score > best_score:
                    best_score = score
                    best_C = C
                    best_gamma = gamma
                    best_kernel = kernel

    print("Best parameters give {0:.4%} accuracy".format(best_score))
    print("C={0}\ngamma={1}\nKernel={2}\nDegree={3}".format(best_C, best_gamma,\
        best_kernel, best_degree))
#results are very unstable but best results seem to be obtained with
#C=10 to 100 and gamma=0.01 to 0.03

#predict survivors on test dataset
svc = svm.SVC(C=100, gamma=0.03, probability=True)
svc.fit(X, Y)
prediction = svc.predict(test)[:, np.newaxis]
prediction = np.hstack((test_original['PassengerId'][:, np.newaxis], prediction))
np.savetxt("titanic_pred_svm.csv", prediction, delimiter=',', fmt='%d')
