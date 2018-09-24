#this script performs classification of titanic passengers by logistic regression

import sys
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

sys.path.append("E:\\Git\\machine-learning")
from logistic_reg import LogisticRegression

#import data
data = pd.read_csv("train.csv")

#convert 'Sex' column to 1 and 0 for male and female
data['Sex'] = (data['Sex'] == 'male').astype(int)

#subset of data
titanic = pd.DataFrame({'class':data['Pclass'],
    'sex':data['Sex'],
    'age':data['Age'],
    'sibling':data['SibSp'],
    'parent':data['Parch'],
    'fare':data['Fare'],
    'survived':data['Survived']})

#remove NaN
titanic_clean = titanic.dropna()

#create input and output matrices
X = np.concatenate((titanic_clean['class'][:, np.newaxis],
    titanic_clean['sex'][:, np.newaxis],
    titanic_clean['age'][:, np.newaxis],
    titanic_clean['sibling'][:, np.newaxis], 
    titanic_clean['parent'][:, np.newaxis], 
    titanic_clean['fare'][:, np.newaxis]), axis=1)
Y = titanic_clean['survived'][:, np.newaxis]

#perform logistic regression
model = LogisticRegression(alpha=0.001, n_iter=60000, norm=True)
theta, history, accuracy = model.fit(X, Y)
model.plot_history(history)
print(accuracy)

#import test data
test = pd.read_csv("test.csv")

#recode 'Sex' to 0 and 1
test_subset = pd.DataFrame({'class':test['Pclass'],
    'sex':(test['Sex'] == 'male').astype(int),
    'age':test['Age'],
    'sibling':test['SibSp'],
    'parent':test['Parch'],
    'fare':test['Fare']})

#prepare input data: normalize data and add intercept
test_norm = model.normalize(test_subset)
test_norm = model.add_intercept(test_norm)

#predict survival of test examples
prediction = np.round(model._s(-np.dot(test_norm, theta)))

#if output is NaN, predict no survival
prediction = np.where(np.isnan(prediction), 0, prediction)

#prediction in column with passenger ID
prediction = np.hstack((test['PassengerId'][:, np.newaxis], prediction))

#convert output float to integer
prediction = prediction.astype(int)

#save data as csv file
np.savetxt("titanic_pred.csv", prediction, delimiter=',', fmt='%d')
