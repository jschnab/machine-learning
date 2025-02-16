#script based on coursera Andrew Ng's ML course exercice 7 on SVM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io #load Octave .mat file
import scipy.optimize #fmin_cg
from sklearn import svm

#load Octave matrix
data = scipy.io.loadmat("ex6data1.mat")
X = data['X']
Y = data['y']

#put data in a dataframe
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Y'] = Y

#filters for data based on Y
positive = df['Y'] == 1
negative = df['Y'] == 0

#plot data
fig, ax = plt.subplots()
ax.scatter(df['X1'][positive], df['X2'][positive], color='C0', \
        label='positive')
ax.scatter(df['X1'][negative], df['X2'][negative], color='C1',\
        label='negative')
legend = ax.legend(loc='lower right', frameon=True)
legend.get_frame().set_facecolor('white')
plt.show()

#support vector machine classifier model
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)

#train the model
svc.fit(df[['X1', 'X2']], df['Y'])
score = svc.score(df[['X1', 'X2']], df['Y'])
print("Score of linear SVM with C=1: {:.3f}".format(score))

#add confidence level to the dataframe and plot data
#we can see that one point is misclassified
df['svm1_conf'] = svc.decision_function(df[['X1', 'X2']])
fig, ax = plt.subplots()
cs = ax.scatter(df['X1'], df['X2'], c=df['svm1_conf'],\
        cmap=plt.cm.get_cmap('RdBu_r', 10))
ax.set_title('SVM (C=1) decision confidence')
fig.colorbar(cs)
plt.show()

#train a new model with a larger value of c
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(df[['X1', 'X2']], df['Y'])
score = svc2.score(df[['X1', 'X2']], df['Y'])
df['svm2_conf'] = svc2.decision_function(df[['X1', 'X2']])
print('Score of linear SVM with C=100: {:.3f}'.format(score))
fig, ax = plt.subplots()
cs = ax.scatter(df['X1'], df['X2'], c=df['svm2_conf'], \
        cmap=plt.cm.get_cmap('RdBu_r', 10))
ax.set_title('SVM (C=100) decision confidence')
fig.colorbar(cs)
plt.show()

#implement SVM classification with gaussian kernel

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
print("Gaussian kernel result: {:.3f}".format(gaussian_kernel(x1, x2, sigma)))

#import new data
data = scipy.io.loadmat("ex6data2.mat")
df = pd.DataFrame(data['X'], columns=['X1', 'X2'])
df['Y'] = data['y']

#create filters for data
positive = df['Y'] == 1
negative = df['Y'] == 0

#plot new data
fig, ax = plt.subplots()
ax.scatter(df['X1'][positive], df['X2'][positive], c='C0', label='positive')
ax.scatter(df['X1'][negative], df['X2'][negative], c='C1', label='negative')
ax.legend()
plt.show()

#definition and training of SVM with RBF (radial basis function) kernel
svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(df[['X1', 'X2']], df['Y'])
score = svc.score(df[['X1', 'X2']], df['Y'])
print("Score of SVM with RBF kernel: {:.3f}".format(score))

#add the probability of the sample for each class and plot it
df['probability'] = svc.predict_proba(df[['X1', 'X2']])[:, 0]
fig, ax = plt.subplots()
cs = ax.scatter(df['X1'], df['X2'], c=df['probability'], cmap='RdBu_r')
fig.colorbar(cs)
plt.show()

#the next part of the exercice aims at finding optimal parameters for an
#SVM model with a grid search
data = scipy.io.loadmat("ex6data3.mat")
X = data['X']
Xval = data['Xval']
Y  = data['y'].ravel()
Yval = data['yval'].ravel()

#define the values to test
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

#variables to store the results
best_score = 0
best_param = {'C':None, 'gamma':None}

#grid search to find best parameters
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, Y)
        score = svc.score(Xval, Yval)

        if score > best_score:
            best_score = score
            best_param['C'] = C
            best_param['gamma'] = gamma

print("Best parameters:\nC = {0}\ngamma = {1}\nScore = {2:.3f}".format(\
        best_param['C'], best_param['gamma'], score))

#the final part of the exercise is to build a spam filter
#load data
spam_train = scipy.io.loadmat("spamTrain.mat")
spam_test = scipy.io.loadmat("spamTest.mat")
X = spam_train['X']
Xtest = spam_test['Xtest']
Y = spam_train['y'].ravel()
Ytest = spam_test['ytest'].ravel()

#train the classifier
svc = svm.SVC()
svc.fit(X, Y)

#show accuracy
print("Training accuracy: {:.3f}".format(svc.score(X, Y)))
print("Test accuracy: {:.3f}".format(svc.score(Xtest, Ytest)))
