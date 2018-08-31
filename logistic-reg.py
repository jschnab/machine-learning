#this script performs a logistic regression on the iris data set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
from scipy.optimize import fmin_bfgs

#import data from csv file
data = pd.read_csv("iris.csv")
#remove first column, irrelevant index
iris = data[data.columns[1:]]

#plot data
fig, ax = plt.subplots()
for species in set(iris["Species"]):
    ax.scatter(iris["Petal.Length"][iris["Species"]==species],
    iris["Petal.Width"][iris["Species"]==species], label=species)
ax.set_xlabel("Petal length")
ax.set_ylabel("Petal width")
ax.legend()
plt.show()

#put data in arrays for logistic regression
#X contains petal length and width
X = np.ones((iris.shape[0], 3))
X[:, 1] = iris["Petal.Length"]
X[:, 2] = iris["Petal.Width"]
m, n = X.shape
#y contains species name
y = np.array(iris["Species"])[:, np.newaxis]

#first we classify species setosa agains the two other species
y_setosa = (y=="setosa").astype(float)

#initialize theta
initial_theta = np.zeros((n, 1))

#define sigmoid function
def s(z):
    """Return result of sigmoid function for opposite of input z"""
    return 1 / (1 + np.exp(-z))

#define cost function
def cost_logistic(theta, X, y):
    """Return cost logistic regression"""
    #number of training examples
    m = X.shape[0]
    #hypothesis function
    h = np.dot(X, theta)
    #cost
    J = (np.dot(-y.T, np.log(s(h))) - np.dot((1 - y).T, np.log(1 - s(h)))) / m
    return J

def grad_logistic(theta, X, y):
    m = X.shape[0]
    h = np.dot(X, theta)
    #gradient
    return np.dot(X.T, s(h) - y) / m

#define functions for fmin_bfgs
def f(theta):
    return np.ndarray.flatten(cost_logistic(theta, X, y_setosa))

def fprime(theta):
    return np.ndarray.flatten(grad_logistic(theta, X, y_setosa))
print(f(initial_theta))
print(fprime(initial_theta))
#results = fmin_bfgs(f, initial_theta, fprime, disp=True, maxiter=400, full_output=True, retall=True)
#theta = results[0]
#cost = results[1]
