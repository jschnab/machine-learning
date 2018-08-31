#this script defines a class which contains methods to perform
#a logistic regression by gradient descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

class LogisticRegression(object):
    """This class provides methods to perform logistic regression on data.
alpha          : learning rate
n_iter         : number of iteration of gradient descent
regularization : set to True if input should be regularized to avoid overfitting
lamb           : lambda parameter for regularization"""
    def __init__(self, alpha=0.001, n_iter=10000, regularization=False, lamb=1, intercept=True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.regularization = regularization
        self.lamb = lamb
        self.intercept = intercept

    def _add_intercept(self, X):
        """Add column of ones to X for vectorized calculations."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _s(self, z):
        """Return result of sigmoid function with z as input."""
        return 1 / (1 + np.exp(-z))

    def _cost(self, X, Y, theta):
        """Return cost of logistic regression."""
        #hypothesis function
        h = self._s(np.dot(X, theta))
        #number of training examples
        m = X.shape[0]
        #cost
        J = (np.dot(-Y.T, np.log(h)) - np.dot((1 - Y).T, np.log(1 - h))) / m
        return J

    def _gradient(self, X, Y, theta):
        """Return gradient of cost of logistic regression."""
        m = X.shape[0]
        h = self._s(np.dot(X, theta))
        return np.dot(X.T, h - Y) / m

    def fit(self, X, Y):
        """Return theta parameters for logistic regression determined by gradient
descent, and history of cost values."""
        #add intercept to X
        if self.intercept:
            X = self._add_intercept(X)
        #initialize theta
        self.theta = np.zeros((X.shape[1], 1))
        #define array to store cost history
        cost_history = np.zeros(self.n_iter)
        #loop for number of iterations to perform gradient descent
        for i in range(self.n_iter):
            #calculate gradient
            gradient = self.alpha * self._gradient(X, Y, self.theta)
            #if regularization is chosen, calculate regularization terms
            #and add them to cost and gradient
            cost_reg = 0
            if self.regularization:
                theta_copy = np.copy(self.theta)
                theta_copy[0] = 0 #regularization will not happen for this
                cost_reg += (self.lamb / 2 / X.shape[0]) * np.dot(theta_copy.T, theta_copy)
                gradient_reg = self.lamb * theta_copy / X.shape[0]
                gradient += gradient_reg
            #update theta
            self.theta -= gradient
            #add cost to history
            cost_history[i] = self._cost(X, Y, self.theta) + cost_reg
            #spy to monitor how cost evolves
            if i % 100 == 0:
                print(cost_history[i])
        return gradient, cost_history

    def accuracy(self, X, Y, theta):
        """Return prediction accuracy of the logistic model."""
        X = self._add_intercept(X)
        p = np.round(self._s(np.dot(X, self.theta)))
        return np.mean((p == Y).astype(int))

    def map_feature(self, X1, X2, degree=6):
        """Feature mapping to polynomial features."""
        mapped = np.ones((X1.shape[0], 1))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                X1_pow = np.power(X1, i - j)[:, np.newaxis]
                X2_pow = np.power(X2, j)[:, np.newaxis]
                mapped = np.concatenate((mapped, X1_pow * X2_pow), axis=1)
        return mapped

    def plot_history(self, cost_history):
        """Plot cost history of logistic regression."""
        x_val = [i for i in range(len(cost_history))]
        fig, ax = plt.subplots()
        ax.plot(x_val, cost_history)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost history of logistic regression")
        plt.show()

if __name__ == "__main__":
    #do logistic regression on wine data set
    wine = datasets.load_wine()

    #extract the data from the data set
    X = wine.data[:, [4, 12]][wine.target != 2]
    Y = (wine.target[wine.target != 2])[:, np.newaxis]

    #fit data with logistic regression
    model = LogisticRegression(alpha=0.00001, n_iter=10000)
    theta, cost_history = model.fit(X, Y)

    #plot data and decision boundary
    #we need only two points for a linear decision boundary
    x_boundary = [np.min(X[:, 0]), np.max(X[:, 0])]
    y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
    #make filters to filter X by wine
    is_A = (Y == 0).flatten()
    is_B = (Y == 1).flatten()
    #plot data
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0][is_A], X[:, 1][is_A], label="wine A", color="C1")
    ax.scatter(X[:, 0][is_B], X[:, 1][is_B], label="wine B", color="C2")
    ax.plot(x_boundary, y_boundary, color="C3", label="Decision boundary")
    ax.set_xlabel("Magnesium")
    ax.set_xlabel("Proline")
    ax.set_title("Classification of wines based on magnesium and proline content")
    ax.legend()
    plt.show()
    model.plot_history(cost_history)

    #dot the same analysis with feature mapping to polynomials for color intensity
    #and total phenols
#problems: overflow in exp in sigmoid function calculation and divid by zero in log of J 
    #intercept is False because function map_feature adds it
    #model = LogisticRegression(alpha=0.000001, n_iter=10000, lamb=100, regularization=True, intercept=False)
    #XX = wine.data[:, [5, 9]]
    #add polynomial features to XX
    #XX = model.map_feature(XX[:, 0], XX[:, 1])
    #convert wine id from 0, 1 and 2 to 1, 0 and 0 (wine A is the positive target)
    #YY = (wine.target[:, np.newaxis] == 0).astype(int)
    #theta2, history2 = model.fit(XX, YY)
    #plot data and decision boundary
    #create filter for target
    #is_target = (YY == 1).flatten()
    #fig, ax = plt.subplots()
    #ax.scatter(XX[:, 1][is_target], XX[:, 2][is_target], label="target wine", color="C0")
    #ax.scatter(XX[:, 1][is_target == 0], XX[:, 2][is_target == 0], label="other wines", color="grey")
    #ax.set_xlabel("Total phenols")
    #ax.set_ylabel("Color intensity")
    #ax.legend()
    #plt.show()
