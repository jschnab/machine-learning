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
    def __init__(self, alpha=0.001, n_iter=10000, regularization=False, lamb=1, intercept=True, norm=False, map_feat=False, feature_degree=4, plot_boundary=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.regularization = regularization
        self.lamb = lamb
        self.intercept = intercept
        self.norm = norm
        self.map_feat = map_feat
        self.plot_boundary = plot_boundary
        self.degree = feature_degree

    def normalize(self, X):
        """Return normalized matrix."""
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma

    def map_feature(self, X, degree):
        """Feature mapping to polynomial features."""
        X1 = X[:, 0]
        X2 = X[:, 1]
        mapped = np.copy(X)
        for i in range(3, self.degree + 1):
            for j in range(i + 1):
                X1_pow = np.power(X1, i - j)[:, np.newaxis]
                X2_pow = np.power(X2, j)[:, np.newaxis]
                mapped = np.concatenate((mapped, X1_pow * X2_pow), axis=1)
        return mapped

    def add_intercept(self, X):
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

    def accuracy(self, X, Y, theta):
        """Return prediction accuracy of the logistic model."""
        if X.shape[1] == theta.shape[0] - 1:
            X = self.add_intercept(X)
        p = np.round(self._s(np.dot(X, theta)))
        return np.mean((p == Y).astype(int))

    def _plot_boundary(self, X, Y, theta):
        """Plot data along with decision boundary."""
        #if we have a maximum of two features (excluding the intercept)
        if X.shape[1] <= 3:
            x_boundary = [np.min(X[:, 1]), np.max(X[:, 1])]
            y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
            #filter for target
            is_target = (Y == 1).flatten()
            #plot data with decision boundary line
            fig, ax = plt.subplots()
            ax.scatter(X[:, 1][is_target], X[:, 2][is_target], label="target", color="C0")
            ax.scatter(X[:, 1][is_target == 0], X[:, 2][is_target == 0], label="others", color="C1")
            ax.plot(x_boundary, y_boundary, color="C3", label="Decision boundary")
            ax.legend()
            plt.show()

        #if we have more than 3 features
        else:
            #grid range
            u = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
            v = np.linspace(min(X[:, 2]), max(X[:, 2]), 100)

            z = np.zeros((len(u), len(v)))
            #calculate z = theta*X over the grid
            for i in range(1, len(u)):
                for j in range(1, len(v)):
                    UV = np.concatenate((u[i].reshape((1, 1)), v[j].reshape((1, 1))), axis=1)
                    UV_mapped = self.add_intercept(self.map_feature(UV, self.degree))
                    z[i, j] = np.dot(UV_mapped, self.theta)

            #plot data and contour line
            #create filter for target
            is_target = (Y == 1).flatten()
            #plot
            fig, ax = plt.subplots()
            ax.scatter(X[:, 1][is_target], X[:, 2][is_target], label="target", color="C2")
            ax.scatter(X[:, 1][is_target == 0], X[:, 2][is_target == 0], label="others", color="grey")
            ax.contour(u, v, z.T, 0, colors="C3")
            ax.set_xlim(np.min(X[:, 1]) - 0.2, np.max(X[:, 1]) + 0.2)
            ax.set_ylim(np.min(X[:, 2]) - 0.2, np.max(X[:, 2]) + 0.2)
            ax.legend()
            plt.show()

    def fit(self, X, Y):
        """Return theta parameters, cost history and accuracy of model for logistic regression 
determined by gradient descent, and history of cost values."""
 
        #normalize X
        if self.norm:
            X = self.normalize(X)
        #map features to polynomial
        if self.map_feat:
            X = self.map_feature(X, self.degree)
        #add intercept to X
        if self.intercept:
            X = self.add_intercept(X)
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

        #plot results
        if self.plot_boundary:
            self._plot_boundary(X, Y, self.theta)

        #calculate accuracy
        accu = self.accuracy(X, Y, self.theta)

        return self.theta, cost_history, accu

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
    model = LogisticRegression(alpha=0.00001, n_iter=10000, plot_boundary=True)
    theta, cost_history, accuracy = model.fit(X, Y)

    model.plot_history(cost_history)

    #dot the same analysis with feature mapping to polynomials for color intensity
    #and total phenols
    #problems: overflow in exp in sigmoid function calculation and divid by zero in log of J 

    model = LogisticRegression(alpha=0.01, n_iter=30000, lamb=0.01, regularization=True, norm=True, map_feat=True, plot_boundary=True)
    ##XX = wine.data[:, [5, 9]]
    XX = wine.data[:, [5, 9]]

    #convert wine id from 0, 1 and 2 to 1, 0 and 0 (wine A is the positive target)
    YY = (wine.target[:, np.newaxis] == 0).astype(int)
    theta2, history2, accuracy2 = model.fit(XX, YY)

    model.plot_history(history2)
