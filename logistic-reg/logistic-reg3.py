#this script performs a logistic regression with one feature on the wine
#data set from sklearn
#the plot suggests an easy classification but the gradient descent has
#trouble converging, and parameters come out wrong

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

wine = datasets.load_wine()

#let's plot wine id for wines 0 and 1 against proline
#first we extract the data from the data set
proline_all = wine.data[:, wine.feature_names.index('proline')]
#generate filters by target
is_target0 = wine.target == 0
is_target1 = wine.target == 1
targets = [is_target0, is_target1]
#use filters to get certain rows only
proline_0 = proline_all[is_target0]
proline_1 = proline_all[is_target1]
#put data in arrays
X = np.concatenate((proline_1, proline_0), axis=0)[:, np.newaxis]
Y = np.concatenate((np.zeros((len(proline_1), 1)), np.ones((len(proline_0), 1))), axis=0)
#plot data
fig, ax = plt.subplots()
ax.scatter(X[:len(proline_1), 0], Y[:len(proline_1), 0], color="C0", alpha=0.3, label="wine 1")
ax.scatter(X[len(proline_1):, 0], Y[len(proline_1):, 0], color="C1", alpha=0.3, label="wine 0")
ax.set_xlabel("Proline")
ax.set_ylabel("Wine identity")
ax.legend()
plt.show()

#add column of ones to X
m = X.shape[0]
X_ones = np.hstack((np.ones((m, 1)), X))
n = X_ones.shape[1]
#initialize theta
i_theta = np.zeros((n, 1))

#sigmoid function
def s(z):
    """Return result of sigmoid function for opposite of input z."""
    return 1 / (1 + np.exp(-z))

def cost_logistic(X, Y, theta):
    """Return cost logistic regression."""
    #hypothesis function
    h = s(np.dot(X, theta))
    #cost
    J = (np.dot(-Y.T, np.log(h)) - np.dot((1 - Y).T, np.log(1 - h))) / m
    return J

def grad_logistic(X, Y, theta):
    """Return gradient of cost of logistic regression."""
    h = s(np.dot(X, theta))
    #gradient
    return np.dot(X.T, h - Y) / m

def gradient_descent(X, Y, theta, alpha, n_iter):
    """Return parameters for logistic regression as determined by gradient
descent, and history of cost values."""
    #defin array to store cost history
    cost_history = np.zeros(n_iter)
    #loop for n_iter to perform gradient descent
    for i in range(n_iter):
        delta = grad_logistic(X, Y, theta)
        theta -= alpha * delta
        cost_history[i] = cost_logistic(X, Y, theta)
    return theta, cost_history

#set parameters of gradient descent and run it
alpha = 0.00003
n_iter = 100000
theta, history = gradient_descent(X_ones, Y, i_theta, alpha, n_iter)

print("Theta :", theta)

#plot cost history
fig, ax = plt.subplots()
x_val = [i for i in range(n_iter)]
ax.plot(x_val, history)
ax.set_xlabel("Iteration number")
ax.set_ylabel("Cost")
plt.show()

#calculate accuracy on training set
p = np.round(s(np.dot(X_ones, theta)))
print("Training accuracy :", np.mean((p == Y).astype(int)))

fig, ax = plt.subplots()
x_val = np.array([i for i in range(int(np.min(X)), int(np.max(X)))])
x_val = np.hstack((np.ones((len(x_val), 1)), x_val[:, np.newaxis]))
y_val = s(np.dot(x_val, theta))
ax.plot(x_val, y_val, color='black', label="Boundary decision")
ax.scatter(X[:len(proline_1), 0], Y[:len(proline_1), 0], color="C0", alpha=0.3, label="wine 1")
ax.scatter(X[len(proline_1):, 0], Y[len(proline_1):, 0], color="C1", alpha=0.3, label="wine 0")
ax.set_xlabel("Proline")
ax.set_ylabel("Wine identity")
ax.legend()
plt.show()


