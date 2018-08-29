import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

wine = datasets.load_wine()

#let's have a look at the descriptor of the dataset
#print(wine.DESCR)
#among the differents variables, magnesium and proline have the highest
#variability, so they are good candidates for target discrimination

#let's plot magnesium VS proline
#first we extract the data
magnesium = wine.data[:, wine.feature_names.index('magnesium')]
proline = wine.data[:, wine.feature_names.index('proline')]
is_target0 = wine.target == 0
is_target1 = wine.target == 1
is_target2 = wine.target == 2
targets = [is_target0, is_target1, is_target2]
#second we plot the data
fig, ax = plt.subplots()
for i in range(len(set(wine.target))):
    ax.scatter(magnesium[targets[i]], proline[targets[i]], label=i)
ax.legend()
plt.show()
#target 0 and 1 seem easier to classify by logistic regression
fig, ax = plt.subplots()
ax.scatter(magnesium[targets[0]], proline[targets[0]], label="target 0", color="C1")
ax.scatter(magnesium[targets[1]], proline[targets[1]], label="target 1", color="C2")
ax.legend()
plt.show()

#let's apply logistic regression on these data
X_temp =  wine.data[:, [4, 12]][wine.target != 2]
m = X_temp.shape[0]
X = np.hstack((np.ones((m, 1)), X_temp))
n = X.shape[1]
#get targets for 1 and 2 only and make column vector
Y = (wine.target[wine.target != 2])[:, np.newaxis]

#initialize theta
initial_theta = np.zeros((n, 1))

#define sigmoid function
def s(z):
    """Return result of sigmoid function for opposite of input z"""
    return 1 / (1 + np.exp(-z))

#define cost function
def cost_logistic(X, Y, theta):
    """Return cost logistic regression"""
    #hypothesis function
    h = s(np.dot(X, theta))
    #cost
    J = (np.dot(-Y.T, np.log(h)) - np.dot((1 - Y).T, np.log(1 - h))) / m
    return J

def grad_logistic(X, Y, theta):
    h = s(np.dot(X, theta))
    #gradient
    return np.dot(X.T, h - Y) / m

def gradient_descent(X, Y, theta, alpha, n_iter):
    """Return parameters for linear regression as determined by gradient
descent, and history of cost values."""
    #define array to store cost history 
    cost_history = np.zeros(n_iter)
    #loop for n_iter to perform gradient descent
    for i in range(n_iter):
        delta = grad_logistic(X, Y, theta)
        theta -= alpha * delta
        cost_history[i] = cost_logistic(X, Y, theta)
    return theta, cost_history

alpha = 0.00001
n_iter = 10000
theta, history = gradient_descent(X, Y, initial_theta, alpha, n_iter)

print(theta)

fig, ax = plt.subplots()
x_val = [i for i in range(n_iter)]
ax.plot(x_val, history)
ax.set_xlabel("Iteration number")
ax.set_ylabel("Cost")
plt.show()

#x and y values for decision boundary line
#we only need two points
plot_x = [np.min(X[:, 1]), np.max(X[:, 1])]
plot_y = -(theta[0] + theta[1] * plot_x) / theta[2]
#plot data with decision boundary line
fig, ax = plt.subplots()
ax.scatter(magnesium[targets[0]], proline[targets[0]], label="target 0", color="C1")
ax.scatter(magnesium[targets[1]], proline[targets[1]], label="target 1", color="C2")
ax.plot(plot_x, plot_y, color="C3", label="Decision boundary")
ax.set_xlabel("Magnesium")
ax.set_ylabel("Proline")
ax.set_title("Classification of two wines with logistic regression and gradient descent")
ax.legend()
plt.show()


