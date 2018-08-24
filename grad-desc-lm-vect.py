#script which implements gradient descent to solve parameters of a
#linear regression using vectorized calculation
#there is also a function to calculate parameters with the normal equation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost_fun(X, Y, theta):
    """Return mean squared error based on data x and y, and linear
parameters."""
    return np.sum((np.dot(X, theta) - Y) ** 2) / 2 / len(Y)

def gradient_descent(X, Y, theta, alpha, n_iter):
    """Return parameters for linear regression as determined by gradient
descent, and history of cost values."""
    #define array to store cost history 
    cost_history = np.zeros(n_iter)
    #loop for n_iter to perform gradient descent
    for i in range(n_iter):
        delta =  np.dot(X.T, (np.dot(X, theta) - Y)) / len(Y)
        theta -= alpha * delta
        cost_history[i] = cost_fun(X, Y, theta)
    return theta, cost_history

def norm_eq(X, Y):
    """Return linear regression parameters for x and y by using the
normal equation."""
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

if __name__ == "__main__":
    #read data from csv file
    data = pd.read_csv("state-x77.csv")
    x_origin = data['HS Grad']
    y_origin = data['Illiteracy']

    #plot to visualize data
    _, ax = plt.subplots()
    ax.scatter(x_origin, y_origin)
    ax.set_xlabel("Percentage of high school graduates")
    ax.set_ylabel("Percentage of illiteracy")
    plt.show()

    #features normalization
    mu = np.mean(x_origin)
    sigma = np.std(x_origin)
    x = (x_origin - mu) / sigma

    #put x into column array and add column of ones for vector operations
    x = np.array(x)[:, np.newaxis] #x into column array
    x1 = np.ones((len(x), 1)) #create column array of ones
    x = np.concatenate((x1, x), axis = 1) #add column of ones

    #put y into column array
    y = np.array(y_origin)[:, np.newaxis]

    #set parameters theta
    theta = np.zeros((2, 1))
    
    #determine parameters theta for linear regression by gradient descent
    n_iter = 10000 #set the number of iterations
    theta, history = gradient_descent(x, y, theta, alpha=0.01, n_iter=n_iter)
    print("\n\nCalculated theta (normalized features) :\nintercept :",
          theta[0],
          "\nslope :",
          theta[1])

    #calculate "de-normalized" theta
    theta_denorm = np.zeros((2, 1))
    theta_denorm[0] = theta[0] - theta[1] * mu / sigma
    theta_denorm[1] = theta[1] / sigma
    print("\n\nCalculated theta ('de-normalized') :\nintercept :",
          theta_denorm[0],
          "\nslope :",
          theta_denorm[1])

    #determine parameters with normal equation (no need to normalize x)
    #first, add column of ones to x_origin
    x_origin_ones = np.concatenate((x1, x_origin[:, np.newaxis]), axis=1)
    theta_n = norm_eq(x_origin_ones, y)
    print("\n\nCalculated theta (normal equation) :\nintercept :",
          theta_n[0],
          "\nslope :",
          theta_n[1])
    
    #plot data with linear regression line
    #hypothesis = np.dot(x, theta) #calculated values for y given theta
    _, ax = plt.subplots()
    ax.scatter(x_origin, y_origin, label="data")
    #ax.plot(x_origin, hypothesis, color="C3", label="linear regression")
    ax.plot(x_origin, np.dot(x_origin_ones, theta_denorm), color="C3", label="linear regression")
    ax.text(58, 2.4, "y = {0:.2f} * x + {1:.2f}".format(theta_denorm[1].flat[0], theta_denorm[0].flat[0]))
    ax.set_xlabel("Percentage of high school graduates")
    ax.set_ylabel("Percentage of illiteracy")
    ax.legend()
    plt.show()

    #do gradient descent again with other learning rates to compare
    #the speed at which gradient descent converges towards minimum cost
    theta = np.zeros((2, 1))
    _, history2 = gradient_descent(x, y, theta, alpha=0.003, n_iter=n_iter)
    theta = np.zeros((2, 1))
    _, history3 = gradient_descent(x, y, theta, alpha=0.001, n_iter=n_iter)
    theta = np.zeros((2, 1))
    _, history4 = gradient_descent(x, y, theta, alpha=0.0003, n_iter=n_iter)

    #plot history of gradient descent
    x_val = [i for i in range(n_iter)]
    _, ax = plt.subplots()
    ax.plot(x_val, history, label="alpha = 0.01")
    ax.plot(x_val, history2, label="alpha = 0.003")
    ax.plot(x_val, history3, label="alpha = 0.001")
    ax.plot(x_val, history4, label="alpha = 0.0003")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Cost")
    ax.legend()
    plt.show()
    
