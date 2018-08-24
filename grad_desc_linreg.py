#this script performs a simple linear regression by gradient descent
#written based on the following web page
#http://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
#and following the coursera intro to machine learning course
#the script does not find the optimal parameters for the state.x77 data set
#x= HS Grad and y = Illiteracy if not given approximate correct parameters
#from the beginning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#cost function for linear regression
def cost_fun(x, y, theta_0, theta_1):
    """Return mean squared error based on data x and y, and linear \
parameters theta_0 (intercept) and theta_1 (slope)."""
    sq_err = (theta_1 * x + theta_0 - y) ** 2
    mse = np.sum(sq_err) / len(x)
    return mse 

def update_param(x, y, theta_0, theta_1, rate):
    """Update linear regression parameters based on cost function and \
learning rate."""
    #calculate derivative of theta_0 and theta_1
    d_theta_0 = np.sum(theta_1 * x + theta_0 - y) / len(x)
    d_theta_1 = np.sum(x * (theta_1 * x + theta_0 - y)) / len(x)
    #update parameters
    theta_0 -= rate * d_theta_0
    theta_1 -= rate * d_theta_1

    return theta_0, theta_1

def train(x, y, theta_0=0, theta_1=0, rate=.0005, n_iters=2000):
    """Return simple linear regression parameters by performing gradient
descent based on data x and y, after n_iter number of iterations."""
    cost_history = []
    for i in range(n_iters): 
        theta_0, theta_1 = update_param(x, y, theta_0, theta_1, rate)
        cost = cost_fun(x, y, theta_0, theta_1)
        cost_history.append(cost)
        if i % 100 == 0:
            print("iter #" + str(i) + ", cost = " +str(cost))
    return theta_0, theta_1, cost_history

if __name__ == "__main__":
    #read data from csv file
    ad = pd.read_csv("state-x77.csv")
    grad = ad["HS Grad"]
    illit = ad["Illiteracy"]

    #plot to visualize data
    _, ax = plt.subplots()
    ax.scatter(grad, illit)
    ax.set_xlabel("grad")
    ax.set_ylabel("illit")
    ax.set_title("")
    plt.show()

    #determine simple linear regression parameters and plot line along with data
    print("\nStarting linear regression by gradient descent\n.")
    print("Please wait while parameters are being computed.")
    theta_0, theta_1, history = train(grad, illit)
    #calculate estimated parameters
    y_est = theta_1 * grad + theta_0
    _, ax = plt.subplots()
    ax.scatter(grad, illit, label="data")
    ax.plot(grad, y_est, color="C3", label="linear regression")
    ax.set_xlabel("grad")
    ax.set_ylabel("illit")
    ax.set_title("")
    ax.legend()
    plt.show()

    #plot parameters update history
    x = [i for i in range(len(history))]
    _, ax = plt.subplots()
    ax.plot(x, history)
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("Cost")
    ax.set_title("History of parameters update")
    plt.show()
