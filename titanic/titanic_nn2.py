# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:51:51 2018

@author: thejo
"""

#script to predict survivors of titanic dataset using a neural network

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time
from sklearn.model_selection import train_test_split
import sys
sys.path.append("D:\\machine-learning")
from neural_network import NeuralNet

#import data
data = pd.read_csv("train_feat_engin.csv", header=None)
X_df = data.iloc[0:data.shape[0], 1:data.shape[1]]
Y_df = data.iloc[0:data.shape[0], 0]

#put data from dataframe or data series to numpy array
X = np.concatenate((X_df.iloc[0:data.shape[0], 0][:, np.newaxis],
                    X_df.iloc[0:data.shape[0], 1][:, np.newaxis],
                    X_df.iloc[0:data.shape[0], 2][:, np.newaxis],
                    X_df.iloc[0:data.shape[0], 3][:, np.newaxis],
                    X_df.iloc[0:data.shape[0], 4][:, np.newaxis], 
                    X_df.iloc[0:data.shape[0], 5][:, np.newaxis], 
                    X_df.iloc[0:data.shape[0], 6][:, np.newaxis]), axis=1)

Y = np.array(Y_df)[:, np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

#import test data
test = pd.read_csv("test_feat_engin.csv", header=None)

#import non-engineered test data to get passenger Id
test_original = pd.read_csv("test.csv")

#generate and train neural network
def train():
    """Train the neural network"""
    model = NeuralNet(alpha=0.003, n_iter=300000, lamb=5, hidden_size=28, n_labels=2)
    
    start = time.time()
    Theta1, Theta2, cost_history = model.fit(X, Y)
    stop = time.time()
    elapsed = stop - start
    
    pred, accu = model.predict(Theta1, Theta2, X_test, Y_test)
    
    print("Time to train: {:.1f} seconds".format(elapsed))
    print("Model accuracy: {:.3f}".format(accu))
    model.plot_history(cost_history)
    
    return Theta1, Theta2, pred, accu, cost_history

def optim():
    """Determine best lambda parameter"""
    #list of lambda values to try
    lambdas = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
    #lambdas = [2.56, 5.12, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    
    #list to record training parameters
    histories = []
    J_train = []
    J_test =[]
    accuracies = []
    
    for l in lambdas:
        #define and train model
        model = NeuralNet(alpha=0.003, n_iter=40000, lamb=l, hidden_size=28, n_labels=2)
        Theta1, Theta2, cost_history = model.fit(X_train, Y_train)
        
        #append parameters to lists
        histories.append(cost_history)
        J_train.append(cost_history[-1])
        X_test_1 = model.add_intercept(X_test)
        J_test.append(model.forward_prop(X_test_1, Y_test, Theta1, Theta2)[4])
        accuracies.append(model.predict(Theta1, Theta2, X_test, Y_test)[1])
        
    #plot costs VS lambda
    fig, ax = plt.subplots()
    ax.plot(lambdas, J_train, color="C0", label="training")
    ax.plot(lambdas, J_test, color="C2", label="testing")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Cost")
    ax.legend()
    plt.show()
    
    #plot accuracies VS lambdas
    fig, ax = plt.subplots()
    ax.plot(lambdas, accuracies)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Accuracy on test sample")
    plt.show()
    
    return accuracies

def output_results():
    """Function to output results for submission to kaggle"""
    #set dummy Y for the test
    dummy = np.ones((418, 1))
    
    #predict survival of test examples
    prediction = model.predict(Theta1, Theta2, test, dummy)[0]
        
    #prediction in column with passenger ID
    prediction = np.hstack((test_original['PassengerId'][:, np.newaxis], prediction))
    
    #save data as csv file
    np.savetxt("titanic_pred_nn.csv", prediction, delimiter=',', fmt='%d')

#accu = optim()

Theta1, Theta2, pred, accu, cost_history = train()

output_results()
