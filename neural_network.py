#script which defines a class for classification using a neural network
#trained by backpropagation
#weights_digits.csv contains weights learned from digits.csv and
#digits_output.csv (coursera machine learning tutorial)
#weights_digits2.csv contains weights learned from mnist_train.csv with
#400 units in the hidden layer

import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

class NeuralNet(object):
    """This class provides method to perform classification of data using
a neural network.
alpha        : learning rate
n_iter       : number of iteration of forward/backward propagation
lamb         : lambda parameter for regularization
hidden_size  : hidden layer size
n_labels     : number of labels, i.e. number of output classes"""
    def __init__(self, alpha=1, n_iter=400, lamb=1, hidden_size=None, n_labels=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lamb = lamb
        self.hidden_size = hidden_size
        self.n_labels = n_labels

    def rand_weights(self, c_in, c_out, epsilon=0.12):
        """Return randomly initialized weights of a layer with c_in incoming 
        connections and c_out and c_out outgoing connection."""
        return np.random.random((c_out, c_in + 1)) * 2 * epsilon - epsilon

    def add_intercept(self, X):
        """Return matrix with an added column of ones."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _s(self, z):
        """Return product of sigmoid function with z as input."""
        return 1 / (1 + np.exp(-z))

    def _sgrad(self, z):
        """Return the gradient of sigmoid function with z as input."""
        return self._s(z) * (1 - self._s(z))

    def forward_prop(self, X, Y, Theta1, Theta2):
        """Return values calculated during forward propagation and cost."""
        z_2 = np.dot(X, Theta1.T)
        a_2 = self._s(z_2)
        a_2 = self.add_intercept(a_2)
        z_3 = np.dot(a_2, Theta2.T)
        h = self._s(z_3)

        #calculate unregularized cost
        m = X.shape[0]
        J = np.sum(np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))) / -m
        #calculate regularization parameters
        T1 = Theta1[:, 1:]
        T2 = Theta2[:, 1:]
        A = self.lamb / 2 / m
        B = np.sum(np.sum(T1 * T1, axis=1))
        C = np.sum(np.sum(T2 * T2, axis=1))
        #calculate regularized cost
        J_reg = J + A * (B + C)

        return z_2, a_2, z_3, h, J_reg

    def backward_prop(self, z_2, a_2, z_3, h, X, Y, Theta1, Theta2, lamb, alpha):
        """Return updated Theta1 and Theta2."""
        #perform backward propagation per se
        m = Y.shape[0]
        T2 = Theta2[:, 1:]
        d_3 = h - Y
        d_2 = np.dot(d_3, T2) * self._sgrad(z_2)
        Delta_1 = np.dot(d_2.T, X)
        Delta_2 = np.dot(d_3.T, a_2)

        #calculate unregularized gradient
        Theta1_grad = Delta_1 / m
        Theta2_grad = Delta_2 / m

        #regularization of gradient
        #replace first column with 0 to avoid regularization of bias
        Theta1_copy = np.copy(Theta1)
        Theta2_copy = np.copy(Theta2)
        Theta1_copy[:, 0] = 0
        Theta2_copy[:, 0] = 0
        #scale Theta1/2 and add to gradient
        Theta1_grad_reg = Theta1_grad + np.dot(lamb / m, Theta1_copy)
        Theta2_grad_reg = Theta2_grad + np.dot(lamb / m, Theta2_copy)

        #update Theta1/2
        Theta1_updated = Theta1 - alpha * Theta1_grad_reg
        Theta2_updated = Theta2 - alpha * Theta2_grad_reg

        return Theta1_updated, Theta2_updated

    def predict(self, Theta1, Theta2, X, Y):
        """Return predicted output and accuracy of model."""
        #add intercept to X
        X_1 = self.add_intercept(X)

        #calculate outputs of neural network layers
        h1 = self._s(np.dot(X_1, Theta1.T))
        h1_1 = self.add_intercept(h1)
        h2 = self._s(np.dot(h1_1, Theta2.T))

        #predicted output (remember indexing starts at zero)
        #prediction must have the same shape as Y
        prediction = (np.argmax(h2, axis=1))[:, np.newaxis]

        #accuracy
        accuracy = np.mean(prediction == Y)

        return prediction, accuracy

    def fit(self, X, Y):
        """Train the neural network."""
        #add intercept to X
        X_1 = self.add_intercept(X)

        #get network architecture parameters
        in_size = X.shape[1] #input layer size
        hid_size = self.hidden_size #hidden layer size
        n_lab = self.n_labels #output layer size

        #recode Y into one-hot labels
        #for Octave data set use Y.item(i) - 1
        Y_1 = np.zeros((Y.shape[0], n_lab))
        for i in range(Y.shape[0]):
            Y_1[i, Y.item(i)] = 1
        #could import OneHotEncoder from sklearn.preprocessing
        #then Y_1 = OneHotEncoder(sparse=False).fit_transform(Y)
        
        #initialize Theta matrices randomly
        Theta1 = self.rand_weights(in_size, hid_size, 0.12)
        Theta2 = self.rand_weights(hid_size, n_lab, 0.12)

        #define array to store cost history
        cost_history = np.zeros(self.n_iter)

        #train the neural network by repetition of for and back prop
        for i in range(self.n_iter):
            #for prop
            z_2, a_2, z_3, h, J_reg = self.forward_prop(X_1, Y_1, Theta1, Theta2)
            cost_history[i] = J_reg

            #back prop
            Theta1, Theta2 = self.backward_prop(z_2, a_2, z_3, h, X_1, Y_1, 
                                                Theta1, Theta2, 
                                                self.lamb, self.alpha)

        return Theta1, Theta2, cost_history

    def plot_history(self, history):
        "Plot cost history to check error of parameters is minimized."""
        fig, ax = plt.subplots()
        ax.plot([i for i in range(self.n_iter)], history)
        ax.set_xlabel("Number of iteration")
        ax.set_ylabel("Cost")
        ax.set_title("Cost history of neural network parameters")
        plt.show()
        
    def export_theta(self, filename, Theta1, Theta2):
        """Export parameters as csv file."""
        #flatten matrices
        thetas = np.concatenate((Theta1.flatten(), Theta2.flatten()))
        
        #save flat array as csv file
        np.savetxt(filename, thetas, delimiter=",")
        
    def import_theta(self, filename, X):
        """Import csv file containing parameters and convert to matrices of
        relevant size. Specify X to reshape Theta1 correctly"""
        #import csv file
        data = pd.read_csv(filename, header=None)
        weights = np.array(data).flatten()
        
        #extract and reshape matrices
        Theta1 = weights[:self.hidden_size * (X.shape[1] + 1)]
        Theta1 = Theta1.reshape((self.hidden_size, X.shape[1] + 1))
        Theta2 = weights[self.hidden_size * (X.shape[1] + 1):]
        Theta2 = Theta2.reshape((self.n_labels, self.hidden_size + 1))
        
        return Theta1, Theta2

if __name__ == "__main__":
    #import data and build input and build input and output matrices
    #data_in = pd.read_csv("digits.csv", header=None)
    #data_out = pd.read_csv("digits_output.csv", header=None)
    #X = np.array(data_in)
    #Y = np.array(data_out)

    #define model, n_iter and alpha determined by trial and error
    model = NeuralNet(hidden_size=25, n_labels=10, n_iter=400, lamb=0.64)

    data_train = np.array(pd.read_csv("mnist_train.csv"))
    X_train = data_train[:, 1:]
    Y_train = data_train[:, 0][:, np.newaxis]
    
    data_test = np.array(pd.read_csv("mnist_test.csv"))
    X_test = data_test[:, 1:]
    Y_test = data_test[:, 0][:, np.newaxis]

    #train the model
    #measure execution time
    start = time.time()
    Theta1, Theta2, cost_history = model.fit(X_train, Y_train)
    stop = time.time()
    print("Time to train the neural network :", str(stop - start))

    #determine accuracy of the model
    pred, accu = model.predict(Theta1, Theta2, X_train, Y_train)
    pred2, accu2 = model.predict(Theta1, Theta2, X_test, Y_test)
    X_test = model.add_intercept(X_test)
    print(model.forward_prop(X_test, Y_test, Theta1, Theta2)[4])
    
    #trying to optimize lambda
    #lambdas = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
    #cost_train = np.zeros(len(lambdas))
    #cost_test = np.zeros(len(lambdas))
    #for i in range(len(lambdas)):
    #    model = NeuralNet(hidden_size=25, n_labels=10, n_iter=50, lamb=lambdas[i])
    #    Theta1, Theta2, cost_history = model.fit(X_train, Y_train)
    #    cost_train[i] = cost_history[-1]
    #    _, _, _, _, cost_test[i] = model.forward_prop(X_test, Y_test, Theta1, Theta2)
    
    #fig, ax = plt.subplots()
    #ax.plot(lambdas, cost_train, color="C0", label="Training cost")
    #ax.plot(lambdas, cost_test, color="C1", label="Testing cost")
    #ax.legend()
    #plt.show()
        
        
        
        
        
        
        
        
        
        
        
    