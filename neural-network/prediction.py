#script which generates prediction for the MNIST dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import sys
import os

sys.path.append(os.getcwd())
os.chdir("..")

data = np.array(pd.read_csv("mnist_train.csv"))
X = data[:, 1:]
Y = data[:, 0][:, np.newaxis]

os.chdir("neural-network")

model = nn.NeuralNet(hidden_size=60, n_labels=10, lamb=1, alpha=0.4)
theta1, theta2 = model.import_theta("weights_digits2.csv", X)
prediction, accuracy = model.predict(theta1, theta2, X, Y)

def show_pred():
    a = np.random.randint(60000)
    print("The predicted number is", prediction[a])
    fig, ax = plt.subplots()
    ax.imshow(X[a, :].reshape((28, 28)), cmap="Greys")
    plt.show()
