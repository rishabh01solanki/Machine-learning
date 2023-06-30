import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/houses.csv')
x = df.iloc[:, 0].values  # size in sq ft
y = df.iloc[:, 1].values  # price of houses

# Initialize parameters
m = len(y)  # number of training examples
x = x.reshape(m, 1)
y = y.reshape(m, 1)
alpha = 0.00000001

# Add a column of ones to x for the bias term
x = np.hstack((np.ones((m, 1)), x))

theta = np.zeros((2, 1))  # theta parameters; it is a column vector

# Run batch gradient descent
for iteration in range(100000):  # example number of iterations
    # define hypothesis that we want to be close to real values (y)
    h = np.dot(x, theta)  # matrix multiplication

    # batch gradient descent update rule
    theta -= alpha * 1/m * np.dot(x.T, (h - y))

print(theta)



# Scatter plot of the data
plt.scatter(x[:, 1], y, color='red', marker='x', label='Training data')

# Line plot of the hypothesis
h = np.dot(x, theta)
plt.plot(x[:, 1], h, color='blue', label='Linear regression')

# Add labels
plt.xlabel('Size in sq ft')
plt.ylabel('Price of houses')
plt.legend()

# Show the plot
plt.show()