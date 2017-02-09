import numpy as np
import matplotlib.pyplot as plt
import csv

# TRAIN DATA

# Retrieve the data from the CSV file
x1, x2, y= np.genfromtxt('./datasets/regression/reg-2d-train.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2))

# Make sure they have the right shape to perfrom matrix multiplication
#print X.shape, y.shape

# Define the OLS regression method (employing closed formx)
def OLS(X, y):
  return np.dot(np.linalg.pinv(np.dot(X.T,X)), np.dot(X.T,y))

# Print out the result
weights = OLS(X,y)

print "Weights (first item being bias): " + str(weights)[1:-1]

# Define the Mean suared error calculation using the linear algebra version of the formula provided in the assignement
def EMSE (w, X, y):
  return (np.dot(w.T, np.dot(np.dot(X.T, X), w)) - 2 * np.dot(np.transpose(np.dot(X, w)), y) + np.dot(y.T, y)) * 1/y.shape[0]

print "Mean squared error train data: " + str(EMSE(weights, X, y))

# Load in the test data
test_x1, test_x2, test_y= np.genfromtxt('./datasets/regression/reg-2d-test.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)
test_ones = np.ones(test_y.shape[0])
test_X = np.column_stack((test_ones,test_x1,test_x2))

# Make sure they have the right shape to perfrom matrix multiplication
#print test_X.shape, test_y.shape

print "Mean squared error test data: " + str(EMSE(weights, test_X, test_y))

# TEST DATA

# Retrieve the data from the CSV file
x1_1d, y_1d= np.genfromtxt('./datasets/regression/reg-1d-train.csv', delimiter=',', usecols=(0,1), unpack=True, dtype=None)

# Add a column of ones
ones_1d = np.ones(y_1d.shape[0])
X_1d= np.column_stack((ones_1d,x1_1d))

# Print out the result
weights_1d = OLS(X_1d,y_1d)

print "Weights (first item being bias): " + str(weights_1d)[1:-1]

# Retrieve the test data from the CSV file
test_x1_1d, test_y_1d= np.genfromtxt('./datasets/regression/reg-1d-test.csv', delimiter=',', usecols=(0,1), unpack=True, dtype=None)

# Add a column of ones
test_ones_1d = np.ones(test_y_1d.shape[0])
test_X_1d= np.column_stack((test_ones_1d,test_x1_1d))

print "Mean squared error test data: " + str(EMSE(weights_1d, test_X_1d, test_y_1d))

# Graphing part

# First lets graph the test data points
plt.scatter(test_x1_1d, test_y_1d)

# Next, lets graph the equation found by our regression
reg_x = test_X_1d
reg_y = np.dot(weights_1d.T, test_X_1d.T)
plt.plot(reg_x, reg_y)

plt.show()

