import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from Task2 import *

# SECOND DATA SET

# Retrieve the data from the CSV file
x1, x2, y = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2))

w_2 = logReg(500000, X, y, 0.0001)
print "final weights: " + str(w_2)[1:-1]

pos, neg = separateByClass(x1,x2,y)
plt.title('Training Data Set 2')
# plot positive
plt.scatter(pos[0], pos[1], c="green")

# plot negative
plt.scatter(neg[0], neg[1], c="red")

# plot boundary separator
w0 = w_2[0]
w1 = w_2[1]
w2 = w_2[2]

x1 = np.array(range(0,2))
x2 = (-w0-w1*x1)/w2
plt.plot(x1,x2)

plt.show()


# SECOND TEST SET

# Retrieve the data from the CSV file
x1, x2, y = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2))

probabilities = np.dot(w_2.T, X.T)
predictions = list(map(lambda x: 1.0 if x >= 0 else 0.0, probabilities))

misclassifiedPoints = np.sum(y != predictions)
print "the number of misclasified points is: " + str(misclassifiedPoints)

plt.title('Testing Data Set 2')
pos, neg = separateByClass(x1,x2,y)

# plot positive
plt.scatter(pos[0], pos[1], c="green")

# plot negative
plt.scatter(neg[0], neg[1], c="red")

# plot boundary separator
w0 = w_2[0]
w1 = w_2[1]
w2 = w_2[2]

x1 = np.array(range(0,2))
x2 = (-w0-w1*x1)/w2
plt.plot(x1,x2)

plt.show()












