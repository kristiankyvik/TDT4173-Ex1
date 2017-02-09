import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from Task2 import *

# FIRST DATA SET

# Retrieve the training data from the CSV file
x1, x2, y= np.genfromtxt('./datasets/classification/cl-train-1.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

#Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2))

# Retrieve the test data from the CSV file
x1_test, x2_test, y_test= np.genfromtxt('./datasets/classification/cl-test-1.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# Add a column of ones
ones_test = np.ones(y_test.shape[0])
X_test = np.column_stack((ones_test,x1_test,x2_test))

w, trainCrossEntropies, testCrossEntropies = logReg(1000000, X, y, 0.1, True, X_test, y_test)

print "final weights: " + str(w)[1:-1]

pos, neg = separateByClass(x1,x2,y)

plt.title('Training Data Set 1')

# plot positive
plt.scatter(pos[0], pos[1], c="red")

# plot negative
plt.scatter(neg[0], neg[1], c="green")

# plot boundary separator
w0 = w[0]
w1 = w[1]
w2 = w[2]
x1 = np.array(range(0,2))
x2 = (-w0-w1*x1)/w2
plt.plot(x1,x2)

plt.show()

# Plot the cross entropies of the training data
iterations = np.array(range(0,1000))
plt.title('Cross-entropy error (1000 first iterations)')
plt.scatter(iterations, trainCrossEntropies, c="green")
plt.scatter(iterations, testCrossEntropies, c="red")

plt.show()

# FIRST TEST DATA

probabilities = np.dot(w.T, X_test.T)
predictions = list(map(lambda x: 1 if x >= 0 else 0, probabilities))
misclassifiedPoints = np.sum(y != predictions)
print "the number of misclasified points is: " + str(misclassifiedPoints)

pos, neg = separateByClass(x1_test,x2_test, y_test)
plt.title('Testing Data Set 1')
# plot positive
plt.scatter(pos[0], pos[1], c="green")

# plot negative
plt.scatter(neg[0], neg[1], c="red")

# plot boundary separator
w0 = w[0]
w1 = w[1]
w2 = w[2]
x1 = np.array(range(0,2))
x2 = (-w0-w1*x1)/w2
plt.plot(x1,x2)

plt.show()


# Cross entropy graphs


