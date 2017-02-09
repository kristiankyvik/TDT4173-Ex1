import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import csv

from Task2 import *

# SECOND DATA SET

# Retrieve the data from the CSV file
x1, x2, y = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# initialize array containing new thrid dimension
x3 = np.zeros((y.shape[0]))

for i in range(y.shape[0]):
  x3[i] = (x1[i]**2 + x2[i]**2)

# Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2,x3))

w_2 = logReg(500000, X, y, 0.0001)
print "final weights: " + str(w_2)[1:-1]

pos, neg = separateByClass3D(x1,x2,x3,y)

fig = pylab.figure()
ax = Axes3D(fig)

ax.set_title("Transformed Training Data Set 2")

# plot positive
ax.scatter(pos[0], pos[1], pos[2], c="green")

# plot negative
ax.scatter(neg[0], neg[1], neg[2], c="red")

plt.show()

# SECOND TEST SET

# Retrieve the data from the CSV file
x1, x2, y = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=',', usecols=(0,1,2), unpack=True, dtype=None)

# initialize array containing new thrid dimension
x3 = np.zeros((y.shape[0]))

for i in range(y.shape[0]):
  x3[i] = (x1[i]**2 + x2[i]**2)

# Add a column of ones
ones = np.ones(y.shape[0])
X = np.column_stack((ones,x1,x2,x3))

probabilities = np.dot(w_2.T, X.T)
predictions = list(map(lambda x: 1.0 if x >= 0 else 0.0, probabilities))
misclassifiedPoints = np.sum(y != predictions)
print "the number of misclasified points is: " + str(misclassifiedPoints)

pos, neg = separateByClass3D(x1,x2,x3,predictions)

fig = pylab.figure()
ax = Axes3D(fig)
ax.set_title("Transformed Testing Data Set 2")

# plot positive
ax.scatter(pos[0], pos[1], pos[2], c="green")

# plot negative
ax.scatter(neg[0], neg[1], neg[2], c="red")

plt.show()












