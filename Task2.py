import numpy as np
import matplotlib.pyplot as plt
import random
import csv

def separateByClass(x1,x2,y):
  pos = [[], []]
  neg = [[], []]

  for i in range(len(x1)):
    if y[i]:
      pos[0].append(x1[i])
      pos[1].append(x2[i])
    else:
      neg[0].append(x1[i])
      neg[1].append(x2[i])
  return pos, neg

def separateByClass3D(x1,x2, x3, y):
  pos = [[], [], []]
  neg = [[], [], []]

  for i in range(len(x1)):
    if y[i]:
      pos[0].append(x1[i])
      pos[1].append(x2[i])
      pos[2].append(x3[i])

    else:
      neg[0].append(x1[i])
      neg[1].append(x2[i])
      neg[2].append(x3[i])

  return pos, neg

def logLikeHood(x, y, w):
  classes = np.dot(x, w)
  logLike = np.sum( y*classes - np.log(1 + np.exp(classes)) )
  return logLike

def crossEntropyError(x, y, w):
  ll = logLikeHood(x, y, w)
  return -1/y.shape[0] * ll

def sigmoid(classes):
  return 1 / (1 + np.exp(-classes))

def logReg(iterations, X, y, learningRate, crossEntropy=False, X_test=None, y_test=None):
  # initialize weights
  weights = np.zeros(X.shape[1])

  trainCrossEntropies = []
  testCrossEntropies = []

  for i in range(iterations):
    # Update rule of equation 20 done by parts
    classes = np.dot(weights.T, X.T)
    pred = sigmoid(classes)
    error = pred - y
    grad = np.dot(error, X)
    weights -= learningRate * grad

    # Add the cross entropies
    if (crossEntropy and (i < 1000)):
      trainCrossEntropies.append(crossEntropyError(X, y, weights))
      testCrossEntropies.append(crossEntropyError(X_test, y_test, weights))

    # Print the log of the likelihood every 10000 steps
    if i % 10000 == 0:
      print logLikeHood(X, y, weights)
      print weights

  if (crossEntropy):
    return weights, trainCrossEntropies, testCrossEntropies

  return weights

