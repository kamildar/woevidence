import numpy as np


def gini(y):
    prob = np.sum(y) / len(y)
    return prob * (1 - prob)


def entropy(y, smooth=0):
    prob = (np.sum(y) + smooth) / (len(y) + smooth)
    return - prob * np.log(prob)
