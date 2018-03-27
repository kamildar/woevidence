import numpy as np


def gini(y, **kwargs):
    prob = np.sum(y) / len(y)
    return prob * (1 - prob)


def entropy(y, smooth=0, **kwargs):
    prob = np.sum(y) / len(y)
    return (- prob * np.log(prob + smooth))
