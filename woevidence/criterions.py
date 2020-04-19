import numpy as np


def gini(y, sample_weight, **kwargs):
    prob = np.sum(y * sample_weight) / sample_weight.sum()
    return prob * (1 - prob)


def entropy(y, sample_weight, smooth=0):
    prob = (np.sum(y * sample_weight) + smooth) / (sample_weight.sum() + smooth)
    return - prob * np.log(prob)
