# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License

import numpy as np
# from collections import Counter


def gini(y):
    prob = np.sum(y) / len(y)
    return prob * (1 - prob)


def entropy(y):
    prob = np.sum(y) / len(y)
    return - prob * np.log(prob)


def split(x, y, criterion,
          min_samples_leaf,
          min_samples_pos,
          min_samples_neg):
    if criterion == 'gini':
        splitter = gini
    elif criterion == 'entropy':
        splitter = entropy

    x = np.array(x)
    y = np.array(y)

    n_obs = len(y)
    sort_inds = np.argsort(x)
    # x = x[sort_inds]
    y = y[sort_inds]

    x_info = np.unique(x, return_count=True)

    impurities = np.zeros_like(x)
    for ind, n_left in enumerate(np.cumsum(x_info[1])):
        impurities[ind] = (
            (splitter(y[:n_left]) * n_left +
             splitter(y[:n_left]) * (n_obs - n_left)) / n_obs)
