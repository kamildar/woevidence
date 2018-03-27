# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License

import numpy as np


def gini(y):
    prob = np.sum(y) / len(y)
    return prob * (1 - prob)


def entropy(y):
    prob = np.sum(y) / len(y)
    return (- prob * np.log(prob))


def split(x, y, criterion):
    if criterion == 'gini':
        splitter = gini
    elif criterion == 'entropy':
        splitter = entropy
    else:
        assert callable(criterion)

    x = np.array(x)
    y = np.array(y)

    n_obs = len(y)
    sort_inds = np.argsort(x)
    # x = x[sort_inds]
    y = y[sort_inds]

    x_info = np.unique(x, return_counts=True)

    impurities = np.zeros_like(x)
    for ind, n_left in enumerate(np.cumsum(x_info[1])[:-1]):
        impurities[ind] = (
            (splitter(y[:n_left]) * n_left +
             splitter(y[n_left:]) * (n_obs - n_left)) / n_obs)
    threshold = np.argmin(impurities[:ind])
    return threshold


class OneFeatureTree(object):

    def __init__(self,
                 criterion,
                 max_depth=None,
                 min_samples_leaf=2,
                 min_samples_class=1):
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_class = min_samples_class

        self._tree = {}

    def _split_vector(self, x, y, value):
        left_ind = x < value
        left_x, right_x = x[left_ind], x[np.logical_not(left_ind)]
        left_y, right_y = y[left_ind], y[np.logical_not(left_ind)]
        return left_x, right_x, left_y, right_y

    def _calc_woe(self, y):
        n_pos = np.sum(y)
        n_neg = np.float32(len(y)) - n_pos
        woe = np.log(n_pos / n_neg)
        return woe

    def _split(self, x, y):
        if self._criterion == 'gini':
            splitter = gini
        elif self._criterion == 'entropy':
            splitter = entropy
        else:
            assert callable(self._criterion)

        x = np.array(x)
        y = np.array(y)

        n_obs = len(y)
        sort_inds = np.argsort(x)
        y = y[sort_inds]

        x_info = np.unique(x, return_counts=True)

        impurities = np.zeros(len(x) - 2)
        for ind, n_left in enumerate(np.cumsum(x_info[1])[:-1]):
            impurities[ind] = (
                (splitter(y[:n_left]) * n_left +
                 splitter(y[n_left:]) * (n_obs - n_left)) / n_obs)
        threshold = np.argmin(impurities)
        return threshold

    def _fit_node(self, x, y,
                  depth, node):

        min_samples = (len(y) > self._min_samples_leaf)
        uniq_x = len(np.unique(x)) > 1
        min_class = np.all(
            np.unique(y, return_counts=True)[1] >
            self._min_samples_class)
        max_depth = (depth < self._max_depth)

        if (min_samples and min_class and max_depth and uniq_x):
            print(len(y))


            # zero node type for non-terminal nodes
            node['type'] = 0

            threshold = self._split(x, y)
            left_x, right_x, left_y, right_y = self._split_vector(
                x, y, threshold)

            # 0 -- left_child, 1 -- right child
            node[0] = {}
            node[1] = {}
            node['thresh'] = threshold
            self._fit_node(left_x, left_y,
                           depth + 1,
                           node[0])
            self._fit_node(right_x, right_y,
                           depth + 1,
                           node[1])
        else:
            node['type'] = 1
            node['woe'] = self._calc_woe(y)

    def fit(self, x, y):
        self._fit_node(x, y,
                       depth=0, node=self._tree)

    def _predict_node(self, x, node):
        if node['type'] == 0:
            if x < node['thresh']:
                return self._predict_node(x, node[0])
            else:
                return self._predict_node(x, node[1])
        return node['woe']

    def predict(self, x):
        predicted = np.zeros_like(x)
        for ind in range(len(x)):
            predicted[ind] = self._predict_node(x[ind], self._tree)
        return predicted
