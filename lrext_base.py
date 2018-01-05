# Author: Kamaldinov Ildar (kamildraf@gmail.com)

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class WeightOfEvidence:
    def __init__(self, max_depth=None, min_samples_leaf=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        pass

    def _woe_transform(self, x, y):
        x = np.float32(x)
        y = np.float32(y)
        if self.min_samples_leaf is None:
            self.min_samples_leaf = np.int(len(y) * 0.1)
        tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=self.random_state)
        tree.fit(x[:, np.newaxis], y)

        nodes = tree.apply(x[:, np.newaxis])

        unique_nodes = np.unique(nodes, return_counts=True)

        x_blank = np.zeros_like(x)
        for ind in range(len(unique_nodes[0])):
            n_pos = sum(y[nodes == unique_nodes[0][ind]])
            woe = np.log(n_pos / (unique_nodes[1][ind] - n_pos))
            x_blank[nodes == unique_nodes[0][ind]] = woe
        return(x_blank)

    def transform(self, X, y):
        """
        Compute intervals for calculating WOE and compute it on full dataset

        Parameters
        __________
        X:  array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        y:  array-like, shape [n_samples, 1]
            target variable for calculating WOE
        """
        X = np.float32(X)
        woe_X = np.hstack((X,
                           np.apply_along_axis(lambda x:
                                               self._woe_transform(x, y),
                                               axis=0,
                                               arr=X)))
        return(woe_X)

    def 
