# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class woe_tree:
    def __init__(self, max_depth=None, min_samples_leaf=None,
                 criterion='gini', splitter='best',
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, class_weight=None, presort=False):
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._criterion = criterion
        self._splitter = splitter
        self._min_samples_split = min_samples_split
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._random_state = random_state
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._class_weight = class_weight
        self._presort = presort

        self._random_state = random_state
        self._tree_dict = None
        self._woe_dict = None
        pass

    def fit(self, X, y):
        """
        Calculate rules for discretizing features and
        mapping rules (weight of evidence)

        X:  array-like, shape [n_samples, n_features]
            feature space for creating rules
        y:  array-like, len [n_samples]
            target variable
        """
        X = np.float32(X)
        y = np.float32(y)

        self._woe_dict = {}
        self._tree_dict = {}
        for feat_ind in range(X.shape[1]):
            x = X[:, feat_ind]
            if self._min_samples_leaf is None:
                self._min_samples_leaf = np.int(len(y) * 0.1)

            tree = DecisionTreeClassifier(
                max_depth=self._max_depth,
                min_samples_leaf=self._min_samples_leaf,
                criterion=self._criterion,
                splitter=self._splitter,
                min_samples_split=self._min_samples_split,
                min_weight_fraction_leaf=self._min_weight_fraction_leaf,
                max_features=None,
                random_state=self._random_state,
                max_leaf_nodes=self._max_leaf_nodes,
                min_impurity_decrease=self._min_impurity_decrease,
                min_impurity_split=self._min_impurity_split,
                class_weight=self._class_weight,
                presort=self._presort)
            tree.fit(x[:, np.newaxis], y)
            self._tree_dict[feat_ind] = tree.apply

            nodes = tree.apply(x[:, np.newaxis])
            unique_nodes = np.unique(nodes, return_counts=True)
            feature_woe_dict = {}
            for ind, node_num in enumerate(unique_nodes[0]):
                n_pos = sum(y[nodes == node_num])
                woe = np.log(n_pos / (unique_nodes[1][ind] - n_pos))
                feature_woe_dict[node_num] = woe
            self._woe_dict[feat_ind] = feature_woe_dict
        pass

    def transform(self, X):
        """
        X:    array-like, shape [n_samples, n_features]
              Input data that will be transformed.
        """
        woe_X = np.zeros_like(X)

        for feat_ind in range(X.shape[1]):
            # map values
            woe_X[:, feat_ind] = np.vectorize(
                self._woe_dict[feat_ind].__getitem__)(
                self._tree_dict[feat_ind](X[:, feat_ind][:, np.newaxis]))
        return woe_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
