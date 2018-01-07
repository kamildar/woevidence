# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class woe_forest:
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._bootstrap = bootstrap
        self._oob_score = oob_score
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._verbose = verbose
        self._warm_start = warm_start
        self._class_weight = class_weight

        self._woe_dict = None
        self._forest_apply = None
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
        rf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            criterion=self._criterion,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            min_weight_fraction_leaf=self._min_weight_fraction_leaf,
            max_features=self._max_features,
            max_leaf_nodes=self._max_leaf_nodes,
            min_impurity_decrease=self._min_impurity_decrease,
            min_impurity_split=self._min_impurity_split,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbose=self._verbose,
            warm_start=self._warm_start,
            class_weight=self._class_weight)
        rf.fit(X, y)
        self._forest_apply = rf.apply

        nodes = rf.apply(X)
        self._woe_dict = {}
        for tree_ind in range(self._n_estimators):
            tree_woe_dict = {}
            unique_nodes = np.unique(nodes[:, tree_ind],
                                     return_counts=True)
            for node_ind, node_num in enumerate(unique_nodes[0]):
                n_pos = sum(y[nodes[:, node_ind] == node_num])
                woe = np.log(
                    n_pos / (unique_nodes[1][node_ind] - n_pos))
                tree_woe_dict[node_num] = woe
            self._woe_dict[tree_ind] = tree_woe_dict
        pass

    def transform(self, X):
        """
        X:  array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        woe_X = np.float32(self._forest_apply(X))

        for tree_ind in range(self._n_estimators):
            woe_X[:, tree_ind] = np.vectorize(
                self._woe_dict[tree_ind].__getitem__)(
                woe_X[:, tree_ind])
        return woe_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
