# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License
import numpy as np
from criterions import gini, entropy


class OneFeatureTree(object):

    def __init__(self,
                 criterion,
                 min_samples_leaf=2,
                 smooth_woe=0.001,
                 min_samples_class=1,
                 max_depth=None,
                 smooth_entropy=0.001,
                 dtype=np.float32):
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_class = min_samples_class
        self._smooth_woe = smooth_woe
        self._dtype = dtype
        self._smooth_entropy = smooth_entropy

        self._tree = {}

    def _split_vector(self, x, y, value):
        left_ind = x < value
        left_x, right_x = x[left_ind], x[np.logical_not(left_ind)]
        left_y, right_y = y[left_ind], y[np.logical_not(left_ind)]
        return left_x, right_x, left_y, right_y

    def _calc_woe(self, y, smooth_woe):
        n_pos = np.sum(y)
        n_neg = np.float32(len(y)) - n_pos
        woe = np.log((n_pos + smooth_woe) / (n_neg + smooth_woe))
        return woe

    def _split(self, x, y):
        if self._criterion == 'gini':
            splitter = gini
        elif self._criterion == 'entropy':
            splitter = entropy
        else:
            assert callable(self._criterion)

        n_obs = len(y)
        y = y[np.argsort(x)]

        x_info = np.unique(x, return_counts=True)

        impurities = np.zeros(len(x_info[0]) - 1)
        for ind, n_left in enumerate(np.cumsum(x_info[1])[:-1]):
            impurities[ind] = (
                (splitter(y[:n_left],
                          self._smooth_entropy) * n_left +
                 splitter(y[n_left:],
                          self._smooth_entropy) * (n_obs - n_left)) / n_obs)
        thresh_ind = np.argmin(impurities)
        threshold = np.mean(
            x_info[0][[thresh_ind, thresh_ind + 1]])
        return threshold

    def _fit_node(self, x, y,
                  depth, node):

        min_samples = (len(y) > self._min_samples_leaf)
        uniq_x = len(np.unique(x)) > 1
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        min_class = np.all(np.array([n_pos, n_neg]) >= self._min_samples_class)
        max_depth = ((depth < self._max_depth)
                     if self._max_depth is not None else True)

        if (min_samples and min_class and max_depth and uniq_x):
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
            node['woe'] = self._calc_woe(y, self._smooth_woe)
        return self

    def fit(self, x, y):
        x = np.array(x, dtype=self._dtype)
        y = np.array(y, dtype=self._dtype)

        self._fit_node(x, y, depth=0, node=self._tree)
        return self

    def _transform_node(self, x, node):
        if node['type'] == 0:
            if x < node['thresh']:
                return self._transform_node(x, node[0])
            else:
                return self._transform_node(x, node[1])
        return node['woe']

    def transform(self, x):
        if len(self._tree) == 0:
            return "Not trained yet"
        transformed = np.zeros_like(x, dtype=self._dtype)
        for ind in range(len(x)):
            transformed[ind] = self._transform_node(x[ind], self._tree)
        return transformed

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)
