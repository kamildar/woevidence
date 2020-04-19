# Author: Kamaldinov Ildar (kamildraf@gmail.com)
# MIT License
import numpy as np
from itertools import chain
from woevidence.criterions import gini, entropy
from sklearn.base import TransformerMixin


class OneFeatureTree(TransformerMixin):

    def __init__(self,
                 criterion='entropy',
                 min_samples_leaf=5,
                 smooth_woe=0.001,
                 min_samples_class=0,
                 max_depth=None,
                 smooth_entropy=1,
                 na_strategy='own',
                 max_bins=255,
                 dtype=None):
        """Weight Of Evidence transformation encoder

        This estimator build tree for feature based
        on criterion and apply Woe transformation defined as

        WOE = log(number of positive obs. / n. of negative)

        Parameters
        ----------
        criterion : str, default='entropy'
            Criterion for building a tree,
            supported: 'gini' and 'entropy'.

        min_samples_leaf : int, default=2
            Minimum number of observation in
            leaf for splitting.

        smooth_woe : float, default=0.001
            Constant for avoiding division on zero
            and log(0).

        min_samples_class : int, default=0,
            Minimum number of observation per class
            for calculating woe.

        max_depth : int or None, default=None
            Maximum depth of three, None means
            unbounded tree.

        smooth_entropy : float, defalut=0.001
            Constant for avoiding log(0).

        na_strategy : str, float, default='own'
            Determine value for missing values.
            if float set na_strategy value to
            woe of NA, 'own' calculate woe for missing
            values or set to zero if there is no missings,
            'min', 'max' strategies set min and max
            woe value respectively

        max_bins : int, default=255
            Number of bins for continious variable.self
            Smaller value speed up computations.
        """
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_class = min_samples_class
        self._smooth_woe = smooth_woe
        self._smooth_entropy = smooth_entropy
        self._max_bins = max_bins
        self._na_strategy = na_strategy
        self._dtype = np.float32 if dtype is None else dtype

        self._tree = dict()
        self._woes = set()
        self._breakpoints = None

    @staticmethod
    def _split_vector(values, threshold, *args):
        """split vector based on value"""
        out = []
        left_ind = values < threshold

        for arg in chain([values], args):
            out.append(arg[left_ind])
            out.append(arg[np.logical_not(left_ind)])
        return out

    @staticmethod
    def _calc_woe(y, sample_weight, smooth_woe):
        """woe calculation"""
        n_pos = np.sum(y * sample_weight)
        n_neg = np.float32(sample_weight.sum() - n_pos)
        woe = np.log((n_pos + smooth_woe) / (n_neg + smooth_woe))
        return woe

    def _set_bins(self, x):
        """calculation breakpoints for splitting"""
        candidates = self._get_bin_candidates(x)
        uniques = np.unique(x)

        if len(candidates) <= len(uniques):
            breakpoints = candidates
        else:
            breakpoints = uniques
        self._breakpoints = breakpoints
        return self

    def _get_bin_candidates(self, x):
        fd_binst = np.histogram(x, bins='fd')[1]
        scott = np.histogram(x, bins='scott')[1]
        doane = np.histogram(x, bins='doane')[1]
        bins = np.histogram(x, bins=self._max_bins)[1]

        candidates = np.unique(
            np.percentile(
                np.concatenate([fd_binst, scott, doane, bins]),
                np.linspace(0, 100, self._max_bins + 2)))[1:-1]
        return candidates

    def _split(self, x, y, sample_weight, **kwargs):
        """threshold for splitting calculation"""
        if self._criterion == 'gini':
            splitter = gini
        elif self._criterion == 'entropy':
            splitter = entropy
        else:
            assert callable(self._criterion)
            splitter = self._criterion

        total_weight = sample_weight.sum()

        # impurites vector
        impurities = np.zeros(len(self._breakpoints))

        # exclude last element
        mask = np.ones_like(x, dtype=bool)
        for ind, brkpoint in enumerate(self._breakpoints):
            mask[mask] = (x[mask] > brkpoint)
            left_weight = sample_weight[mask].sum()

            impurities[ind] = (
                    splitter(y[np.logical_not(mask)],
                             sample_weight[np.logical_not(mask)],
                             **kwargs) * (total_weight - left_weight) +
                    splitter(y[mask], sample_weight[mask], **kwargs) * left_weight)
        thresh_ind = np.argmin(impurities)

        # threshold is middle of two points
        threshold = self._breakpoints[thresh_ind]
        return threshold

    def _fit_node(self, x, y, sample_weight, depth, node):
        """set node to terminal or non-terminal"""
        if self._breakpoints is None:
            self._set_bins(x)

        # values for determine end of recursion
        min_samples = (len(y) > self._min_samples_leaf)
        uniq_x = len(np.unique(x)) > 1
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        min_class = np.all(np.array([n_pos, n_neg]) >= self._min_samples_class)
        max_depth = ((depth < self._max_depth)
                     if self._max_depth is not None else True)

        # whether exit from recursion or not
        if min_samples and min_class and max_depth and uniq_x:
            # zero node type for non-terminal nodes
            node['type'] = 0

            threshold = self._split(x, y, sample_weight, smooth=self._smooth_entropy)
            left_x, right_x, left_y, right_y, left_sw, right_sw = self._split_vector(
                x, threshold, y, sample_weight)

            # 0 -- left_child, 1 -- right child
            node[0] = {}
            node[1] = {}
            node['thresh'] = threshold
            self._fit_node(left_x, left_y, left_sw,
                           depth + 1,
                           node[0])
            self._fit_node(right_x, right_y, right_sw,
                           depth + 1,
                           node[1])
        else:
            # setting node value
            node['type'] = 1
            node['woe'] = self._calc_woe(y, sample_weight, self._smooth_woe)
            self._woes.update([node['woe']])
        return self

    def _filter_na(self, x, y, sample_weight):
        """filted dataset from NA"""
        self._na_woe = None
        na_inds = np.argwhere(np.isnan(x))
        return (np.delete(x, na_inds),
                np.delete(y, na_inds),
                np.delete(sample_weight, na_inds))

    def _handle_na(self, x, y, sample_weight):
        """set woe value for NA"""
        na_inds = np.argwhere(np.isnan(x))

        if self._na_strategy == 'own':
            # calc woe for NA if there are NAs
            if len(na_inds) != 0:
                self._na_woe = self._calc_woe(y[na_inds], sample_weight[na_inds],
                                              self._smooth_woe)
                self._woes.update([self._na_woe])
            # set woe for NA to zero if there is no NA
            else:
                self._na_woe = 0
                self._woes.update([self._na_woe])
        elif self._na_strategy == 'min':
            self._na_woe = min(self._woes)
        elif self._na_strategy == 'max':
            self._na_woe = max(self._woes)
        else:
            raise AttributeError("{} is not valid strategy for missings"
                                 .format(self._na_strategy))

        return None

    def fit(self, x, y, sample_weight):
        # initialize tree and woes
        """start recursion to calculate woe tree"""

        x = np.array(x, dtype=self._dtype).ravel()
        y = np.array(y, dtype=self._dtype).ravel()
        sample_weight = np.array(sample_weight, dtype=self._dtype).ravel()
        
        fltr_x, fltr_y, fltr_sw = self._filter_na(x=x, y=y,
                                                  sample_weight=sample_weight)

        self._fit_node(fltr_x, fltr_y, fltr_sw,
                       depth=0, node=self._tree)
        self._handle_na(x, y, sample_weight)
        return self

    def _transform_node(self, x, node):
        """transform feature value to woe"""
        if node['type'] == 0:
            if x < node['thresh']:
                return self._transform_node(x, node[0])
            elif np.isnan(x):
                return self._na_woe
            else:
                return self._transform_node(x, node[1])
        return node['woe']

    def transform(self, x):
        """transform feature"""
        if len(self._tree) == 0:
            return "Not trained yet"
        transformed = np.zeros_like(x, dtype=self._dtype)
        for ind in range(len(x)):
            transformed[ind] = self._transform_node(x[ind], self._tree)
        return transformed

    def fit_transform(self, x, y, sample_weight):
        self.fit(x, y, sample_weight)
        return self.transform(x)
