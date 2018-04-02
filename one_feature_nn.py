import numpy as np
from scipy.spatial import cKDTree


class OneFeatureNN(object):

    def __init__(self, k,
                 n_jobs=1,
                 dtype=np.float32,
                 smooth_woe=0.001):
        self._n_jobs = n_jobs
        self._k = k
        self._dtype = dtype
        self._smooth_woe = smooth_woe
        pass

        self._tree = None

    def _calc_woe(self, y, smooth_woe):
        n_pos = np.sum(y)
        n_neg = np.float32(len(y)) - n_pos
        woe = np.log((n_pos + smooth_woe) / (n_neg + smooth_woe))
        return woe

    def fit(self, x, y, copy_target=True):
        try:
            x.shape[1]
        except IndexError:
            x = x[:, np.newaxis]
        self._tree = cKDTree(x, balanced_tree=False)

        if copy_target:
            self._target = y
        return self

    def transform(self, x, y=None, k=None, n_jobs=None):
        try:
            x.shape[1]
        except IndexError:
            x = x[:, np.newaxis]

        if y is None:
            y = self._target

        if n_jobs is None:
            n_jobs = self._n_jobs
        if k is None:
            k = self._k

        transformed = np.zeros_like(x, dtype=np.float32)
        for i, obs in enumerate(x):
            inds = self._tree.query(obs, k)[1]
            transformed[i] = self._calc_woe(
                y[inds].ravel(), self._smooth_woe)
        return transformed

    def fit_transform(self, x, y, k=None, n_jobs=None):
        self.fit(x, y)
        return self.transform(x, y, k=k, n_jobs=n_jobs)
