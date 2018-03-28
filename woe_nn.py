import numpy as np
from one_feature_nn import OneFeatureNN


class WoeNN(object):

    def __init__(self, k,
                 n_jobs=1,
                 dtype=np.float32,
                 smooth_woe=0.001):
        self._k = k
        self._n_jobs = n_jobs
        self._dtype = dtype
        self._smooth_woe = smooth_woe

        self._nns = []

    def _to_arglist(self, arg, shape):
        if isinstance(arg, list):
            return arg
        else:
            return [arg] * shape

    def fit(self, X, y):
        n_features = X.shape[1]
        k = self._to_arglist(self._k, n_features)
        n_jobs = self._to_arglist(self._n_jobs, n_features)
        smooth_woe = self._to_arglist(self._smooth_woe, n_features)

        for feature in range(n_features):
            cpy_tar = True if (feature == 0) else False
            self._nns.append(
                OneFeatureNN(k=k[feature], n_jobs=n_jobs[feature],
                             smooth_woe=smooth_woe[feature])
            )
            self._nns[feature].fit(X[:, feature], y, copy_target=cpy_tar)
        return self

    def transform(self, X):
        transformed = np.zeros_like(X, dtype=self._dtype)
        for ind in range(X.shape[1]):
            transformed[:, ind] = self._nns[
                ind].transform(X[:, ind], y=self._nns[0]._target,
                               k=self._k, n_jobs=self._n_jobs).ravel()
        return transformed

    def fit_transfrom(self, X, y):
        self.fit(X, y)
        return self.transform(X)
