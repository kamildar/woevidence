import numpy as np


class MeanEncoder(object):
    def __init__(self, alpha):
        self._alpha = alpha
        self._total_mean = None
        self._mean_map = {}

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        sample_weight = sample_weight if sample_weight is not None else np.ones(len(y))
        self._save_total_mean(y, sample_weight)
        uniques = np.unique(x)
        for cat in uniques:
            idx = (x == cat)
            self._mean_map[cat] = self.calc_value(y[idx], sample_weight[idx])
        return self

    def calc_value(self, y, sample_weight):
        return np.average(
            [self._total_mean, y.mean()],
            weights=[self._alpha, sample_weight.sum()])

    def transform(self, x):
        return np.array([self._mean_map[val] for val in x])

    def fit_transform(self, x, y, sample_weight=None):
        self.fit(x, y, sample_weight)
        return self.transform(x)

    def _save_total_mean(self, y, sample_weight):
        self._total_mean = np.average(y, weights=sample_weight)
