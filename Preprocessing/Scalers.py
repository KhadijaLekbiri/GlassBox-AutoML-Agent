import numpy as np
from core.utils import detect_column_types

class MinMaxScaler:
    def fit(self, X):
        self.min_ = {}
        self.max_ = {}
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            if column_types[i] == "numerical":
                col = X[:, i].astype(float)
                self.min_[i] = np.nanmin(col)
                self.max_[i] = np.nanmax(col)
        return self

    def transform(self, X):
        X_s = X.astype(float).copy()
        column_types = detect_column_types(X)
        for i in range(X_s.shape[1]):
            if column_types[i] == "numerical" and i in self.min_:
                rng = self.max_[i] - self.min_[i] + 1e-8
                X_s[:, i] = (X_s[:, i] - self.min_[i]) / rng
        return X_s

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardScaler:
    def fit(self, X):
        self.mean_ = {}
        self.std_  = {}
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            if column_types[i] == "numerical":
                col = X[:, i].astype(float)
                self.mean_[i] = np.nanmean(col)
                self.std_[i]  = np.nanstd(col) + 1e-8
        return self

    def transform(self, X):
        X_s = X.astype(float).copy()
        column_types = detect_column_types(X)
        for i in range(X_s.shape[1]):
            if column_types[i] == "numerical" and i in self.mean_:
                X_s[:, i] = (X_s[:, i] - self.mean_[i]) / self.std_[i]
        return X_s

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
