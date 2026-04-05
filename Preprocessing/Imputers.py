import numpy as np
from core.utils import detect_column_types

class SimpleImputer:
    def __init__(self, strategy="mean"):
        self.strategy    = strategy
        self.statistics_ = {}

    def _mode(self, col):
        col_clean = col[col != ""] if col.dtype.kind in ("U", "S", "O") else col[~np.isnan(col.astype(float))]
        values, counts = np.unique(col_clean, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X):
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if column_types[i] == "numerical":
                col_f = col.astype(float)
                if self.strategy == "mean":
                    self.statistics_[i] = np.nanmean(col_f)
                elif self.strategy == "median":
                    self.statistics_[i] = np.nanmedian(col_f)
                else:
                    self.statistics_[i] = self._mode(col_f)
            else:
                self.statistics_[i] = self._mode(col)
        return self

    def transform(self, X):
        X_out = X.copy().astype(object)
        column_types = detect_column_types(X)
        for i, fill in self.statistics_.items():
            col = X_out[:, i]
            if column_types[i] == "numerical":
                for j, v in enumerate(col):
                    try:
                        if np.isnan(float(v)):
                            col[j] = fill
                    except (ValueError, TypeError):
                        col[j] = fill
            else:
                col[col == ""] = fill
            X_out[:, i] = col
        return X_out

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
