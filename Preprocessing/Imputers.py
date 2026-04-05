import numpy as np

from core.utils import detect_column_types

class SimpleImputer:
    def __init__(self, strategy='mean'): # default one is mean
        self.strategy = strategy
        self.statistics_ = {}

    def _numpy_mode(self, col):
        col_clean = col[col != ""] if col.dtype.kind in ("U", "S", "O") else col[~np.isnan(col.astype(float))]
        values, count = np.unique(column_no_nan, return_counts=True)
        return values[np.argmax(count)]
        
    def fit(self, X):
        """
        X         : 2D NumPy array
        col_types : list of 'numerical' or 'categorical' per column
        """
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if column_types[i] == 'numerical':
                col_f = col.astype(float)
                if self.strategy == 'mean':
                    self.statistics_[i] = np.nanmean(col_f)
                elif self.strategy == 'median':
                    self.statistics_[i] = np.nanmedian(col_f)
                else:
                    self.statistics_[i] = self._numpy_mode(col_f)
            else:
                self.statistics_[i] = self._numpy_mode(col)
        return self

    def transform(self, X):
        X_imputed = X.copy().astype(object)
        column_types = detect_column_types(X)
        for i, value in self.statistics_.items():
            col = X_imputed[:, i]
            if column_types[i] == 'numerical':
                for j, v in enumerate(col):
                    try:
                        if np.isnan(float(v)):
                            col[j] = value
                    except (ValueError, TypeError):
                        col[j] = value
            else:
                col[col == ''] = value
            X_imputed[:, i] = col
        return X_imputed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

  
