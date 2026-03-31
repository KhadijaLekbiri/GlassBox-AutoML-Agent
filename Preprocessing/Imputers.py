import numpy as np

class SimpleImputer:
    def __init__(self, strategy='mean'): # default one is mean
        self.strategy = strategy
        self.statistics_ = {}

    def _numpy_mode(self, col):
        column_no_nan = col[col != ''] if col.dtype.type == np.str_ else col[~np.isnan(col)]

        values, count = np.unique(column_no_nan, return_counts=True)
        return values[np.argmax(count)]
        
    def fit(self, X):
        """
        X         : 2D NumPy array
        col_types : list of 'numerical' or 'categorical' per column
        """
        for i in range(X.shape[1]):
            col = X[:, i]
            column_types = detect_column_types(X)
            if column_types[i] == 'numerical':
                if self.strategy == 'mean':
                    self.statistics_[i] = np.nanmean(col)
                elif self.strategy == 'median':
                    self.statistics_[i] = np.nanmedian(col)
                else:
                    raise ValueError("Invalid strategy for numerical values. Use 'mean', 'median'.")
            else:
                self.statistics_[i] = self._numpy_mode(col)

    def transform(self, X):
        X_imputed = X.copy()
        column_types = detect_column_types(X)
        for i, value in self.statistics_.items():
            col = X_imputed[:, i]
            if column_types[i] == 'numerical':
                col[np.isnan(col)] = value
            else:
                col[col == ''] = value
        return X_imputed
    
    def fit_transform(self, X, col_types):
        self.fit(X, col_types)
        return self.transform(X)

  