import numpy as np

class MinMaxScaler:
    def fit(self, X):
        self.min_ = {}
        self.max_ = {}
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if column_types[i] == 'numerical':
                self.min_[i] = np.nanmin(col)
                self.max_[i] = np.nanmax(col)

        return self

    def transform(self, X):
        X_scaled = X.copy()
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if column_types[i] == 'numerical':
                X_scaled[:, i] = (col - self.min_[i]) / (self.max_[i] - self.min_[i]) if self.max_[i] != self.min_[i] else 0
        return X_scaled

    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class StandardScaler:
    def fit(self, X):
        self.mean_ = {}
        self.std_ = {}
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if column_types[i] == 'numerical':
                self.mean_[i] = np.nanmean(col)
                self.std_[i] = np.nanstd(col)

        return self

    def transform(self, X):
        X_scaled = X.copy()
        column_types = detect_column_types(X)
        for i in range(X.shape[1]):
            col = X_scaled[:, i]
            if column_types[i] == 'numerical':
                X_scaled[:, i] = (col - self.mean_[i]) / self.std_[i] if self.std_[i] != 0 else 0
        return X_scaled

    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 
  