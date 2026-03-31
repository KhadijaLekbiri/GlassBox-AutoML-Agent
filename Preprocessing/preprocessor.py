import numpy as np

class Preprocessor:
    def __init__(self, imputer=None, encoders=None, scaler=None):
        """
        imputer  : instance of your SimpleImputer
        encoders : dict {column_index: Encoding instance}
        scaler   : instance of your StandardScaler or MinMaxScaler
        """
        self.imputer_ = imputer
        self.encoders_ = encoders or {}
        self.scaler_ = scaler

    def fit(self, X):
        if self.imputer_:
            X = self.imputer_.fit(X)

        encoded_cols = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if i in self.encoders_:
                encoder = self.encoders_[i]
                col_encoded = encoder.fit_transform(col)
                if col_encoded.ndim == 1:
                    col_encoded = col_encoded.reshape(-1, 1)
                encoded_cols.append(col_encoded)
            else:
                encoded_cols.append(col.reshape(-1, 1))
        X = np.hstack(encoded_cols)

        if self.scaler_:
            self.scaler_.fit(X)

        return self

    def transform(self, X):
        if self.imputer_:
            X = self.imputer_.transform(X)

        encoded_cols = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if i in self.encoders_:
                col_encoded = self.encoders_[i].transform(col)
                if col_encoded.ndim == 1:
                    col_encoded = col_encoded.reshape(-1, 1)
                encoded_cols.append(col_encoded)
            else:
                encoded_cols.append(col.reshape(-1, 1))
        X = np.hstack(encoded_cols)

        if self.scaler_:
            X = self.scaler_.transform(X)

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
