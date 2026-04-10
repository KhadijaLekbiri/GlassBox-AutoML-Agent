import numpy as np
from Preprocessing.Imputers  import SimpleImputer
from Preprocessing.Scalers   import StandardScaler
from Preprocessing.Encoders  import LabelEncoder, OneHotEncoder
from Preprocessing.outliers import detect_outliers
from core.utils import detect_column_types

class Preprocessor:
    def __init__(self, scaler="standard", imputer_strategy="mean"):
        self.imputer  = SimpleImputer(strategy=imputer_strategy)
        self.scaler   = StandardScaler() if scaler == "standard" else None
        self.encoders = {}          
        self.feature_names_out = []
    
    def fit_transform(self, X, y, feature_names=None):
        """
        Parameters
        ----------
        X             : raw 2-D NumPy array (may contain strings / NaNs)
        y             : 1-D target array
        feature_names : list of column names (optional)

        Returns
        -------
        X_out         : clean float NumPy array
        y_out         : float/int NumPy array
        names_out     : list of output feature names
        """
        n_cols = X.shape[1]
        if feature_names is None:
            feature_names = [f"col_{i}" for i in range(n_cols)]

        # 1 — impute missing values
        X = self.imputer.fit_transform(X)

        col_types = detect_column_types(X)
        num_indices = [i for i, t in enumerate(col_types) if t == "numerical"]
        X[:, num_indices] = self._handle_outliers(X[:, num_indices])

        out_cols   = []
        names_out  = []

        for i in range(n_cols):
            col  = X[:, i]
            name = feature_names[i]

            if col_types[i] == "numerical":
                out_cols.append(col.astype(float).reshape(-1, 1))
                names_out.append(name)

            elif col_types[i] == "boolean":
                out_cols.append(col.astype(float).reshape(-1, 1))
                names_out.append(name)

            elif col_types[i] == "categorical":
                unique_vals = np.unique(col[col != ""])
                if len(unique_vals) <= 2:
                    # binary → label encode
                    enc = LabelEncoder()
                    enc.fit(col)
                    out_cols.append(enc.transform(col).astype(float).reshape(-1, 1))
                    names_out.append(name)
                    self.encoders[i] = enc
                else:
                    # nominal → one-hot
                    enc = OneHotEncoder()
                    enc.fit(col)
                    ohe = enc.transform(col).astype(float)
                    out_cols.append(ohe)
                    names_out.extend([f"{name}_{cat}" for cat in enc.categories_])
                    self.encoders[i] = enc

        X_out = np.hstack(out_cols).astype(float)
        
        # 4 — scale numerical columns
        if self.scaler:
            X_out = self.scaler.fit_transform(X_out)
        y_out = self._clean_y(y)

        self.feature_names_out = names_out
        return X_out, y_out, names_out
    def _handle_outliers(self, X):
        """
        Replace outliers using IQR clipping.
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Clip values
        X_clipped = np.clip(X, lower_bound, upper_bound)

        return X_clipped
    def _clean_y(self, y):
        try:
            return y.astype(float)
        except (ValueError, AttributeError):
            enc = LabelEncoder()
            enc.fit(y)
            return enc.transform(y).astype(float)
