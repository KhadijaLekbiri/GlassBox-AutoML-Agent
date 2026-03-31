import numpy as np

class LabelEncoder:
    def __init__(self, categories=None):
        """
        categories: list defining order (e.g. ['low', 'medium', 'high'])
        """
        self.categories = categories
        self.mapping_ = {}

    def fit(self, col):
        if self.categories is not None:
            self.mapping_ = {cat: i for i, cat in enumerate(self.categories)}
        else:
            unique_vals = np.unique(col[col != ''])
            self.mapping_ = {val: i for i, val in enumerate(unique_vals)}
        return self

    def transform(self, col):
        return np.array([self.mapping_.get(val, -1) for val in col])

    def fit_transform(self, col):
        self.fit(col)
        return self.transform(col)
    
class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, col):
        self.categories_ = np.unique(col[col != ''])
        return self
    
    def transform(self, col):
        one_hot = np.zeros((len(col), len(self.categories_)), dtype=int)
        for i, category in enumerate(self.categories_):
            one_hot[:, i] = (col == category).astype(int)
        return one_hot
    
    def fit_transform(self, col):
        self.fit(col)
        return self.transform(col)
    
class Encoding:
    def __init__(self, category_type, categories=None):
        self.category_type = category_type
        self.categories = categories
        self.encoder_ = None

    def fit(self, col):
        if self.category_type == 'ordinal':
            self.encoder_ = LabelEncoder(categories=self.categories)
        elif self.category_type == 'nominal':
            self.encoder_ = OneHotEncoder()
        else:
            raise ValueError("Invalid category type. Use 'ordinal' or 'nominal'.")
        self.encoder_.fit(col)
        return self

    def transform(self, col):
        return self.encoder_.transform(col)

    def fit_transform(self, col):
        self.fit(col)
        return self.transform(col)
