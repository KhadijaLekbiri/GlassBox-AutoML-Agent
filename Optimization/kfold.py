import numpy as np 
class KFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop
        return folds
    def get_splits(self, X):
        folds = self.split(X)
        for i in range(self.n_splits):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_idx, test_idx
            