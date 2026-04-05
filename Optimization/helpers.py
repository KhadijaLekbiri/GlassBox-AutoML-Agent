import numpy as np
from core.utils import detect_task
from core.metrics import accuracy, f1_score, mse, mae, r2_score
def evaluate_model(model, X, y, kf, metric=None):
    """
    Evaluates a model using K-Fold CV.
    If metric is None, it will auto-detect classification vs regression.
    """
    if metric is None:
        task = detect_task(y)
        metric = accuracy if task == "classification" else mse
    scores = []
    for train_idx, test_idx in kf.get_splits(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(metric(y[test_idx], preds))
    return float(np.mean(scores))
