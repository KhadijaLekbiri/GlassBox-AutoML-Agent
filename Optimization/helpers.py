import numpy as np 
def evaluate_model(model, X, y, kf, metric):
    scores = []
    for train_idx, test_idx in kf.get_splits(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        score = metric(y[test_idx], preds)
        scores.append(score)
    return np.mean(scores)