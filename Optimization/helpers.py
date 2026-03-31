from core.utils import detect_task
def evaluate_model(model, X, y, kf, metric=None):
    """
    Evaluates a model using K-Fold CV.
    If metric is None, it will auto-detect classification vs regression.
    """
    # Detect task type if metric is not provided
    if metric is None:
        task = detect_task(y)
        if task == "classification":
            metric = f1_score
        else:
            metric = mse

    scores = []
    for train_idx, test_idx in kf.get_splits(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        score = metric(y[test_idx], preds)
        scores.append(score)

    return np.mean(scores)