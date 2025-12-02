from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
    }
