import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def plot_roc(model, X_test, y_test, path):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(path)
    plt.close()

def plot_pr(model, X_test, y_test, path):
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(path)
    plt.close()
