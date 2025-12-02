from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_regressor(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "preds": preds.tolist()
    }
