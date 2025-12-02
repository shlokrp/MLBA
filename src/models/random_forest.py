from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def build_rf_classifier():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced"
    )

def build_rf_regressor():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12
    )
