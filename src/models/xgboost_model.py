from xgboost import XGBClassifier, XGBRegressor

def build_xgb_classifier():
    return XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=250,
        eval_metric='logloss'
    )

def build_xgb_regressor():
    return XGBRegressor(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        objective='reg:squarederror'
    )
