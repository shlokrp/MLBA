import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame):
    df = df.copy()

    # Example cleanup steps (customize based on real columns)
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday

    # Drop impossible negative delays
    if 'delay_minutes' in df.columns:
        df = df[df['delay_minutes'] >= 0]

    df = df.dropna()

    return df

def split_for_tasks(df, target_class='delayed', target_reg='delay_minutes', test_size=0.2):
    X = df.drop(columns=[target_class, target_reg])
    y_class = df[target_class]
    y_reg = df[target_reg]

    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=test_size, random_state=42, stratify=y_class
    )

    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
