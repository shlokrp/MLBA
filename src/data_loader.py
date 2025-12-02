import pandas as pd
from pathlib import Path

def load_data(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    # Supports Excel or CSV
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return df
