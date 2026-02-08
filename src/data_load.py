from pathlib import Path
import pandas as pd

# Repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset lives at repo root
DATA_RAW = PROJECT_ROOT / "diabetic_data.csv"


def load_raw():
    return pd.read_csv(DATA_RAW)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace '?' markers with NaN
    df = df.replace("?", np.nan)

    # Binary target: y=1 if <30 else 0 (NO or >30)
    df["y"] = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)

    # Drop identifiers
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    return df
