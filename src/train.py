import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import MODELS_DIR, RANDOM_STATE
from .data_load import load_raw, basic_clean
from .preprocess import make_splits, build_preprocessor
from .utils import ensure_dirs

def train_logreg(preprocessor, X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgb(preprocessor, X_train, y_train):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe

def main():
    ensure_dirs([MODELS_DIR])

    df = basic_clean(load_raw())
    splits = make_splits(df, target="y", random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(splits.X_train)

    logreg_pipe = train_logreg(preprocessor, splits.X_train, splits.y_train)
    xgb_pipe = train_xgb(preprocessor, splits.X_train, splits.y_train)

    joblib.dump(logreg_pipe, MODELS_DIR / "baseline_logreg.joblib")
    joblib.dump(xgb_pipe, MODELS_DIR / "xgb_model.joblib")

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"random_state": RANDOM_STATE}, f, indent=2)

    print("✅ Models saved in:", MODELS_DIR)

if __name__ == "__main__":
    main()
