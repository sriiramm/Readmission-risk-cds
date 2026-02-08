import json
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from .config import MODELS_DIR, REPORTS_DIR
from .utils import ensure_dirs

def pick_threshold_cost(y_true, y_prob, c_fn=5000, c_fp=200):
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"thr": 0.5, "cost": float("inf")}
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * c_fn + fp * c_fp
        if cost < best["cost"]:
            best = {"thr": float(thr), "cost": float(cost), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best

def projected_savings(tp, fp, cost_readmit, cost_intervene, effectiveness, avoidable_fraction=1.0):
    """
    tp, fp can be ints or floats (e.g., daily averages).
    effectiveness: fraction of avoidable readmissions prevented among selected true positives.
    avoidable_fraction: fraction of readmissions considered avoidable (realism knob).
    """
    tp = float(tp)
    fp = float(fp)

    prevented = tp * float(avoidable_fraction) * float(effectiveness)
    gross_savings = prevented * float(cost_readmit)
    intervention_cost = (tp + fp) * float(cost_intervene)
    net_savings = gross_savings - intervention_cost

    return {
        "prevented_readmissions": prevented,
        "gross_savings": gross_savings,
        "intervention_cost": intervention_cost,
        "net_savings": net_savings,
    }


def evaluate_model(model_path, X_test, y_test,
                   c_fn=5000, c_fp=200,
                   cost_readmit=12000, cost_intervene=300, effectiveness=0.25):
    model = joblib.load(model_path)
    prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, prob)
    pr = average_precision_score(y_test, prob)

    best = pick_threshold_cost(y_test.values, prob, c_fn=c_fn, c_fp=c_fp)
    savings = projected_savings(best["tp"], best["fp"], cost_readmit, cost_intervene, effectiveness)

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold": best,
        "savings": savings
    }

def save_report(report: dict, filename: str):
    ensure_dirs([REPORTS_DIR])
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
