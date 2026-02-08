import joblib
import numpy as np

from src.data_load import load_raw, basic_clean
from src.preprocess import make_splits
from src.config import RANDOM_STATE, MODELS_DIR
from src.evaluate import save_report


def simulate_capacity_policy(probs, y_true, cost_readmit, cost_intervene, effectiveness, k):
    """
    Rank patients by predicted risk and intervene on the top-k only.
    Returns TP/FP for the intervened set plus ROI.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    # Indices of top-k risks
    top_idx = np.argsort(-probs)[:k]
    selected_y = y_true[top_idx]

    tp = int((selected_y == 1).sum())
    fp = int((selected_y == 0).sum())

    prevented = tp * effectiveness
    gross_savings = prevented * cost_readmit
    intervention_cost = (tp + fp) * cost_intervene
    net_savings = gross_savings - intervention_cost

    return {
        "k": int(k),
        "tp": tp,
        "fp": fp,
        "prevented_readmissions": float(prevented),
        "gross_savings": float(gross_savings),
        "intervention_cost": float(intervention_cost),
        "net_savings": float(net_savings),
    }


def main():
    # --- Business assumptions (editable later) ---
    COST_READMISSION = 12000      # $ per readmission
    COST_INTERVENTION = 300       # $ per outreach / care coordination
    EFFECTIVENESS = 0.25          # % prevented among true positives intervened

    # --- Operational reality: staffing capacity ---
    K_OPTIONS = [10, 20, 30, 40, 50, 75, 100]

    df = basic_clean(load_raw())
    splits = make_splits(df, target="y", random_state=RANDOM_STATE)

    model = joblib.load(MODELS_DIR / "xgb_model.joblib")
    probs = model.predict_proba(splits.X_test)[:, 1]
    y_true = splits.y_test.values

    results = []
    for k in K_OPTIONS:
        results.append(
            simulate_capacity_policy(
                probs, y_true,
                COST_READMISSION, COST_INTERVENTION, EFFECTIVENESS,
                k
            )
        )

    # Best K by max net savings on the test cohort
    best = max(results, key=lambda r: r["net_savings"])

    report = {
        "assumptions": {
            "cost_readmission": COST_READMISSION,
            "cost_intervention": COST_INTERVENTION,
            "effectiveness": EFFECTIVENESS,
        },
        "policy": "Top-K by predicted risk (capacity constrained)",
        "candidates": results,
        "best": best,
        "test_size": int(len(splits.X_test)),
        "positive_rate_test": float(np.mean(y_true)),
    }

    path = save_report(report, "capacity_policy_report.json")

    print("Saved report:", path)
    print("Best capacity policy (top-K):", best)


if __name__ == "__main__":
    main()
