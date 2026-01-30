from src.data_load import load_raw, basic_clean
from src.preprocess import make_splits
from src.config import RANDOM_STATE, MODELS_DIR
from src.evaluate import evaluate_model, save_report

def main():
    df = basic_clean(load_raw())
    splits = make_splits(df, target="y", random_state=RANDOM_STATE)

    # Business assumptions (adjustable later)
    c_fn = 5000   # cost of missing a true readmission (proxy)
    c_fp = 200    # cost of unnecessary outreach
    cost_readmit = 12000
    cost_intervene = 300
    effectiveness = 0.25

    xgb_report = evaluate_model(
        MODELS_DIR / "xgb_model.joblib",
        splits.X_test,
        splits.y_test,
        c_fn=c_fn,
        c_fp=c_fp,
        cost_readmit=cost_readmit,
        cost_intervene=cost_intervene,
        effectiveness=effectiveness
    )

    path = save_report(xgb_report, "xgb_evaluation_report.json")
    print("Saved report:", path)
    print("ROC-AUC:", xgb_report["roc_auc"])
    print("PR-AUC:", xgb_report["pr_auc"])
    print("Best threshold:", xgb_report["threshold"])
    print("Projected savings (test cohort):", xgb_report["savings"])

if __name__ == "__main__":
    main()
