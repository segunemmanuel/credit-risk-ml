# score.py — batch scoring for the credit risk model
import argparse, json
from pathlib import Path
import pandas as pd
import joblib

def load_model(model_path="models/credit_pipe.pkl"):
    m = joblib.load(model_path)
    # If it's a CalibratedClassifierCV, use directly; predict_proba exists.
    return m

def load_threshold(meta_path="models/metadata.json", default=0.5):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # use cost-optimal threshold if you later add it; else fall back to 0.5
        return float(meta.get("threshold_cost", default))
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser(description="Score applicants with trained credit risk model")
    ap.add_argument("--input", required=True, help="Path to applicants CSV (no 'default' column)")
    ap.add_argument("--output", default="predictions.csv", help="Where to save predictions CSV")
    ap.add_argument("--model", default="models/credit_pipe.pkl", help="Path to .pkl model")
    ap.add_argument("--meta", default="models/metadata.json", help="Path to metadata.json")
    args = ap.parse_args()

    print(f">>> Loading model: {args.model}")
    model = load_model(args.model)
    thr = load_threshold(args.meta, default=0.5)
    print(f">>> Using decision threshold: {thr:.3f}")

    print(f">>> Reading input: {args.input}")
    X = pd.read_csv(args.input)

    # Sanity: refuse if 'default' exists accidentally
    if "default" in X.columns:
        raise ValueError("Input CSV must NOT contain a 'default' column for scoring.")

    print(">>> Predicting probabilities…")
    pd_scores = model.predict_proba(X)[:, 1]  # PD = probability of default
    out = X.copy()
    out["PD"] = pd_scores

    # Simple decision bands
    approve_cut = min(thr * 0.6, 0.30)  # adjustable heuristic
    bins = [0.0, approve_cut, thr, 1.0]
    labels = ["Approve", "Refer", "Decline"]
    out["Decision"] = pd.cut(out["PD"], bins=bins, labels=labels, include_lowest=True)

    print(">>> Preview:")
    print(out.head(10))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f">>> Saved predictions → {out_path.resolve()}")

if __name__ == "__main__":
    main()
