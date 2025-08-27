# make_shap.py — robust SHAP generator
import joblib, pandas as pd
from pathlib import Path
from src.data import load_csv
from src.models import build_lgbm_pipeline
from src.explain import shap_global, shap_local

from sklearn.utils.validation import check_is_fitted

DATA = "data/processed/uci_credit_default.csv"
MODEL_PATH = "models/credit_pipe.pkl"
OUT = "reports/figures"

print(">>> Loading data…")
X, y, num_cols, cat_cols = load_csv(DATA)
print(f"X={X.shape} y={len(y)}")

print(">>> Loading model…")
model = joblib.load(MODEL_PATH)

# Unwrap calibrators, otherwise use as-is
inner = model
if hasattr(inner, "base_estimator"):   # scikit-learn < 1.4
    inner = inner.base_estimator
elif hasattr(inner, "estimator"):      # scikit-learn >= 1.4
    inner = inner.estimator

# If not a Pipeline, or not fitted, fit a fresh pipeline for SHAP visuals
needs_fit = False
if not hasattr(inner, "named_steps"):
    print("Model is not a Pipeline; will fit a fresh pipeline for visuals.")
    inner = build_lgbm_pipeline(num_cols, cat_cols)
    needs_fit = True
else:
    # Check if the preprocessor is fitted
    try:
        check_is_fitted(inner.named_steps["pre"])
    except Exception:
        needs_fit = True

if needs_fit:
    print(">>> Fitting pipeline for SHAP visuals…")
    inner.fit(X, y)

Path(OUT).mkdir(parents=True, exist_ok=True)
print(">>> Generating SHAP figures…")
sample = X.sample(min(500, len(X)), random_state=42)
shap_global(inner, sample, OUT)
shap_local(inner, X.iloc[[0]], OUT, idx=0)
print(">>> Done. Check:", OUT)
