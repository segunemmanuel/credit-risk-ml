# Credit Risk Scoring with Explainability (Starter Repo)

An advanced yet laptop-friendly end‑to‑end project: **predict probability of default (PD)** on consumer credit and **explain** each decision with SHAP. Includes calibration, thresholding by cost, Optuna tuning, MLflow tracking, and a Streamlit demo.

---

## 📁 Project Structure
```
credit-risk/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ reports/
│  └─ figures/
├─ models/
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ features.py
│  ├─ models.py
│  ├─ explain.py
│  └─ utils.py
├─ train.py
└─ app/
   └─ app.py
```

---

## 🚀 Quickstart
```bash
# 1) Create env & install deps
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt

# 2) Fetch the UCI Credit Default dataset & make a clean CSV
python -m src.data --download --to data/raw --make-clean-csv data/processed/uci_credit_default.csv

# 3) Train (with CV, calibration, SHAP figures, MLflow optional)
python train.py --data data/processed/uci_credit_default.csv --calibrate --save-model models/credit_pipe.pkl

# 4) Run demo app
streamlit run app/app.py
```

> If the auto-download fails, the script prints an alternate manual link hint and expects you to place the file under `data/raw/`.

---

## 🧱 README.md
```markdown
# Credit Risk Scoring with Explainability

This repo predicts **probability of default (PD)** on the UCI Credit Default dataset and provides **transparent explanations** using SHAP. The pipeline is production‑lean: reproducible splits, CV, cost‑aware thresholds, calibration, and a Streamlit app for interactive scoring.

## Main Features
- LightGBM pipeline with preprocessing (OHE + scaling), class imbalance handling, and CV.
- **Calibration** (isotonic) + calibration curves & Brier score.
- **Threshold selection** by expected cost (tunable FN/FP costs) and KS metric.
- **Explainability**: global (summary) + local (waterfall) SHAP plots.
- **Optuna** tuning (optional) and **MLflow** tracking (optional).
- Streamlit app: upload CSV → PD + decision band + local explanation.

## Data
- UCI "Default of Credit Card Clients" (~30k rows). The `src.data` module auto-downloads the Excel, cleans columns, and outputs a CSV.

## Usage
See the Quickstart in the root of this document.

## Deliverables for a Portfolio
- `reports/figures/` (SHAP, calibration, ROC, PR curves).
- `models/credit_pipe.pkl` (fitted pipeline) and `models/metadata.json` (metrics & thresholds).
- A short model card in the README: data window, metrics, intended use, fairness notes.

## Notes
- Ensure Python ≥ 3.10. Laptop‑friendly (CPU OK). For GPU, LightGBM GPU build is optional.
```

---

## 📦 requirements.txt
```txt
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=1.7
imbalanced-learn>=0.12
matplotlib>=3.7
seaborn>=0.13
shap>=0.44
scikit-plot>=0.3.7
optuna>=3.6
mlflow>=2.12
requests>=2.31
openpyxl>=3.1
joblib>=1.3
streamlit>=1.36
evidently>=0.4.28
```

---

## 🪣 .gitignore
```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
models/*.pkl
models/mlruns/
mlruns/
.DS_Store
*.png
*.svg
*.pdf
.env
```

---

## 🧩 src/__init__.py
```python
# empty for package discovery
```

---

## 🧩 src/utils.py
```python
from __future__ import annotations
import json, os, random
import numpy as np
from dataclasses import dataclass

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)

@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    brier: float
    ks: float

    def to_dict(self):
        return {"roc_auc": self.roc_auc, "pr_auc": self.pr_auc, "brier": self.brier, "ks": self.ks}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(d: dict, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def ks_stat(y_true, y_prob):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    return float(ks)

def pick_threshold_cost(y_true, y_prob, fn_cost=5.0, fp_cost=1.0):
    import numpy as np
    ts = np.linspace(0, 1, 1001)
    losses = []
    for t in ts:
        pred = (y_prob >= t)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        losses.append(fp * fp_cost + fn * fn_cost)
    t_best = float(ts[int(np.argmin(losses))])
    return t_best
```

---

## 🧩 src/data.py
```python
from __future__ import annotations
import argparse, io, os, sys
import pandas as pd
import requests
from .utils import ensure_dir

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
)

RENAMES = {
    "default payment next month": "default",
    "PAY_0": "PAY_1",  # normalize naming (dataset sometimes uses PAY_0)
}

DROP_COLS = ["ID"]

CAT_CANDIDATES = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1,7)]


def download_uci(to_dir: str) -> str:
    ensure_dir(to_dir)
    out_path = os.path.join(to_dir, "default_of_credit_card_clients.xls")
    try:
        r = requests.get(UCI_URL, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded to {out_path}")
        return out_path
    except Exception as e:
        print("\n[WARN] Auto-download failed:", e)
        print("If this persists, manually download the UCI Excel file and place it under data/raw/ then rerun --make-clean-csv.")
        return out_path  # may not exist yet


def make_clean_csv(excel_path: str, out_csv: str):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found at {excel_path}")
    df = pd.read_excel(excel_path, header=1)
    # Standardize columns
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=RENAMES, inplace=True)
    for c in df.columns:
        if c.upper() == c:
            # keep as-is for known fields
            pass
    if any(col in df.columns for col in DROP_COLS):
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    # Lowercase target name
    if "default" not in df.columns:
        raise ValueError("Target 'default' column not found after renaming.")
    # Save processed
    ensure_dir(os.path.dirname(out_csv))
    df.to_csv(out_csv, index=False)
    print(f"Wrote clean CSV: {out_csv}  shape={df.shape}")


def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df["default"].astype(int).values
    X = df.drop(columns=["default"])
    # Identify categoricals
    cat_cols = [c for c in CAT_CANDIDATES if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download UCI Excel to data/raw/")
    p.add_argument("--to", default="data/raw", help="Download directory")
    p.add_argument("--make-clean-csv", default=None, help="Output CSV path (from Excel)")
    p.add_argument("--excel", default="data/raw/default_of_credit_card_clients.xls", help="Excel path if already downloaded")
    args = p.parse_args()

    if args.download:
        download_uci(args.to)
    if args.make_clean_csv:
        make_clean_csv(args.excel, args.make_clean_csv)
```

---

## 🧩 src/features.py
```python
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num_cols, cat_cols):
    # Scaling w/out centering to preserve sparsity when combined
    return ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
    ], remainder='drop', sparse_threshold=0.3)
```

---

## 🧩 src/models.py
```python
from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from .features import build_preprocessor
from .utils import ks_stat


def build_lgbm_pipeline(num_cols, cat_cols):
    pre = build_preprocessor(num_cols, cat_cols)
    clf = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([('pre', pre), ('clf', clf)])


def cv_scores(pipe, X, y, n_splits=5, calibrate=False):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, prs, briers, kss = [], [], [], []
    for tr, va in cv.split(X, y):
        if calibrate:
            model = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:,1]
        else:
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
        prs.append(average_precision_score(y[va], p))
        briers.append(brier_score_loss(y[va], p))
        kss.append(ks_stat(y[va], p))
    return float(np.mean(aucs)), float(np.mean(prs)), float(np.mean(briers)), float(np.mean(kss))
```

---

## 🧩 src/explain.py
```python
from __future__ import annotations
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def shap_global(pipe, X_sample, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Preprocess first; ensure dense for plotting convenience
    X_trans = pipe.named_steps['pre'].transform(X_sample)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    model = pipe.named_steps['clf']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    # Binary task: class 1 explanations at index 1 if list
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    plt.figure(figsize=(10,6))
    shap.summary_plot(sv, X_trans, show=False)
    plt.tight_layout(); plt.savefig(out / 'shap_summary.png', dpi=160); plt.close()


def shap_local(pipe, X_row, out_dir: str, idx: int = 0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X_trans = pipe.named_steps['pre'].transform(X_row)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    model = pipe.named_steps['clf']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    base = explainer.expected_value
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_val = base[1] if isinstance(base, (list, np.ndarray)) else base
    else:
        sv = shap_values[0]
        base_val = base
    exp = shap.Explanation(values=sv, base_values=base_val, data=X_trans[0])
    shap.plots.waterfall(exp, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout(); plt.savefig(out / f'shap_waterfall_{idx}.png', dpi=160); plt.close()
```

---

## 🧩 train.py
```python
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

from src.data import load_csv
from src.models import build_lgbm_pipeline, cv_scores
from src.explain import shap_global, shap_local
from src.utils import ensure_dir, save_json, ks_stat, pick_threshold_cost


def main(args):
    X, y, num_cols, cat_cols = load_csv(args.data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_lgbm_pipeline(num_cols, cat_cols)

    if args.calibrate:
        model = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
        model.fit(Xtr, ytr)
        p_tr = model.predict_proba(Xtr)[:,1]
        p_te = model.predict_proba(Xte)[:,1]
    else:
        pipe.fit(Xtr, ytr)
        model = pipe
        p_tr = pipe.predict_proba(Xtr)[:,1]
        p_te = pipe.predict_proba(Xte)[:,1]

    # CV metrics
    auc, pr, brier, ks = cv_scores(build_lgbm_pipeline(num_cols, cat_cols), Xtr, ytr, calibrate=args.calibrate)

    # Holdout metrics
    holdout = {
        'roc_auc': float(roc_auc_score(yte, p_te)),
        'pr_auc': float(average_precision_score(yte, p_te)),
        'brier': float(brier_score_loss(yte, p_te)),
        'ks': float(ks_stat(yte, p_te)),
    }

    # Thresholds
    t_cost = pick_threshold_cost(yte, p_te, fn_cost=args.fn_cost, fp_cost=args.fp_cost)

    # Save model & metadata
    ensure_dir(os.path.dirname(args.save_model))
    joblib.dump(model, args.save_model)
    save_json({
        'cv': {'roc_auc': auc, 'pr_auc': pr, 'brier': brier, 'ks': ks},
        'holdout': holdout,
        'threshold_cost': t_cost,
        'fn_cost': args.fn_cost,
        'fp_cost': args.fp_cost,
        'data': args.data
    }, os.path.join('models', 'metadata.json'))

    # Explainability figures
    sample = Xte.sample(min(2000, len(Xte)), random_state=42)
    if hasattr(model, 'base_estimator'):
        # CalibratedClassifierCV wrapper – inner pipeline is at .base_estimator
        inner = model.base_estimator
    else:
        inner = model
    shap_global(inner, sample, 'reports/figures')
    shap_local(inner, Xte.iloc[[0]], 'reports/figures', idx=0)

    print("\n=== CV Metrics ===")
    print({k: round(v, 4) for k, v in {'AUROC': auc, 'PR_AUC': pr, 'BRIER': brier, 'KS': ks}.items()})
    print("\n=== Holdout Metrics ===")
    print({k.upper(): round(v, 4) for k, v in holdout.items()})
    print(f"\nBest threshold by cost: {t_cost:.3f}  (FN cost={args.fn_cost}, FP cost={args.fp_cost})")
    print(f"Saved model → {args.save_model}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to processed CSV with target column `default`.')
    p.add_argument('--calibrate', action='store_true', help='Apply isotonic calibration.')
    p.add_argument('--fn_cost', type=float, default=5.0)
    p.add_argument('--fp_cost', type=float, default=1.0)
    p.add_argument('--save_model', default='models/credit_pipe.pkl')
    args = p.parse_args()
    main(args)
```

---

## 🧩 app/app.py (Streamlit)
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap

st.set_page_config(page_title="Credit Risk Scorer", layout="wide")
st.title("Credit Risk Scoring – Explainable Demo")
st.write("Upload a CSV of applicants (same columns as the training data, excluding the `default` target). You'll get PD, decision band, and a local SHAP explanation for a selected row.")

MODEL_PATH = Path('models/credit_pipe.pkl')
META_PATH = Path('models/metadata.json')

if not MODEL_PATH.exists():
    st.error("Model not found. Train it first: `python train.py --data data/processed/uci_credit_default.csv --calibrate --save-model models/credit_pipe.pkl`.")
    st.stop()

model = joblib.load(MODEL_PATH)
meta = {}
if META_PATH.exists():
    import json
    meta = json.load(open(META_PATH))
    st.sidebar.success(f"Cost-optimal threshold: {meta.get('threshold_cost', 0.5):.3f}")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write("Preview:")
    st.dataframe(data.head(20), use_container_width=True)

    # Predict probabilities
    proba = model.predict_proba(data)[:, 1]
    df_out = data.copy()
    df_out['PD'] = proba

    thr = float(meta.get('threshold_cost', 0.5))
    bins = [0, min(thr*0.6, 0.3), thr, 1.0]
    labels = ['Approve', 'Refer', 'Decline']
    df_out['Decision'] = pd.cut(df_out['PD'], bins=bins, labels=labels, include_lowest=True)

    st.subheader("Predictions")
    st.dataframe(df_out, use_container_width=True)

    st.subheader("Local Explanation")
    idx = st.number_input("Row index to explain", min_value=0, max_value=len(df_out)-1, value=0, step=1)

    # SHAP local on the selected row
    # Handle CalibratedClassifierCV wrappers
    inner = getattr(model, 'base_estimator', model)
    pre = inner.named_steps['pre']
    clf = inner.named_steps['clf']

    X_row = data.iloc[[idx]]
    X_trans = pre.transform(X_row)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    base = explainer.expected_value
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_val = base[1] if isinstance(base, (list, np.ndarray)) else base
    else:
        sv = shap_values[0]
        base_val = base
    exp = shap.Explanation(values=sv, base_values=base_val, data=X_trans[0])
    st.pyplot(shap.plots.waterfall(exp, show=False))
else:
    st.info("Upload a CSV to score applicants.")
```

---

## 🧪 Optional: Optuna Tuning Snippet (drop-in)
```python
# Add to a new file src/tune.py if desired
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from .features import build_preprocessor
import lightgbm as lgb


def objective(trial, X, y, num_cols, cat_cols):
    pre = build_preprocessor(num_cols, cat_cols)
    clf = lgb.LGBMClassifier(
        n_estimators=trial.suggest_int('n_estimators', 300, 1500),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int('num_leaves', 16, 128, log=True),
        min_child_samples=trial.suggest_int('min_child_samples', 10, 100),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    score = cross_val_score(pipe, X, y, scoring='roc_auc', cv=3, n_jobs=-1).mean()
    return score
```

---

## ✅ Next Steps (What to Showcase)
- Add a **model card** to README with metrics, calibration curves, top SHAP features, and decision policy.
- Include a **fairness snapshot**: calibration and error rates by age band or education (observational only; do not train on protected attributes).
- Optionally log runs with **MLflow** and render a small **dashboard** in Streamlit showing metric history and drift using `evidently`.

---

**You’re good to go.** Run the Quickstart, commit your results, and add the plots to your portfolio!

