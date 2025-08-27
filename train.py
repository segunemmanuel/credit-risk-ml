# train.py
from src.data import load_csv
from src.models import build_lgbm_pipeline, cv_scores
from src.explain import shap_global, shap_local
from src.utils import ks_stat, save_json, ensure_dir

import argparse, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def main(args):
    X, y, num_cols, cat_cols = load_csv(args.data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_lgbm_pipeline(num_cols, cat_cols)

    if args.calibrate:
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        model.fit(Xtr, ytr)
        p_te = model.predict_proba(Xte)[:, 1]
    else:
        pipe.fit(Xtr, ytr)
        model = pipe
        p_te = pipe.predict_proba(Xte)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(yte, p_te),
        "pr_auc": average_precision_score(yte, p_te),
        "brier": brier_score_loss(yte, p_te),
        "ks": ks_stat(yte, p_te),
    }

        # Save model & metadata
    ensure_dir("models")
    joblib.dump(model, args.save_model)
    save_json(metrics, "models/metadata.json")

    # === Build an explainer-friendly Pipeline ===
    if args.calibrate:
        # Fit a fresh copy of the pipeline on the same training data purely for SHAP visuals
        inner = build_lgbm_pipeline(num_cols, cat_cols)
        inner.fit(Xtr, ytr)
    else:
        inner = model  # already a Pipeline

    # SHAP figures
    sample = Xte.sample(min(500, len(Xte)), random_state=42)
    shap_global(inner, sample, "reports/figures")
    shap_local(inner, Xte.iloc[[0]], "reports/figures", idx=0)

    print("Metrics:", metrics)
