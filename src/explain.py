# src/explain.py
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np

def _unwrap(model):
    # If it's a calibrator, unwrap either `.base_estimator` or `.estimator`
    if hasattr(model, "base_estimator") and not hasattr(model, "named_steps"):
        model = model.base_estimator
    if hasattr(model, "estimator") and not hasattr(model, "named_steps"):
        model = model.estimator
    if hasattr(model, "named_steps"):
        return model.named_steps["pre"], model.named_steps["clf"]
    raise TypeError("Expected a Pipeline with 'pre' and 'clf' steps")

def shap_global(model, X_sample, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pre, clf = _unwrap(model)
    X_trans = pre.transform(X_sample)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_trans)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(figsize=(10,6))
    shap.summary_plot(sv, X_trans, show=False)
    plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=160); plt.close()

def shap_local(model, X_row, out_dir: str, idx: int = 0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pre, clf = _unwrap(model)
    X_trans = pre.transform(X_row)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_trans)
    base = explainer.expected_value
    if isinstance(sv, list):
        values = sv[1][0]
        base_val = base[1] if isinstance(base, (list, np.ndarray)) else base
    else:
        values = sv[0]; base_val = base
    exp = shap.Explanation(values=values, base_values=base_val, data=X_trans[0])
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout(); plt.savefig(out / f"shap_waterfall_{idx}.png", dpi=160); plt.close()
