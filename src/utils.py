# src/utils.py
import json, os
import numpy as np
from sklearn.metrics import roc_curve

def ensure_dir(path: str):
    if path in ("", None): return
    os.makedirs(path, exist_ok=True)

def save_json(d: dict, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def ks_stat(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))
