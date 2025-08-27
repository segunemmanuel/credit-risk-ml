# src/models.py
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
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def cv_scores(pipe, X, y, n_splits=5, calibrate=False):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, prs, briers, kss = [], [], [], []
    for tr, va in cv.split(X, y):
        if calibrate:
            model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
        else:
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
        prs.append(average_precision_score(y[va], p))
        briers.append(brier_score_loss(y[va], p))
        kss.append(ks_stat(y[va], p))
    return float(np.mean(aucs)), float(np.mean(prs)), float(np.mean(briers)), float(np.mean(kss))
