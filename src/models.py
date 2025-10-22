# src/models.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# --------- metrics & thresholding ---------
def pr_auc(y_true, y_prob): return average_precision_score(y_true, y_prob)
def roc_auc(y_true, y_prob): return roc_auc_score(y_true, y_prob)

def confusion_at(y_true, y_prob, thr):
    y_pred = (y_prob >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": float(sens), "specificity": float(spec)}

def pick_threshold(y_true, y_prob, mode="f1", target=None):
    """
    Choose probability threshold using VAL:
      mode ∈ {"f1","youden","target_recall","target_precision"}.
    """
    thr_candidates = np.unique(np.r_[0.0, np.sort(y_prob), 1.0])
    best = None
    for thr in thr_candidates:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        spec = tn/(tn+fp) if (tn+fp) else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        youden = rec + spec - 1.0

        if mode == "f1": score = f1
        elif mode == "youden": score = youden
        elif mode == "target_recall":
            if target is None or not (0 < target < 1): raise ValueError("target_recall needs target in (0,1)")
            if rec < target: continue
            score = -thr
        elif mode == "target_precision":
            if target is None or not (0 < target < 1): raise ValueError("target_precision needs target in (0,1)")
            if prec < target: continue
            score = -thr
        else:
            raise ValueError("Unknown mode")

        if best is None or score > best["score"]:
            best = {"thr": float(thr), "score": float(score), "precision": float(prec),
                    "recall": float(rec), "specificity": float(spec), "f1": float(f1)}
    if best is None:
        return 0.5, {"precision":0.0, "recall":0.0, "specificity":0.0, "f1":0.0}
    return best["thr"], best

# --------- XGBoost: tuner / refit / val-probs ---------
def tune_xgb(X_train, y_train, X_val, y_val, CFG, seed):
    grid = {
        "max_depth": [2,3,4,5,6],
        "learning_rate": [0.01,0.05,0.1,0.2],
        "subsample": [0.5,0.7,0.9,1.0],
        "colsample_bytree": [0.5,0.7,0.9,1.0],
        "reg_lambda": [0.0,0.5,1.0,5.0,10.0],
        "reg_alpha":  [0.0,0.5,1.0,5.0,10.0],
    }
    sampler = list(ParameterSampler(grid, n_iter=CFG.XGB_N_SAMPLES, random_state=seed))
    dtr, dv = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val)
    best = {"mode":"tuned", "val_pr_auc": -1.0, "params":None, "best_iteration":None}
    for p in sampler:
        params = {
            "max_depth": p["max_depth"],
            "eta": p["learning_rate"],
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
            "lambda": p["reg_lambda"],
            "alpha": p["reg_alpha"],
            "objective":"binary:logistic", "eval_metric":"aucpr",
            "tree_method":"hist", "seed":seed,
        }
        booster = xgb.train(
            params, dtr, num_boost_round=CFG.XGB_MAX_BOOST_ROUNDS,
            evals=[(dv, "val")], early_stopping_rounds=CFG.XGB_EARLY_STOP_ROUNDS,
            verbose_eval=False,
        )
        val_pr = float(booster.best_score)
        if val_pr > best["val_pr_auc"]:
            best["val_pr_auc"] = val_pr
            best["params"] = params
            best["best_iteration"] = int(booster.best_iteration)
    return best

def retrain_and_evaluate_xgb(best, X_trainval, y_trainval, X_test, y_test):
    rounds = int(best["best_iteration"]) + 1
    booster = xgb.train(best["params"], xgb.DMatrix(X_trainval, label=y_trainval),
                        num_boost_round=rounds, verbose_eval=False)
    y_prob = booster.predict(xgb.DMatrix(X_test))
    return {"mode":"tuned", "n_rounds": rounds, "test_pr_auc": pr_auc(y_test, y_prob),
            "test_roc_auc": roc_auc(y_test, y_prob), "y_prob_test": y_prob, "booster": booster}

def xgb_val_probs(best, Xtr, ytr, Xv):
    rounds = int(best["best_iteration"]) + 1
    bst = xgb.train(best["params"], xgb.DMatrix(Xtr, label=ytr),
                    num_boost_round=rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(Xv))

# --------- Logistic Regression: tuner / refit ---------
def tune_logreg_lr(X_train, y_train, X_val, y_val, CFG, seed, allow_class_weight=True):
    grid = {"penalty": ["l1","l2"], "C": np.logspace(-3,3,30)}
    cw = [None, "balanced"] if allow_class_weight else [None]
    sampler = list(ParameterSampler({"penalty": grid["penalty"], "C": grid["C"], "class_weight": cw},
                                    n_iter=CFG.LR_N_SAMPLES, random_state=seed))
    best = {"val_pr_auc": -1.0, "params": None}
    for p in sampler:
        lr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(
                penalty=p["penalty"], C=p["C"], class_weight=p["class_weight"],
                solver="liblinear", max_iter=2000, n_jobs=-1, random_state=seed
            ))
        ])
        lr.fit(X_train, y_train)
        val_pr = pr_auc(y_val, lr.predict_proba(X_val)[:,1])
        if val_pr > best["val_pr_auc"]:
            best["val_pr_auc"] = float(val_pr)
            best["params"] = {"penalty": p["penalty"], "C": p["C"],
                              "class_weight": p["class_weight"], "solver": "liblinear"}
    return best

def retrain_and_evaluate_lr(best, X_trainval, y_trainval, X_test, y_test, seed):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(
            penalty=best["params"]["penalty"], C=best["params"]["C"],
            class_weight=best["params"]["class_weight"], solver="liblinear",
            max_iter=2000, n_jobs=-1, random_state=seed
        ))
    ])
    lr.fit(X_trainval, y_trainval)
    y_prob = lr.predict_proba(X_test)[:,1]
    return {"test_pr_auc": pr_auc(y_test, y_prob), "test_roc_auc": roc_auc(y_test, y_prob),
            "y_prob_test": y_prob, "pipeline": lr}
