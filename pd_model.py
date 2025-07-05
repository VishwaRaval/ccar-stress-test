"""
Fit a logistic‑regression PD model and persist to models/pd_logreg.pkl
"""
from __future__ import annotations
import argparse, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── helper to make modelling panel ──────────────────────────────────────────────

def make_panel(borrowers: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:  # ← 2‑arg signature
    latest = macro.xs("baseline").iloc[[-1]].reset_index(drop=True)
    panel  = borrowers.merge(latest, how="cross")

    # engineered features
    panel["ln_income"] = np.log(panel["annual_income"])
    panel["nasdaq_lr"] = np.log(panel["nasdaq"]).diff().fillna(0)

    # product dummies
    panel = pd.get_dummies(panel, columns=["product"], drop_first=True)
    panel["fico_bin"] = (panel["fico"] // 40).astype(int)
    panel = pd.get_dummies(panel, columns=["fico_bin"], drop_first=True)

    # fill card-only fields so all rows are finite
    panel["limit"] = panel["limit"].fillna(0.0)

    # default label – depends on borrower + macro
    rng = np.random.default_rng(42)
    base_pd = 0.01 + 0.25*(panel["fico"] < 620) + 0.15*(panel["dti"] > 0.4)
    macro_factor = (panel["unemployment_rate"] - 4.0) / 6.0
    final_pd = base_pd * (1 + macro_factor)
    panel["defaulted"] = rng.random(len(panel)) < final_pd
    return panel

# ── train‑and‑persist ──────────────────────────────────────────────────────────

def train(borrowers_path: pathlib.Path, macro_path: pathlib.Path):
    borrowers = pd.read_parquet(borrowers_path)
    macro     = pd.read_parquet(macro_path)

    df = make_panel(borrowers, macro)
    feats = [c for c in df.columns if c not in {"loan_id", "defaulted"}]
    X, y = df[feats], df["defaulted"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print(f"AUC test: {auc:.3f}")

    pathlib.Path("models").mkdir(exist_ok=True)
    pd.to_pickle(clf, "models/pd_logreg.pkl")
    print("✓ model saved → models/pd_logreg.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--borrowers", type=pathlib.Path, required=True)
    ap.add_argument("--macro",     type=pathlib.Path, required=True)
    args = ap.parse_args()
    train(args.borrowers, args.macro)