"""
Generate a synthetic retail-loan book and save to Parquet.

Run:
    python borrower_generator.py --n_loans 100000 --out data/borrowers.parquet
"""
from __future__ import annotations
import argparse, pathlib, numpy as np, pandas as pd

# ---- parameter priors -------------------------------------------------------
P_PRODUCT = {"mortgage": 0.55, "auto": 0.25, "card": 0.20}
AGE_TRIANG = (21, 75, 40)                      # min, max, mode
INCOME_LOGN = (10.5, 0.6)                      # μ, σ of log-income
FICO_BETA = (2.5, 7.0)                         # α, β → mean ≈690
BALANCE_BY_PRODUCT = {
    "mortgage": (80_000, 550_000),
    "auto":     (6_000,  50_000),
    "card":     (500,    15_000),
}
LIMIT_CARD_FACTOR = (1.2, 2.5)                 # balance × U()
DTI_PERT = (0.05, 0.45, 0.18)                  # min, max, mode
LGD_PRIORS = {"mortgage": 0.35, "auto": 0.65, "card": 0.90}

# ---- helper dists -----------------------------------------------------------
def triangular(rng, size, lo, hi, mode):        return rng.triangular(lo, mode, hi, size)
def beta_pert(rng, size, lo, hi, mode, lamb=4):
    α = 1 + lamb*(mode-lo)/(hi-lo);  β = 1 + lamb*(hi-mode)/(hi-lo)
    return lo + (hi-lo)*rng.beta(α, β, size)

# ---- main generator ---------------------------------------------------------
def generate_borrowers(n_loans: int, rng=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng()
    prod = rng.choice(list(P_PRODUCT), n_loans, p=list(P_PRODUCT.values()))
    age  = triangular(rng, n_loans, *AGE_TRIANG).round().astype(int)
    income = np.exp(rng.normal(*INCOME_LOGN, n_loans)).clip(None, 350_000).round(2)
    fico = (500 + 350*rng.beta(*FICO_BETA, n_loans)).round().astype(int)
    dti  = beta_pert(rng, n_loans, *DTI_PERT).round(3)

    bal = np.empty(n_loans)
    for p, (lo, hi) in BALANCE_BY_PRODUCT.items():
        idx = prod == p
        bal[idx] = rng.uniform(lo, hi, idx.sum())
    bal = bal.round(2)

    limit = np.where(prod=="card",
                     np.minimum(bal* rng.uniform(*LIMIT_CARD_FACTOR, n_loans), 25_000),
                     np.nan).round(2)

    return pd.DataFrame(dict(
        loan_id=np.arange(n_loans),
        product=prod, age=age, annual_income=income, fico=fico, dti=dti,
        balance=bal, limit=limit, lgd_prior=[LGD_PRIORS[p] for p in prod]
    ))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_loans", type=int, default=100_000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/borrowers.parquet"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate_borrowers(args.n_loans).to_parquet(args.out, index=False)
    print(f"✓ saved {args.n_loans:,} loans → {args.out}")
