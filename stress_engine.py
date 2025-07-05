"""
Compute quarterly expected losses and CET1 ratio under a chosen scenario.
The feature‑engineering here **exactly mirrors** what was used to train
`pd_logreg.pkl`, and we guarantee the final design‑matrix columns
(`feature_names_in_`) match the model.
"""
from __future__ import annotations
import pathlib, numpy as np, pandas as pd
from sklearn.base import BaseEstimator

RWA_WEIGHT_RETAIL = 0.75
START_CET1_RATIO  = 0.12     # 12 %
HORIZON_Q         = 9

class StressEngine:
    def __init__(self, borrowers: pd.DataFrame, macro: pd.DataFrame, pd_model: BaseEstimator):
        self.borrowers = borrowers.copy()
        self.macro     = macro
        self.pd_model  = pd_model
        self.expected_feats = list(pd_model.feature_names_in_)

    # ── run one scenario ───────────────────────────────────────────────────────
    def run(self, scenario: str = "severely_adverse") -> pd.DataFrame:
        # 1️⃣  macro slice (9‑quarter horizon)
        macro_slice = (self.macro.xs(scenario)
                                   .iloc[-HORIZON_Q:]
                                   .reset_index())     # keep "date" column
        macro_slice["nasdaq_lr"] = np.log(macro_slice["nasdaq"]).diff().fillna(0)

        # 2️⃣  cross‑join with borrower book
        panel = (self.borrowers.assign(tmp=1)
                               .merge(macro_slice.assign(tmp=1), on="tmp")
                               .drop(columns="tmp"))

        # 3️⃣  replicate PD‑model feature engineering --------------------------
        panel["ln_income"] = np.log(panel["annual_income"])
        panel["limit"]     = panel["limit"].fillna(0.0)

        # product dummies
        panel = pd.get_dummies(panel, columns=["product"], drop_first=True)
        # FICO buckets
        panel["fico_bin"] = (panel["fico"] // 40).astype(int)
        panel = pd.get_dummies(panel, columns=["fico_bin"], drop_first=True)

        # 4️⃣  align columns with the model spec --------------------------------
        for col in self.expected_feats:
            if col not in panel:
                panel[col] = 0.0
        X = panel[self.expected_feats]

        # 5️⃣  PD & EL ----------------------------------------------------------
        panel["pd"] = self.pd_model.predict_proba(X)[:, 1]
        panel["ead"] = np.where(panel["product_card"]==1,  # card dummy exists after get_dummies
                                 np.minimum(panel["limit"], panel["balance"]*1.1),
                                 panel["balance"])
        panel["lgd"] = panel["lgd_prior"]
        panel["el"]  = panel.eval("pd * ead * lgd")

        # 6️⃣  aggregate to portfolio -----------------------------------------
        losses = panel.groupby("date")["el"].sum() / 1e6
        rwa     = panel.groupby("date")["ead"].sum()*RWA_WEIGHT_RETAIL / 1e6

        capital = [START_CET1_RATIO * rwa.iloc[0]]
        for q in range(HORIZON_Q):
            capital.append(capital[-1] - losses.iloc[q])   # no revenue assumption
        capital = pd.Series(capital[1:], index=losses.index, name="capital_mn")
        cet1    = (capital / rwa).rename("cet1_ratio")

        return pd.concat([losses.rename("expected_loss_mn"),
                          rwa.rename("rwa_mn"),
                          capital,
                          cet1], axis=1)

# ── demo run --------------------------------------------------------------------
if __name__ == "__main__":
    borrowers = pd.read_parquet("data/borrowers.parquet")
    macro     = pd.read_parquet("data/macro.parquet")
    model     = pd.read_pickle("models/pd_logreg.pkl")

    eng = StressEngine(borrowers, macro, model)
    print("Severely‑Adverse:", eng.run("severely_adverse").head(), "")
    print("Baseline:", eng.run("baseline").head())