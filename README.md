# Simulated CCAR Stress Test – Recession Shock on Retail Loan Defaults

> **Goal**
> Demonstrate the end‑to‑end workflow of a CCAR‑style stress‑testing model: building a probability‑of‑default (PD) model, applying a “Severely Adverse” macroeconomic scenario, and quantifying the incremental capital required.

---

## 1 . Business Context

Under the Federal Reserve’s **Comprehensive Capital Analysis and Review (CCAR)**, large U.S. banks must prove they hold enough capital to survive severe recessions.
This repo focuses on **retail loan default risk**. We:

1. Assemble a retail‑loan portfolio (synthetic or real),
2. Predict baseline PD using borrower, loan, and macro features,
3. Shock macro variables (unemployment ↑, GDP growth ↓, inflation ↑) to mimic a recession,
4. Re‑estimate PD and compute **expected loss (EL)** under stress,
5. Report the **incremental capital** the bank would need.

---

## 2 . Data Sources

| Data                  | Example Source                                                                                            | Notes                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Retail‑loan records   | **LendingClub Loan Stats** (Kaggle) • HMDA public data • *or* synthetic generator in `src/stress_test.py` | Each row = 1 loan • key fields: amount, FICO, income, term, default flag    |
| Macroeconomic factors | **FRED (St. Louis Fed)**                                                                                  | Unemployment rate (`UNRATE`), GDP YoY (`A191RL1Q225SBEA`), CPI (`CPIAUCSL`) |
| Capital parameters    | Industry heuristics (EAD %, LGD %)                                                                        | Tunable to your assumptions                                                 |

> **Note:** The starter script ships with a **synthetic portfolio builder** so you can run end‑to‑end with zero external files. Swap in LendingClub or HMDA if you want real data.

---

## 3 . Repo Layout

```text
.
├── data/                # Optional raw files + FRED cache
├── notebooks/           # Jupyter EDA / visuals
├── src/
│   ├── stress_test.py   # Main script (run this!)
│   ├── …                # future: data_prep.py, modeling.py
├── requirements.txt
└── README.md
```

Run:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/stress_test.py
```

---

## 4 . Key Formulas

| Metric                  | Formula                         |
| ----------------------- | ------------------------------- |
| **Expected Loss (EL)**  | `EL = PD × EAD × LGD`           |
| **Incremental Capital** | `ΔEL = EL_stress − EL_baseline` |

---

## 5 . Next Steps / Extensions

* Replace synthetic loans with **LendingClub** or **HMDA** datasets.
* Upgrade PD model to **Gradient Boosting** / **XGBoost**.
* Split portfolio by **product type** (mortgage, auto, credit‑card) and apply product‑specific LGD.
* Add **capital buffers** (Basel III) and compute post‑stress **CET1 ratio**.
* Build a **dashboard** in Streamlit to interactively change macro shocks.

---

## 6 . References

* Federal Reserve – **Dodd‑Frank Act Stress Test Scenarios**
* Basel Committee – **Basel III: A Global Regulatory Framework**
* St. Louis Fed (**FRED**) economic time series

---

