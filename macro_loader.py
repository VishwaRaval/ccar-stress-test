"""
Fetch quarterly macro series from FRED, create baseline & Fed‑style severely‑adverse
9‑quarter projections, save as Parquet.
"""
from __future__ import annotations
import argparse, datetime as dt, os, pathlib, numpy as np, pandas as pd
from fredapi import Fred

SERIES = {
    "UNRATE"   : "unemployment_rate",
    "GDPC1"    : "gdp_real",
    "CPIAUCSL" : "cpi",
    "NASDAQCOM": "nasdaq",        # daily, history back to 1971
}
START       = dt.date(1990, 1, 1)
HORIZON_Q   = 9

# Fed 2025 Severely‑Adverse (stylised)
U_PATH   = [4.5, 5.3, 6.5, 7.6, 8.4, 9.1, 9.7, 10.0, 10.0]
GDP_PCT  = [-1.5, -3.4, -2.8, -1.2, -0.5, 0.3, 1.0, 1.7, 2.2]   # SAAR %
EQ_PCT   = [0, -10, -18, -27, -33, -36, -30, -25, -20]          # % q/q cum

# ── helpers ─────────────────────────────────────────────────────────────────────

def pct_to_level(hist_series: pd.Series, pct_path: list[float]) -> list[float]:
    out, last = [], hist_series.iloc[-1]
    for pct in pct_path:
        last *= (1 + pct / 100)
        out.append(round(last, 2))
    return out


def fetch_history() -> pd.DataFrame:
    fred = Fred(api_key="d081ad8d50315ef914946fe22296ed73")
    frames = []
    for code, name in SERIES.items():
        s = fred.get_series(code, observation_start=START).to_frame(name=name)
        s.index = pd.to_datetime(s.index)
        s_q = s.resample( "QE").last().ffill()     # one obs / quarter
        frames.append(s_q)
    return pd.concat(frames, axis=1).sort_index().ffill().bfill()


def build_scenarios(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    last_q = hist.index[-1]
    proj_ix = pd.date_range(last_q + pd.offsets.QuarterEnd(), periods=HORIZON_Q, freq= "QE")

    # baseline: flat‑forward
    flat_future = pd.DataFrame(np.repeat(hist.tail(1).values, HORIZON_Q, axis=0),
                               index=proj_ix, columns=hist.columns)
    baseline = pd.concat([hist, flat_future])

    # severely‑adverse
    severe = baseline.copy()
    severe.loc[proj_ix, "unemployment_rate"] = U_PATH
    severe.loc[proj_ix, "gdp_real"]          = pct_to_level(hist["gdp_real"], GDP_PCT)
    severe.loc[proj_ix, "nasdaq"]            = pct_to_level(hist["nasdaq"], EQ_PCT)

    return {"baseline": baseline, "severely_adverse": severe}

# ── main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/macro.parquet"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hist = fetch_history()
    tidy = pd.concat(build_scenarios(hist), names=["scenario", "date"])
    tidy.to_parquet(args.out)
    print(f"✓ Saved macro history + projections → {args.out}")