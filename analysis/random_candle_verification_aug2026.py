# -*- coding: utf-8 -*-
"""random_candle_verification_aug2026.py — builds a random-sample verification Excel for the newly-pulled
+ densified NIFTY options data (4 new expiries). Includes: 20 fully random candles across the new expiries,
1 REAL candle from the CAS-extension window (15:30-15:39), and 1 SYNTHETIC (densify-filled) candle from
that same window, so both real and filled behavior in the new post-CAS tail window can be eyeballed.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "fullchain_audit"; OUTDIR.mkdir(parents=True, exist_ok=True)
NEW_EXPS = ["20260804", "20260811", "20260818", "20260825"]
SEED = 42


def main():
    files = []
    for e in NEW_EXPS:
        files += sorted(glob.glob(str(OPTDIR / e / "*.parquet")))
    print(f"pool: {len(files)} contracts across {len(NEW_EXPS)} new expiries", flush=True)

    rng = np.random.default_rng(SEED)

    # ---- 1) 20 fully random candles, random contract + random row ----
    sample_files = rng.choice(files, size=20, replace=False)
    general_rows = []
    for f in sample_files:
        df = pd.read_parquet(f)
        r = df.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
        general_rows.append(r)
    GENERAL = pd.DataFrame(general_rows)
    GENERAL.insert(0, "sample_type", "random_general")

    # ---- 2) 1 REAL candle from the 15:30-15:39 CAS-extension window ----
    real_tail = None
    for f in rng.permutation(files):
        df = pd.read_parquet(f)
        m = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        cand = df[(m >= 930) & (m <= 939) & (~df["is_synthetic"])]
        if len(cand):
            real_tail = cand.sample(1, random_state=SEED).iloc[0]
            real_tail_file = Path(f).name
            break
    print(f"real 15:30-15:39 candle from: {real_tail_file}", flush=True)

    # ---- 3) 1 SYNTHETIC candle from the 15:30-15:39 CAS-extension window ----
    synth_tail = None
    for f in rng.permutation(files):
        df = pd.read_parquet(f)
        m = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        cand = df[(m >= 930) & (m <= 939) & (df["is_synthetic"])]
        if len(cand):
            synth_tail = cand.sample(1, random_state=SEED).iloc[0]
            synth_tail_file = Path(f).name
            break
    print(f"synthetic 15:30-15:39 candle from: {synth_tail_file}", flush=True)

    SPECIAL = pd.DataFrame([real_tail, synth_tail])
    SPECIAL.insert(0, "sample_type", ["real_15_30_to_15_39", "synthetic_15_30_to_15_39"])
    SPECIAL.insert(1, "source_file", [real_tail_file, synth_tail_file])

    ALL = pd.concat([GENERAL, SPECIAL.drop(columns=["source_file"])], ignore_index=True)

    with pd.ExcelWriter(OUTDIR / "random_candle_verification_aug2026.xlsx", engine="openpyxl") as w:
        ALL.to_excel(w, sheet_name="Random_Candles", index=False)
        SPECIAL.to_excel(w, sheet_name="CAS_Window_Samples", index=False)

    pd.set_option("display.width", 220)
    print("\n=== ALL SAMPLED CANDLES ===")
    print(ALL.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/random_candle_verification_aug2026.xlsx")


if __name__ == "__main__":
    main()
