# -*- coding: utf-8 -*-
"""
build_diag_mcap5k10k.py
=======================
Builds a SEPARATE diagnostic table for the ₹5,000–10,000 Cr market-cap band, using
the identical entry pipeline as the locked baseline (lookback 36, vol 6x, +5% move,
15:15 entry). ONLY the mcap band differs. Writes results/diagnostic_table_mcap5k10k.csv
— the existing ₹1,500–5,000 Cr diagnostic_table.csv is never touched.

Band convention matches the existing code: mcap_cr >= MIN & mcap_cr <= MAX (inclusive).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_data as pd_mod

# ── only the band changes; everything else identical to the baseline build ──
pd_mod.MCAP_MIN_CR = 5_000
pd_mod.MCAP_MAX_CR = 10_000

OUT = pd_mod.RESULTS / "diagnostic_table_mcap5k10k.csv"

if __name__ == "__main__":
    print(f"Building ₹5,000–10,000 Cr diagnostic table -> {OUT.name}")
    pd_mod.build_diagnostic_table(vol_window=36, vol_mult=6.0,
                                  output_path=str(OUT), save_csv=True, verbose=True)
    print(f"\nDone -> {OUT}")
