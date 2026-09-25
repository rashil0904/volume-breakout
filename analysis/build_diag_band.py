# -*- coding: utf-8 -*-
"""
build_diag_band.py
==================
Generic driver: build a diagnostic table for ANY market-cap band, using the identical
entry pipeline as the locked baseline (lookback 36, vol 6x, +5%, 15:15 entry). Only the
band changes. Writes results/diagnostic_table_<suffix>.csv — never touches existing tables.

Usage:
  python analysis/build_diag_band.py <MCAP_MIN_CR> <MCAP_MAX_CR> <suffix>
  e.g.  python analysis/build_diag_band.py 5000 7500 mcap5k7p5k

Band convention matches existing code: mcap_cr >= MIN & mcap_cr <= MAX (inclusive both).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_data as pd_mod


def build(mcap_min, mcap_max, suffix):
    pd_mod.MCAP_MIN_CR = mcap_min
    pd_mod.MCAP_MAX_CR = mcap_max
    out = pd_mod.RESULTS / f"diagnostic_table_{suffix}.csv"
    print(f"Building ₹{mcap_min:,}–{mcap_max:,} Cr diagnostic table -> {out.name}")
    pd_mod.build_diagnostic_table(vol_window=36, vol_mult=6.0,
                                  output_path=str(out), save_csv=True, verbose=True)
    print(f"\nDone -> {out}")
    return out


if __name__ == "__main__":
    mn, mx, sfx = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    build(mn, mx, sfx)
