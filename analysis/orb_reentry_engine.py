# -*- coding: utf-8 -*-
"""orb_reentry_engine.py — shared SL + RE-ENTRY timeline for the NIFTY ORB + prev-day bias variants.
Spot-only: produces the per-day sequence of (entry -> exit) segments. Entry1 = first DESIRED-direction OR
break (reversal-entry allowed). SL = opposite OR extreme (= the undesired-direction level), spot TOUCH-based.
On SL hit, re-enter when the DESIRED level breaks AGAIN (touch), fresh ATM each time; repeat until 3:15 or the
last open position isn't stopped. Variant modules price their own structure (buy / credit spread) onto these
segments. ATM recomputed fresh at every entry (spot may have moved).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import run_backtest as rb

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OR_START, OR_END = 9 * 60 + 15, 9 * 60 + 29; SCAN_START, EXIT_MOD = 9 * 60 + 30, 15 * 60 + 15
FLOOR = pd.Timestamp("2024-10-01")


def opt_px(folder, strike, ot, day, ts):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return np.nan
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[o.ts.dt.normalize() == pd.Timestamp(day).normalize()].set_index("ts")["close"].sort_index()
    if not len(o): return np.nan
    v = o.asof(ts); return float(v) if pd.notna(v) else np.nan


def day_segments(ed, bias, D):
    """returns (segments, ORH, ORL) or (None, ORH, ORL). Each segment: entry_no, entry_ts, atm,
    exit_ts, exit_reason, entry_kind."""
    orb = ed[(ed["mod"] >= OR_START) & (ed["mod"] <= OR_END)]
    if len(orb) < 10: return None, None, None
    ORH = float(orb["high"].max()); ORL = float(orb["low"].min())
    scan = ed[(ed["mod"] >= SCAN_START) & (ed["mod"] <= EXIT_MOD)].sort_values("mod").reset_index(drop=True)
    if len(scan) < 2: return None, ORH, ORL
    long_dir = (bias == "bullish")
    desired = (scan["high"].values >= ORH) if long_dir else (scan["low"].values <= ORL)   # desired-direction level touched
    sl_lvl = (scan["low"].values <= ORL) if long_dir else (scan["high"].values >= ORH)     # opposite OR extreme touched
    n = len(scan); exit_pos = n - 1                                                         # last scan row ~ 15:15
    d0 = int(np.argmax(desired)) if desired.any() else None
    if d0 is None: return None, ORH, ORL
    undesired_first = bool(sl_lvl[:d0].any())
    segs = []; cur = d0; entry_no = 0
    while cur is not None and cur < exit_pos:
        entry_no += 1; er = scan.iloc[cur]; atm = int(round(float(er["close"]) / 50) * 50)
        sl_rel = np.where(sl_lvl[cur + 1:])[0]                                              # SL touch strictly after entry
        if len(sl_rel):
            sidx = cur + 1 + int(sl_rel[0]); xr = scan.iloc[sidx if False else sidx]; reason = "SL-hit"
            red = np.where(desired[sidx + 1:])[0]; nxt = (sidx + 1 + int(red[0])) if len(red) else None
        else:
            xr = scan.iloc[exit_pos]; reason = "3:15pm"; nxt = None
        kind = ("reversal (undesired-first)" if undesired_first else "first-breakout (desired)") if entry_no == 1 else "re-entry"
        segs.append({"entry_no": entry_no, "entry_ts": pd.Timestamp(er["ts"]), "atm": atm,
                     "exit_ts": pd.Timestamp(xr["ts"]) if reason == "SL-hit" else pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15),
                     "exit_reason": reason, "entry_kind": kind})
        cur = nxt
    return segs, ORH, ORL
