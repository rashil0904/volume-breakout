# -*- coding: utf-8 -*-
"""supertrend_retrace_dual_tf.py — STANDALONE NEW variant of the NIFTY dual-TF Supertrend(10,3) (15min +
1hr), with a 50%-RETRACEMENT entry instead of close entry. Does NOT touch the original close-based system.

Per leg: Supertrend flip fires on the confirming candle's close. midpoint = (conf_high+conf_low)/2. Then wait
(FLAT) for a subsequent 1-min TOUCH of that midpoint -> enter at midpoint (long if bull flip / retrace down;
short if bear / retrace up). If the next OPPOSITE flip fires first -> abandon (no entry), switch to the new
flip's midpoint. Already positioned: same-dir re-flip = no-op; opposite flip = EXIT at that flip candle's
CLOSE (flip mechanics as before), then wait for the new midpoint. Not always-in: legs can be FLAT.
Net = legA+legB in {-2..+2} incl 0. 1 lot/leg. GROSS index points. Own output folder.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend, ATR_N, FACT

DATA1 = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "supertrend_retrace"; OUTDIR.mkdir(parents=True, exist_ok=True)


def leg_walk(bof1, bdir, bflip, bmid, bclose, hi, lo, tsv):
    """retracement-entry state machine for one leg. bof1=bucket id per 1-min row. returns trades, pos_series, counts."""
    n = len(bof1); pos = 0; entry = np.nan; entry_t = None; edir = 0; pending = None
    trades = []; posser = np.zeros(n, dtype=np.int8)
    n_flips = 0; n_entry = 0; n_abandon = 0; prev_b = -1
    for k in range(n):
        b = bof1[k]
        if b != prev_b and prev_b >= 0 and bflip[prev_b]:            # a leg-candle just closed AND it flipped
            n_flips += 1; fdir = bdir[prev_b]; m = bmid[prev_b]; fc = bclose[prev_b]
            if pos == 0:
                if pending is not None: n_abandon += 1                # old untouched setup discarded
                pending = (fdir, m)
            elif pos == fdir:
                pass                                                  # same-direction re-flip: no action
            else:                                                     # opposite flip while positioned -> EXIT at flip close
                pnl = (fc - entry) if pos == 1 else (entry - fc)
                trades.append((("Long" if pos == 1 else "Short"), entry_t, round(entry, 2), tsv[prev_b_last], round(fc, 2), round(pnl, 2)))
                pos = 0; pending = (fdir, m)
        if b != prev_b:
            prev_b = b
        prev_b_last = k                                               # 1-min index tracking the current bar (for exit timestamp)
        if pos == 0 and pending is not None:                          # ---- watch retracement touch ----
            pdir, m = pending
            if pdir == 1 and lo[k] <= m:
                pos = 1; entry = m; entry_t = tsv[k]; edir = 1; pending = None; n_entry += 1
            elif pdir == -1 and hi[k] >= m:
                pos = -1; entry = m; entry_t = tsv[k]; edir = -1; pending = None; n_entry += 1
        posser[k] = pos
    if pos != 0:                                                      # period end
        fc = bclose[bof1[n - 1]]; pnl = (fc - entry) if pos == 1 else (entry - fc)
        trades.append((("Long" if pos == 1 else "Short"), entry_t, round(entry, 2), tsv[n - 1], round(fc, 2), round(pnl, 2)))
    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "pnl"])
    return T, posser, {"flips": n_flips, "entries": n_entry, "abandoned": n_abandon}


def close_based(dir_arr, close, start):
    """original always-in close-to-close flip P&L on the same candles (for comparison)."""
    flips = [i for i in range(start + 1, len(dir_arr)) if dir_arr[i] != dir_arr[i - 1]]
    pnl = []
    for k, i in enumerate(flips):
        j = flips[k + 1] if k + 1 < len(flips) else len(close) - 1
        pnl.append(dir_arr[i] * (close[j] - close[i]))
    return round(float(np.sum(pnl)), 1), len(flips)


def main():
    d = pd.read_csv(DATA1); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d = d[(d["mod"] >= 555) & (d["mod"] <= 929)].reset_index(drop=True)
    d["b15"] = pd.factorize(d["date"].dt.strftime("%Y%m%d") + "_" + ((d["mod"] - 555) // 15).astype(int).astype(str))[0]
    d["b1h"] = pd.factorize(d["date"].dt.strftime("%Y%m%d") + "_" + ((d["mod"] - 555) // 60).astype(int).astype(str))[0]
    hi = d["high"].values; lo = d["low"].values; tsv = d["ts"].values
    period = f"{d.date.iloc[0].date()} .. {d.date.iloc[-1].date()}"

    def buckets(bcol):
        g = d.groupby(bcol).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last")).sort_index()
        dirn = supertrend(g["h"].values, g["l"].values, g["c"].values, ATR_N, FACT)
        flip = np.zeros(len(g), bool); flip[ATR_N + 1:] = dirn[ATR_N + 1:] != dirn[ATR_N:-1]
        mid = (g["h"].values + g["l"].values) / 2.0
        return d[bcol].values, dirn, flip, mid, g["c"].values

    bofA, dirA, flipA, midA, closeA = buckets("b15")
    bofB, dirB, flipB, midB, closeB = buckets("b1h")
    TA, posA, cA = leg_walk(bofA, dirA, flipA, midA, closeA, hi, lo, tsv)
    TB, posB, cB = leg_walk(bofB, dirB, flipB, midB, closeB, hi, lo, tsv)
    cbA, fA = close_based(dirA, closeA, ATR_N + 1); cbB, fB = close_based(dirB, closeB, ATR_N + 1)

    n = len(d); net = posA.astype(int) + posB.astype(int)

    def legrow(name, T, c, cb):
        return {"leg": name, "flip_signals": c["flips"], "entered": c["entries"], "abandoned": c["abandoned"],
                "entry_rate_%": round(c["entries"] / c["flips"] * 100, 1) if c["flips"] else 0,
                "trades": len(T), "win_%": round((T.pnl > 0).mean() * 100, 1) if len(T) else 0,
                "retrace_pnl": round(T.pnl.sum(), 1), "closebased_pnl(same candles)": cb,
                "retrace-close_delta": round(T.pnl.sum() - cb, 1)}
    LEGS = pd.DataFrame([legrow("15min", TA, cA, cbA), legrow("1hour", TB, cB, cbB)])
    comb_retrace = round(TA.pnl.sum() + TB.pnl.sum(), 1); comb_close = round(cbA + cbB, 1)

    # time-in-state per leg + net-state
    tstate = pd.DataFrame([
        {"leg": "15min", "%flat_waiting": round((posA == 0).mean() * 100, 1), "%long": round((posA == 1).mean() * 100, 1), "%short": round((posA == -1).mean() * 100, 1)},
        {"leg": "1hour", "%flat_waiting": round((posB == 0).mean() * 100, 1), "%long": round((posB == 1).mean() * 100, 1), "%short": round((posB == -1).mean() * 100, 1)}])
    nstates = pd.DataFrame([{"net_state": s, "pct_time": round((net == s).mean() * 100, 1), "minutes": int((net == s).sum())} for s in [2, 1, 0, -1, -2]])

    for T in (TA, TB):
        for col in ("entry_time", "exit_time"):
            T[col] = pd.to_datetime(T[col]).dt.strftime("%Y-%m-%d %H:%M")
        T.insert(0, "trade_no", range(1, len(T) + 1))

    with pd.ExcelWriter(OUTDIR / "supertrend_retrace.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "STANDALONE dual-TF Supertrend(10,3) with 50%-RETRACEMENT entry (NOT the original close-based system)"},
            {"metric": "Period", "value": period}, {"metric": "P&L", "value": "GROSS index points, 1 lot/leg; entry@midpoint (touch), exit@flip close"},
            {"metric": "Assumptions", "value": "1) touch-based retrace  2) opposite flip replaces pending  3) same-dir re-flip = no-op (all CONFIRMED)"},
            {"metric": "Not always-in", "value": "legs are FLAT while waiting for a retrace touch -> net can be 0"},
            {"metric": "", "value": ""},
            {"metric": "COMBINED retrace P&L", "value": comb_retrace},
            {"metric": "COMBINED close-based P&L (same candles)", "value": comb_close},
            {"metric": "COMBINED delta (retrace - close)", "value": round(comb_retrace - comb_close, 1)},
            {"metric": "15min entered/abandoned", "value": f"{cA['entries']}/{cA['abandoned']} of {cA['flips']} flips"},
            {"metric": "1hour entered/abandoned", "value": f"{cB['entries']}/{cB['abandoned']} of {cB['flips']} flips"},
            {"metric": "Verdict", "value": "retrace " + ("IMPROVES" if comb_retrace > comb_close else "HURTS") + f" gross vs close-based by {round(comb_retrace-comb_close,1)} pts"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        LEGS.to_excel(w, sheet_name="Leg_Performance", index=False)
        tstate.to_excel(w, sheet_name="Time_In_State", index=False)
        nstates.to_excel(w, sheet_name="Net_State", index=False)
        TA.to_excel(w, sheet_name="15min_Trades", index=False); TB.to_excel(w, sheet_name="1hour_Trades", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nDUAL-TF SUPERTREND — 50%-RETRACEMENT ENTRY (standalone) | " + period + "\n" + "=" * 96)
    print("\n--- PER LEG (retrace vs close-based on same candles) ---"); print(LEGS.to_string(index=False))
    print(f"\nCOMBINED: retrace {comb_retrace} vs close-based {comb_close} -> delta {round(comb_retrace-comb_close,1)} pts")
    print("\n--- TIME IN STATE (per leg, % of minutes) ---"); print(tstate.to_string(index=False))
    print("\n--- NET-STATE (-2..+2, incl 0) ---"); print(nstates.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
