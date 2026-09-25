# -*- coding: utf-8 -*-
"""breakout_target_sweep.py — PROFIT-TARGET sweep on the DAILY PURE breakout (0-pt margin, WITH gap handling).
Adds a target-exit-to-FLAT state (distinct variant; base always-in logic untouched). Target 50..2000 step 50
(40 values) + no-target baseline. Touch-based target; on hit -> go FLAT; re-enter on next breakout (no gap
treatment from flat). Opposite breakout still FLIPS (the 'flipped' bucket). Reports per-target trades/win/P&L,
exit-type mix (target/flip/period-end), avg flat time, IS/OOS, best stable region. GROSS index points.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_target_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
FIRST15_END = 570; TARGETS = list(range(50, 2001, 50)); OOS_SPLIT = pd.Timestamp("2024-07-01")


def walk(A, target):
    """target=None -> always-in flip baseline. else target-exit-to-flat variant. returns trades list.
    Re-entry from flat requires a FRESH breakout: the level must be re-armed by price retreating inside it
    (low<=up re-arms long; high>=down re-arms short) before it can trigger again — prevents re-firing while
    price sits above/below a level already broken."""
    up_a, dn_a, p2h_a, p2l_a, do_a, fh_a, fl_a, dt_a, mod_a, hi_a, lo_a, op_a, tsr, cl_a = A
    pos = 0; entry = np.nan; entry_t = None; trades = []; armed_long = True; armed_short = True
    cur_day = None; gap = None; gap_flipped = False; last_exit_t = None; flats = []
    n = len(dt_a)
    for k in range(n):
        if dt_a[k] != cur_day:
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if pos != 0 and not np.isnan(p2l_a[k]):
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        if pos == 0:                                             # ---- FLAT: re-enter on a FRESH breakout (no gap logic) ----
            if l <= up: armed_long = True                        # price back inside -> level re-armed
            if h >= dn: armed_short = True
            hu, hd = (h >= up and armed_long), (l <= dn and armed_short)
            if hu or hd:
                pos, entry = (1, up) if (hu and (not hd or abs(o - up) <= abs(o - dn))) else (-1, dn); entry_t = t
                if pos == 1: armed_long = False
                else: armed_short = False
                if last_exit_t is not None: flats.append((t - last_exit_t) / np.timedelta64(1, "h")); last_exit_t = None
            continue
        tgt = (entry + target) if (target is not None and pos == 1) else ((entry - target) if target is not None else None)
        if gap in ("down", "up") and not gap_flipped:           # ---- GAP MODE (in position, adverse overnight gap) ----
            if m < FIRST15_END:
                continue
            if target is not None and ((pos == 1 and h >= tgt) or (pos == -1 and l <= tgt)):
                ex = tgt; trades.append((("Long" if pos == 1 else "Short"), entry_t, entry, t, ex, (ex - entry) if pos == 1 else (entry - ex), "target"))
                pos = 0; last_exit_t = t; gap = None; gap_flipped = True; continue
            if pos == 1 and l <= fl_a[k]:
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry, "flip")); pos, entry, entry_t = -1, fl_a[k], t; gap = None; gap_flipped = True; armed_short = False
            elif pos == -1 and h >= fh_a[k]:
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k], "flip")); pos, entry, entry_t = 1, fh_a[k], t; gap = None; gap_flipped = True; armed_long = False
            continue
        # ---- NORMAL (in position): target (favorable, -> flat) then opposite breakout (-> flip) ----
        if target is not None and ((pos == 1 and h >= tgt) or (pos == -1 and l <= tgt)):
            ex = tgt; trades.append((("Long" if pos == 1 else "Short"), entry_t, entry, t, ex, (ex - entry) if pos == 1 else (entry - ex), "target"))
            pos = 0; last_exit_t = t
        elif pos == 1 and l <= dn:
            trades.append(("Long", entry_t, entry, t, dn, dn - entry, "flip")); pos, entry, entry_t = -1, dn, t; armed_short = False
        elif pos == -1 and h >= up:
            trades.append(("Short", entry_t, entry, t, up, entry - up, "flip")); pos, entry, entry_t = 1, up, t; armed_long = False
    if pos != 0:                                                 # ---- period end ----
        ex = cl_a[n - 1]; trades.append((("Long" if pos == 1 else "Short"), entry_t, entry, tsr[n - 1], ex, (ex - entry) if pos == 1 else (entry - ex), "period_end"))
    return trades, flats


def summarize(trades, flats, target):
    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry", "exit_time", "exit", "pnl", "exit_reason"])
    ent = pd.to_datetime(T["entry_time"]); ism = ent < OOS_SPLIT
    n = len(T); c = T["exit_reason"].value_counts()
    return {"target": ("none" if target is None else target), "trades": n,
            "win_%": round((T.pnl > 0).mean() * 100, 1), "total_pts": round(T.pnl.sum(), 1),
            "pct_target": round(c.get("target", 0) / n * 100, 1), "pct_flip": round(c.get("flip", 0) / n * 100, 1),
            "pct_end": round(c.get("period_end", 0) / n * 100, 1),
            "avg_flat_hrs": round(float(np.mean(flats)), 1) if flats else 0.0,
            "IS_pts": round(T.pnl[ism].sum(), 1), "OOS_pts": round(T.pnl[~ism].sum(), 1)}


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    dopen = d.groupby("date")["open"].first().rename("dopen")
    daily = daily.join(f15).join(dopen)
    dm = d.merge(daily[["p2h", "p2l", "fh15", "fl15", "dopen"]], left_on="date", right_index=True, how="left")
    A = (dm["p2h"].values, dm["p2l"].values, dm["p2h"].values, dm["p2l"].values, dm["dopen"].values, dm["fh15"].values, dm["fl15"].values,
         dm["date"].values, dm["mod"].values, dm["high"].values, dm["low"].values, dm["open"].values, dm["ts"].values, dm["close"].values)
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"

    rows = []
    tr0, fl0 = walk(A, None); base = summarize(tr0, fl0, None); rows.append(base)
    for tg in TARGETS:
        tr, fl = walk(A, tg); rows.append(summarize(tr, fl, tg))
        print(f"  target {tg}: trades {rows[-1]['trades']} | P&L {rows[-1]['total_pts']}")
    R = pd.DataFrame(rows)
    S = R[R.target != "none"].reset_index(drop=True)                 # sweep only (numeric targets)

    # best stable region: 3-neighbour moving avg of total_pts along the target axis
    v = S["total_pts"].values; nb = np.array([np.mean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
    bi = int(np.argmax(nb)); center = int(S.target.iloc[bi])
    lo, hi = max(0, bi - 1), min(len(S), bi + 2); region = S.iloc[lo:hi]
    best_single = int(S.target.iloc[int(np.argmax(v))])

    with pd.ExcelWriter(OUTDIR / "breakout_target_sweep.xlsx", engine="openpyxl") as w:
        R.to_excel(w, sheet_name="Sweep_and_Baseline", index=False)
        pd.DataFrame([
            {"metric": "Strategy", "value": "DAILY pure breakout (0-pt margin, WITH gap) + PROFIT-TARGET-to-FLAT sweep (distinct variant)"},
            {"metric": "Period", "value": period}, {"metric": "P&L", "value": "GROSS index points"},
            {"metric": "Gap-while-flat", "value": "CONFIRMED: gap handling only applies while IN a position; re-entry from flat is a plain breakout"},
            {"metric": "Exit types", "value": "target (->flat), flip (opposite breakout, stays in market), period_end"},
            {"metric": "NO-TARGET baseline", "value": f"{base['trades']} trades, {base['total_pts']} pts, win {base['win_%']}% (always-in flip)"},
            {"metric": "Best single target", "value": f"{best_single} = {S.total_pts.iloc[int(np.argmax(v))]} pts"},
            {"metric": "BEST STABLE REGION (3-neighbour)", "value": f"center {center} | targets {list(region.target)} | pts {list(region.total_pts)}"},
            {"metric": "Region IS/OOS", "value": f"IS {list(region.IS_pts)} | OOS {list(region.OOS_pts)}"},
            {"metric": "OOS note", "value": f"IS < {OOS_SPLIT.date()} <= OOS ; check the region stays positive in both"},
        ]).to_excel(w, sheet_name="Summary", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(S.target, S.total_pts, "-o", ms=3, color="#1F6FB2", label="target variant")
    ax.axhline(base["total_pts"], color="#C0392B", ls="--", lw=1.4, label=f"no-target baseline ({base['total_pts']:.0f})")
    ax.axvspan(region.target.min(), region.target.max(), color="#2E8B57", alpha=.12, label=f"best region ~{center}")
    ax.set_title("NIFTY daily pure breakout — profit-target sweep (gross points)"); ax.set_xlabel("Target distance (points)"); ax.set_ylabel("Total P&L (points)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "target_sweep_curve.png", dpi=130); plt.close(fig)
    R.to_csv(OUTDIR / "breakout_target_sweep.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96 + "\nPROFIT-TARGET SWEEP — daily pure breakout (0-margin, with gap) | " + period + "\n" + "=" * 96)
    print(f"\nNO-TARGET baseline: {base['trades']} trades | {base['total_pts']:,} pts | win {base['win_%']}% | IS {base['IS_pts']} OOS {base['OOS_pts']}")
    print("\n--- sweep (every 4th shown) ---")
    print(S.iloc[::4][["target", "trades", "win_%", "total_pts", "pct_target", "pct_flip", "pct_end", "avg_flat_hrs", "IS_pts", "OOS_pts"]].to_string(index=False))
    print(f"\nbest single target {best_single} | BEST REGION center {center}: targets {list(region.target)} -> pts {list(region.total_pts)}")
    print(f"  region IS {list(region.IS_pts)} | OOS {list(region.OOS_pts)}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
