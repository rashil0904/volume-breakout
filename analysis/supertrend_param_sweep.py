# -*- coding: utf-8 -*-
"""supertrend_param_sweep.py — Supertrend (period,factor) sweep for the NIFTY dual-TF strategy, 15min & 1hr
legs swept INDEPENDENTLY. period 7..14 x factor 2..7 = 48 combos/leg. Baseline (10,3). Same per-leg logic
as supertrend_dual_tf (always-in +1/-1, close-confirmed flip @close, index points, 1 lot). Reports full grid
(points/trades/win) per leg, best stable REGION (3x3 neighbourhood, overfit-guarded), baseline-vs-best delta,
and IN-SAMPLE -> OUT-OF-SAMPLE validation. Only the Supertrend inputs vary; nothing else changes.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend

DATA = rb.BASE / "data" / "nifty_15min_ohlc.csv"
OUTDIR = rb.RESULTS / "supertrend_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
PERIODS = list(range(7, 15)); FACTORS = [2, 3, 4, 5, 6, 7]; BASE = (10, 3)
WARM = max(PERIODS) + 1                          # common start so every combo is scored on identical bars
OOS_SPLIT = "2024-07-01"                         # in-sample < split ; out-of-sample >= split
COST_PER_TRADE = 15.0                            # flat 15 index points per trade (round-trip) -> NET = gross - 15*trades


def mtm(direction, close, i0, i1):               # index-point P&L over bars [i0,i1) (clean window attribution)
    dc = np.diff(close)
    return float(np.sum(direction[i0:i1 - 1] * dc[i0:i1 - 1]))


def trades(direction, close, i0, i1):            # flip-to-flip within window -> n_trades, win%
    fl = [i for i in range(max(i0, 1), i1) if direction[i] != direction[i - 1]]
    pn = []
    for k, i in enumerate(fl):
        j = fl[k + 1] if k + 1 < len(fl) else i1 - 1
        pn.append(direction[i] * (close[j] - close[i]))
    pn = np.array(pn)
    return len(fl), (round((pn > 0).mean() * 100, 1) if len(pn) else 0.0)


def sweep_leg(high, low, close, is_end):
    n = len(close); rows = {}
    for p in PERIODS:
        for f in FACTORS:
            d = supertrend(high, low, close, p, f)
            nt, wr = trades(d, close, WARM, n); nti = trades(d, close, WARM, is_end)[0]; nto = trades(d, close, is_end, n)[0]
            gf = mtm(d, close, WARM, n); gi = mtm(d, close, WARM, is_end); go = mtm(d, close, is_end, n)
            rows[(p, f)] = {"points_full": round(gf, 1), "trades_full": nt, "win_full": wr,
                            "net_full": round(gf - COST_PER_TRADE * nt, 1), "cost_full": round(COST_PER_TRADE * nt, 1),
                            "points_is": round(gi, 1), "points_oos": round(go, 1),
                            "net_is": round(gi - COST_PER_TRADE * nti, 1), "net_oos": round(go - COST_PER_TRADE * nto, 1),
                            "trades_is": nti, "trades_oos": nto}
    return rows


def grid(rows, key):
    return pd.DataFrame({f: [rows[(p, f)][key] for p in PERIODS] for f in FACTORS}, index=[f"p{p}" for p in PERIODS])


def best_region(rows, key="points_full"):
    """3x3 neighbourhood-mean best cell (robust to single-cell overfit). returns center + members + stats."""
    M = np.array([[rows[(p, f)][key] for f in FACTORS] for p in PERIODS], float)
    nb = np.full(M.shape, -1e18)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            sub = M[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            nb[i, j] = sub.mean()
    bi, bj = np.unravel_index(np.argmax(nb), nb.shape)
    members = [(PERIODS[i], FACTORS[j]) for i in range(max(0, bi - 1), min(len(PERIODS), bi + 2))
               for j in range(max(0, bj - 1), min(len(FACTORS), bj + 2))]
    vals = [rows[m][key] for m in members]
    return {"center": (PERIODS[bi], FACTORS[bj]), "neigh_mean": round(float(nb[bi, bj]), 1),
            "members": members, "region_min": round(min(vals), 1), "region_max": round(max(vals), 1)}


def rank_of(rows, combo, key="points_full"):
    vals = sorted((rows[(p, f)][key] for p in PERIODS for f in FACTORS), reverse=True)
    v = rows[combo][key]; r = vals.index(v) + 1; n = len(vals)
    return r, n, round((n - r) / (n - 1) * 100, 1)   # rank (1=best), n, percentile


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"]); d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    is_end15 = int((d["ts"] < OOS_SPLIT).sum())
    h15, l15, c15 = d["high"].values, d["low"].values, d["close"].values
    mod = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d["hb"] = pd.factorize(d["ts"].dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0]
    hourly = d.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"), ts=("ts", "last")).sort_index()
    is_end1h = int((hourly["ts"] < OOS_SPLIT).sum())
    period = f"{d.ts.iloc[0].date()} .. {d.ts.iloc[-1].date()}"

    legs = {"15min": sweep_leg(h15, l15, c15, is_end15),
            "1hour": sweep_leg(hourly["h"].values, hourly["l"].values, hourly["c"].values, is_end1h)}

    with pd.ExcelWriter(OUTDIR / "supertrend_sweep.xlsx", engine="openpyxl") as w:
        info = [{"metric": "Sweep", "value": "period 7..14 x factor 2..7 = 48 combos/leg; 15min & 1hr INDEPENDENT"},
                {"metric": "Baseline", "value": "(10,3) both legs"}, {"metric": "Period", "value": period},
                {"metric": "Metric", "value": "index points (spot); MTM per leg; flip-to-flip trades for win%"},
                {"metric": "OOS split", "value": f"in-sample < {OOS_SPLIT} <= out-of-sample"},
                {"metric": "Best region", "value": "highest 3x3 neighbourhood-mean of full-period points (overfit-guarded; report cluster not single cell)"}]
        info += [{"metric": "COST MODEL", "value": f"NET = gross - {int(COST_PER_TRADE)} pts x trades; best-region & ranks now on NET points"}]
        cmp_rows = []
        for leg, rows in legs.items():
            br = best_region(rows, "net_full"); bc = br["center"]
            br_rank = rank_of(rows, bc, "net_full"); base_rank = rank_of(rows, BASE, "net_full")
            best_single = max(((p, f) for p in PERIODS for f in FACTORS), key=lambda c: rows[c]["net_full"])
            for tag, combo in [("BASELINE (10,3)", BASE), ("BEST-REGION center (net)", bc), ("best single cell (net)", best_single)]:
                r = rows[combo]
                cmp_rows.append({"leg": leg, "which": tag, "period": combo[0], "factor": combo[1], "trades": r["trades_full"], "win_%": r["win_full"],
                                 "GROSS_pts": r["points_full"], "cost_pts": r["cost_full"], "NET_pts": r["net_full"], "NET_IS": r["net_is"], "NET_OOS": r["net_oos"]})
            cmp_rows.append({"leg": leg, "which": ">>> DELTA best-region - baseline (NET)", "period": bc[0], "factor": bc[1],
                             "trades": rows[bc]["trades_full"] - rows[BASE]["trades_full"], "win_%": round(rows[bc]["win_full"] - rows[BASE]["win_full"], 1),
                             "GROSS_pts": round(rows[bc]["points_full"] - rows[BASE]["points_full"], 1), "cost_pts": round(rows[bc]["cost_full"] - rows[BASE]["cost_full"], 1),
                             "NET_pts": round(rows[bc]["net_full"] - rows[BASE]["net_full"], 1), "NET_IS": round(rows[bc]["net_is"] - rows[BASE]["net_is"], 1), "NET_OOS": round(rows[bc]["net_oos"] - rows[BASE]["net_oos"], 1)})
            info += [{"metric": "", "value": ""},
                     {"metric": f"[{leg}] baseline (10,3) NET", "value": f"{rows[BASE]['net_full']} pts | rank {base_rank[0]}/{base_rank[1]} (pctile {base_rank[2]})"},
                     {"metric": f"[{leg}] best single cell (NET)", "value": f"{best_single} = {rows[best_single]['net_full']} net pts"},
                     {"metric": f"[{leg}] BEST REGION center (NET)", "value": f"{bc} | 3x3 neigh-mean {br['neigh_mean']} | region net {br['region_min']}..{br['region_max']}"},
                     {"metric": f"[{leg}] best-region members", "value": str(br["members"])}]
            grid(rows, "net_full").to_excel(w, sheet_name=f"{leg}_NET")
            grid(rows, "points_full").to_excel(w, sheet_name=f"{leg}_Gross")
            grid(rows, "trades_full").to_excel(w, sheet_name=f"{leg}_Trades")
            grid(rows, "win_full").to_excel(w, sheet_name=f"{leg}_Win")
            grid(rows, "net_is").to_excel(w, sheet_name=f"{leg}_NET_IS")
            grid(rows, "net_oos").to_excel(w, sheet_name=f"{leg}_NET_OOS")
        pd.DataFrame(info).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(cmp_rows).to_excel(w, sheet_name="Baseline_vs_Best", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nSUPERTREND (period,factor) SWEEP — NIFTY dual-TF, legs INDEPENDENT | " + period + "\n" + "=" * 96)
    print(f"*** NET = gross - {int(COST_PER_TRADE)} pts/trade ***")
    for leg, rows in legs.items():
        br = best_region(rows, "net_full"); bc = br["center"]; base_rank = rank_of(rows, BASE, "net_full")
        print(f"\n### {leg} — full-period NET POINTS grid (rows=period, cols=factor) ###")
        print(grid(rows, "net_full").to_string())
        print(f"baseline (10,3): NET {rows[BASE]['net_full']} (gross {rows[BASE]['points_full']} - cost {rows[BASE]['cost_full']}), {rows[BASE]['trades_full']} tr | rank {base_rank[0]}/{base_rank[1]} (pctile {base_rank[2]})")
        print(f"BEST REGION center {bc}: NET {rows[bc]['net_full']} (gross {rows[bc]['points_full']} - cost {rows[bc]['cost_full']}), {rows[bc]['trades_full']} tr | 3x3 neigh-mean {br['neigh_mean']} | members {br['members']}")
        print(f"  NET IS/OOS: baseline {rows[BASE]['net_is']}/{rows[BASE]['net_oos']}  vs  region-center {rows[bc]['net_is']}/{rows[bc]['net_oos']}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
