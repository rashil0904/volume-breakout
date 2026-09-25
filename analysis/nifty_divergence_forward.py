# -*- coding: utf-8 -*-
"""
nifty_divergence_forward.py — forward-move diagnostic on the detected Nifty RSI divergences.
No trades: measures how Nifty behaves after each divergence vs an unconditional baseline.

REFERENCE (entry proxy, flag a): for each divergence, r = the candle AFTER the confirming candle
(confirm_seq + 1) — the first actionable price post-confirmation, no look-ahead. reference_open = open[r].
HORIZONS (flag b): N in [3,5,7,14] hourly candles. Window = [r, r+N-1] (the reference/entry candle
plus the next N-1 — N bars held from entry).
DIRECTION (flag c/d): bullish predicts UP, bearish predicts DOWN.
  bullish : favorable = (maxHigh - ref)/ref*100 ; adverse = (minLow - ref)/ref*100 (neg)
  bearish : favorable = (ref - minLow)/ref*100 ; adverse = (ref - maxHigh)/ref*100 (neg)
  final_directional = signed close-at-horizon move in the predicted direction.
BASELINE (flag e): the SAME directional excursions over EVERY hourly candle (unconditional) at each N,
so the divergence numbers can be judged against Nifty's baseline drift/range.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

HOURLY = rb.RESULTS / "nifty_hourly_rsi" / "nifty_hourly_rsi.csv"
DIVS = rb.RESULTS / "nifty_rsi_divergence" / "divergences.csv"
OUTDIR = rb.RESULTS / "nifty_divergence_forward"
HORIZONS = [3, 5, 7, 14]


def excursions(o, hi, lo, cl, r, N, n):
    """Return (up_exc, down_exc, up_final) as % of ref open, over window [r, r+N-1]. None if OOB."""
    if r < 0 or r + N > n:
        return None
    ref = o[r]
    if not (ref == ref and ref > 0):
        return None
    mh = np.max(hi[r:r + N]); ml = np.min(lo[r:r + N]); fc = cl[r + N - 1]
    return ((mh - ref) / ref * 100.0, (ref - ml) / ref * 100.0, (fc - ref) / ref * 100.0)


def stat_block(fav, adv, fin):
    fav = np.asarray(fav); adv = np.asarray(adv); fin = np.asarray(fin)
    aadv = np.abs(adv)
    return {
        "n": len(fav),
        "avg_favorable": round(float(fav.mean()), 4), "median_favorable": round(float(np.median(fav)), 4),
        "avg_adverse": round(float(adv.mean()), 4), "median_adverse": round(float(np.median(adv)), 4),
        "avg_final_directional": round(float(fin.mean()), 4),
        "favor_against_ratio": round(float(fav.mean() / aadv.mean()), 3) if aadv.mean() else np.nan,
        "pct_favor_gt_adverse": round(float((fav > aadv).mean() * 100), 1),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    H = pd.read_csv(HOURLY, parse_dates=["date"])           # same row order as seq in divergences.csv
    o = H["open"].values.astype(float); hi = H["high"].values.astype(float)
    lo = H["low"].values.astype(float); cl = H["close"].values.astype(float)
    n = len(H)
    D = pd.read_csv(DIVS, parse_dates=["p2_date", "confirm_date"])
    D["ref_seq"] = D["confirm_seq"].astype(int) + 1

    # ── per-divergence forward moves ──
    rows_out = []
    div_rows = {N: {"bullish": {"fav": [], "adv": [], "fin": []},
                    "bearish": {"fav": [], "adv": [], "fin": []}} for N in HORIZONS}
    detail = []
    for _, d in D.iterrows():
        r = int(d["ref_seq"]); typ = d["type"]
        rowd = {"type": typ, "p2_date": d["p2_date"].date(), "confirm_date": d["confirm_date"].date(),
                "ref_seq": r, "ref_open": round(float(o[r]), 2) if r < n else np.nan}
        for N in HORIZONS:
            ex = excursions(o, hi, lo, cl, r, N, n)
            if ex is None:
                continue
            up, down, upf = ex
            if typ == "bullish":
                fav, adv, fin = up, -down, upf
            else:
                fav, adv, fin = down, -up, -upf
            div_rows[N][typ]["fav"].append(fav); div_rows[N][typ]["adv"].append(adv); div_rows[N][typ]["fin"].append(fin)
            rowd[f"N{N}_favorable"] = round(fav, 4); rowd[f"N{N}_adverse"] = round(adv, 4); rowd[f"N{N}_final"] = round(fin, 4)
        detail.append(rowd)
    detail = pd.DataFrame(detail)

    # ── unconditional baseline over ALL candles ──
    base_up = {N: [] for N in HORIZONS}; base_down = {N: [] for N in HORIZONS}; base_fin = {N: [] for N in HORIZONS}
    for r in range(n):
        for N in HORIZONS:
            ex = excursions(o, hi, lo, cl, r, N, n)
            if ex is None:
                continue
            up, down, upf = ex
            base_up[N].append(up); base_down[N].append(down); base_fin[N].append(upf)

    # ── assemble tables ──
    for N in HORIZONS:
        for typ in ["bullish", "bearish"]:
            b = div_rows[N][typ]
            if not b["fav"]:
                continue
            rows_out.append({"kind": "divergence", "type": typ, "N": N, **stat_block(b["fav"], b["adv"], b["fin"])})
        # baselines mirror the directional convention
        up = np.array(base_up[N]); down = np.array(base_down[N]); fin = np.array(base_fin[N])
        rows_out.append({"kind": "baseline", "type": "bullish", "N": N, **stat_block(up, -down, fin)})
        rows_out.append({"kind": "baseline", "type": "bearish", "N": N, **stat_block(down, -up, -fin)})
    T = pd.DataFrame(rows_out)

    # divergence-vs-baseline edge table (favorable & final lift over baseline)
    edge = []
    for N in HORIZONS:
        for typ in ["bullish", "bearish"]:
            dv = T[(T.kind == "divergence") & (T.type == typ) & (T.N == N)]
            bs = T[(T.kind == "baseline") & (T.type == typ) & (T.N == N)]
            if len(dv) and len(bs):
                dv = dv.iloc[0]; bs = bs.iloc[0]
                edge.append({"type": typ, "N": N, "n_div": dv["n"],
                             "div_avg_favorable": dv["avg_favorable"], "base_avg_favorable": bs["avg_favorable"],
                             "favorable_lift_vs_base": round(dv["avg_favorable"] - bs["avg_favorable"], 4),
                             "div_avg_final": dv["avg_final_directional"], "base_avg_final": bs["avg_final_directional"],
                             "final_lift_vs_base": round(dv["avg_final_directional"] - bs["avg_final_directional"], 4),
                             "div_ratio": dv["favor_against_ratio"], "base_ratio": bs["favor_against_ratio"],
                             "div_pct_favor_gt_adv": dv["pct_favor_gt_adverse"], "base_pct": bs["pct_favor_gt_adverse"]})
    E = pd.DataFrame(edge)

    # ── distribution at N=7 ──
    dist_rows = []
    for typ in ["bullish", "bearish"]:
        for metric, arr in [("favorable", div_rows[7][typ]["fav"]), ("adverse", div_rows[7][typ]["adv"])]:
            a = np.asarray(arr)
            dist_rows.append({"type": typ, "metric": metric, "n": len(a),
                              "min": round(a.min(), 3), "q25": round(np.percentile(a, 25), 3),
                              "median": round(np.median(a), 3), "q75": round(np.percentile(a, 75), 3),
                              "max": round(a.max(), 3), "mean": round(a.mean(), 3)})
    dist = pd.DataFrame(dist_rows)

    with pd.ExcelWriter(OUTDIR / "nifty_divergence_forward.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="forward_stats", index=False)
        E.to_excel(w, sheet_name="edge_vs_baseline", index=False)
        dist.to_excel(w, sheet_name="distribution_N7", index=False)
        detail.to_excel(w, sheet_name="per_divergence", index=False)

    pd.set_option("display.width", 240)
    print("=" * 100 + "\nNIFTY RSI-DIVERGENCE FORWARD MOVE (hourly; ref = open of candle after confirm; window [r, r+N-1])\n" + "=" * 100)
    print("\n--- DIVERGENCE vs BASELINE (favorable = move in predicted direction; all % of ref open) ---")
    show = ["kind", "type", "N", "n", "avg_favorable", "median_favorable", "avg_adverse", "median_adverse",
            "avg_final_directional", "favor_against_ratio", "pct_favor_gt_adverse"]
    print(T[show].sort_values(["type", "N", "kind"]).to_string(index=False))
    print("\n--- EDGE vs BASELINE (does the divergence beat unconditional Nifty drift/range?) ---")
    print(E.to_string(index=False))
    print("\n--- DISTRIBUTION at N=7 (spread, not just averages) ---")
    print(dist.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
