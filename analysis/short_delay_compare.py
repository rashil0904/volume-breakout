# -*- coding: utf-8 -*-
"""short_delay_compare.py — SHORTS-ONLY: compare the double-down short leg under two open timings.
CONFIG 1 (baseline): short opens AT the long exit (price=exit_price, minute=xhm).
CONFIG 2 (delay 1m): short opens at the OPEN of candle xhm+1 (09:25->09:26, 11:59->12:00, target->+1min).
Short size = filled long qty (unchanged). Cover = 14:39 open OR earlier if a 1-min low <= short_open*0.95;
in CONFIG 2 the 5% target is recomputed off the NEW delayed open. Long side identical (not re-reported).

Flags: (a) +1min applies to ALL shorts; (b) cover unchanged (14:39 / 5% target off new open);
(c) short size unchanged; (d) shorts-only isolation; (e) all prices are one-min candle OPENS.
Net = gross - 0.1% round-trip on short notional (shares*short_open).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
OUTDIR = rb.RESULTS / "short_delay_compare"
IST = rb.IST
T1, COVER_HM = 565, 879
SHORT_TGT, SR = 0.95, 0.0010
BASE_POOL = 500_000


def hm2s(m):
    return f"{int(m)//60:02d}:{int(m)%60:02d}"


def cover_from(hm, op, lo, open_min, short_open, o879):
    """Return (cover_price, cover_type) scanning minutes in (open_min, COVER_HM] within [T1,COVER_HM]."""
    tgt = short_open * SHORT_TGT
    mask = (hm > open_min) & (hm >= T1) & (hm <= COVER_HM)
    if mask.any() and (lo[mask] <= tgt).any():
        return tgt, "short_target_5pct"
    return o879, "short_cover_1439"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = pd.read_excel(BASE_XLSX, sheet_name="all_trades")
    T["exit_date"] = pd.to_datetime(T["exit_date"]).dt.date
    T["xhm"] = T["exit_time"].apply(lambda s: int(str(s)[:2]) * 60 + int(str(s)[3:5]))
    S = T[T["short_exit_type"].isin(["short_target_5pct", "short_cover_1439"]) & T["cover_price"].notna()].copy()
    print(f"baseline trades {len(T):,} | with a short leg {len(S):,}")

    rows, n_missing_delay = [], 0
    for sym, g in S.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq, columns=["timestamp", "open", "low"])
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(d=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: (x["hm"].values, x["open"].values.astype(float), x["low"].values.astype(float))
              for d, x in raw.groupby("d")}
        for r in g.itertuples():
            day = bd.get(r.exit_date)
            if day is None:
                continue
            hm, op, lo = day
            omap = dict(zip(hm, op))
            o879 = omap.get(COVER_HM, np.nan)
            if not (o879 == o879):
                continue
            sh = float(r.shares); xhm = int(r.xhm)
            # CONFIG 1 — open AT exit (price = exit_price)
            so1 = float(r.exit_price)
            cov1, ct1 = cover_from(hm, op, lo, xhm, so1, o879)
            pnl1 = sh * (so1 - cov1); notl1 = sh * so1; net1 = pnl1 - SR * notl1
            # CONFIG 2 — open 1 min later (price = open of candle xhm+1)
            xhm2 = xhm + 1
            so2 = omap.get(xhm2, np.nan)
            missing = not (so2 == so2)
            if missing:                                   # no candle to delay into -> keep config-1 open
                n_missing_delay += 1; so2 = so1
            cov2, ct2 = cover_from(hm, op, lo, xhm2, so2, o879)
            pnl2 = sh * (so2 - cov2); notl2 = sh * so2; net2 = pnl2 - SR * notl2
            rows.append({
                "symbol": sym, "date": str(r.exit_date), "category": r.category,
                "long_exit_type": r.long_exit_type, "long_exit_time": r.exit_time, "shares": int(sh),
                "c1_short_open_time": hm2s(xhm), "c2_short_open_time": hm2s(xhm2),
                "c1_short_open_price": round(so1, 4), "c2_short_open_price": round(so2, 4),
                "open_price_diff_c2_minus_c1": round(so2 - so1, 4),
                "c1_cover_type": ct1, "c1_cover_price": round(cov1, 4),
                "c2_cover_type": ct2, "c2_cover_price": round(cov2, 4),
                "c1_short_pnl": round(pnl1, 2), "c2_short_pnl": round(pnl2, 2),
                "c1_short_pnl_net": round(net1, 2), "c2_short_pnl_net": round(net2, 2),
                "c1_short_ret_pct": round((so1 - cov1) / so1 * 100, 4),
                "c2_short_ret_pct": round((so2 - cov2) / so2 * 100, 4),
                "delayed_open_missing": missing,
                "baseline_stored_short_pnl": round(float(r.short_pnl), 2)})
    D = pd.DataFrame(rows)

    # ── validation: config-1 must reproduce the stored baseline short pnl ──
    vdiff = (D["c1_short_pnl"] - D["baseline_stored_short_pnl"]).abs()
    print(f"validation config-1 vs stored short_pnl: max|diff|={vdiff.max():.2f}, "
          f"n mismatched(>1)={int((vdiff > 1).sum())} of {len(D)}")

    def block(cfg):
        pnl = D[f"{cfg}_short_pnl"]; net = D[f"{cfg}_short_pnl_net"]; ret = D[f"{cfg}_short_ret_pct"]
        ct = D[f"{cfg}_cover_type"]
        return {"config": "at_exit(C1)" if cfg == "c1" else "delayed_1min(C2)",
                "n_shorts": len(D),
                "short_total_pnl_gross_inr": round(pnl.sum(), 0),
                "short_total_pnl_net_inr": round(net.sum(), 0),
                "short_return_fixedbase_gross_pct": round(pnl.sum() / BASE_POOL * 100, 3),
                "short_return_fixedbase_net_pct": round(net.sum() / BASE_POOL * 100, 3),
                "short_win_rate_gross_pct": round((pnl > 0).mean() * 100, 2),
                "short_win_rate_net_pct": round((net > 0).mean() * 100, 2),
                "avg_short_ret_gross_pct": round(ret.mean(), 4), "median_short_ret_gross_pct": round(ret.median(), 4),
                "n_cover_5pct_target": int((ct == "short_target_5pct").sum()),
                "n_cover_1439": int((ct == "short_cover_1439").sum())}
    C1, C2 = block("c1"), block("c2")
    delta = {"config": "delta(C2-C1)"}
    for k in C1:
        if k == "config":
            continue
        delta[k] = round(C2[k] - C1[k], 4) if isinstance(C1[k], (int, float)) else ""
    CMP = pd.DataFrame([C1, C2, delta])
    avg_open_move = D["open_price_diff_c2_minus_c1"].mean()
    avg_open_move_pct = (D["open_price_diff_c2_minus_c1"] / D["c1_short_open_price"] * 100).mean()

    with pd.ExcelWriter(OUTDIR / "short_delay_compare.xlsx", engine="openpyxl") as w:
        CMP.to_excel(w, sheet_name="comparison", index=False)
        D.to_excel(w, sheet_name="per_trade", index=False)
    D.to_csv(OUTDIR / "short_delay_per_trade.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 96)
    print("DOUBLE-DOWN SHORT — open AT long exit (C1) vs 1-MIN DELAY (C2)  [shorts-only]")
    print("=" * 96)
    show = ["config", "n_shorts", "short_total_pnl_gross_inr", "short_total_pnl_net_inr",
            "short_return_fixedbase_gross_pct", "short_return_fixedbase_net_pct",
            "short_win_rate_gross_pct", "avg_short_ret_gross_pct", "median_short_ret_gross_pct",
            "n_cover_5pct_target", "n_cover_1439"]
    print(CMP[show].to_string(index=False))
    print(f"\n  avg short_open move (C2 - C1): {avg_open_move:+.4f} rupees  ({avg_open_move_pct:+.4f}% of C1 open)")
    print(f"  -> {'stocks tick UP in that minute on avg -> delayed short enters HIGHER (better)' if avg_open_move>0 else 'stocks tick DOWN in that minute on avg -> delayed short enters LOWER (worse)'}")
    if n_missing_delay:
        print(f"  note: {n_missing_delay} shorts had no candle at xhm+1 (kept C1 open, no delay applied)")
    print("\n--- per-trade sample (first 10) ---")
    print(D[["symbol", "date", "long_exit_type", "c1_short_open_time", "c2_short_open_time",
             "c1_short_open_price", "c2_short_open_price", "open_price_diff_c2_minus_c1",
             "c1_short_pnl", "c2_short_pnl"]].head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
