# -*- coding: utf-8 -*-
"""vb_vwap_close_move_segmentation.py — diagnostic segmentation (NO strategy-logic changes) of the locked
Volume-Breakout baseline's existing trades (baseline_final_performance.xlsx, 3,436 trades) by day-over-day
VWAP-CLOSE move: (entry_day's own VWAP-close - previous_day's VWAP-close) / previous_day's VWAP-close *
100. "VWAP-close" = VWAP of the last 30 one-min candles (15:00-15:29), typical price (H+L+C)/3, EXACTLY
the same formula prepare_data.py already uses for prev_day_vwap_close (the strategy's own entry-filter
reference) -- reused here, not reinvented. previous_day's VWAP-close is already available directly from
diagnostic_table.csv's own prev_day_vwap_close column (joined on symbol+entry_date); entry_day's OWN
VWAP-close is computed fresh from each stock's raw 1-min parquet (not already stored anywhere), using the
identical window/formula.

This is explicitly NOT the entry fill price (which differs per Category A/B/C mechanics) -- it is a pure
day-over-day VWAP-close comparison, independent of how the trade was actually filled.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

TRADES = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
DIAG = rb.RESULTS / "diagnostic_table.csv"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "vwap_close_move_segmentation"; OUTDIR.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"
VWAP_LO, VWAP_HI = 900, 929   # 15:00-15:29, 30 one-min candles


def compute_vwap_by_date(sym):
    pq = rb.MASTER_DIR / f"{sym}.parquet"
    if not pq.exists():
        return {}
    raw = pd.read_parquet(pq, columns=["timestamp", "high", "low", "close", "volume"])
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    last30 = raw[(raw["hm"] >= VWAP_LO) & (raw["hm"] <= VWAP_HI)].copy()
    last30["tp"] = (last30["high"] + last30["low"] + last30["close"]) / 3.0
    last30["tp_vol"] = last30["tp"] * last30["volume"]
    g = last30.groupby("date")[["tp_vol", "volume"]].sum()
    vwap = (g["tp_vol"] / g["volume"]).where(g["volume"] > 0, np.nan)
    return vwap.to_dict()


def main():
    tr = pd.read_excel(TRADES, sheet_name="all_trades",
                        usecols=["symbol", "entry_date", "category", "long_pnl", "short_pnl", "gross_pnl",
                                 "short_exit_type", "capital_deployed"])
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date
    print(f"backtest trades: {len(tr):,} | distinct symbols: {tr['symbol'].nunique():,}", flush=True)

    diag = pd.read_csv(DIAG, usecols=["symbol", "date", "prev_day_vwap_close"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    tr = tr.merge(diag.rename(columns={"date": "entry_date", "prev_day_vwap_close": "prev_day_vwap"}),
                  on=["symbol", "entry_date"], how="left")
    print(f"joined prev_day_vwap: {tr['prev_day_vwap'].notna().sum():,} of {len(tr):,}", flush=True)

    entry_vwaps = []
    for sym, g in tr.groupby("symbol", sort=False):
        vwap_map = compute_vwap_by_date(sym)
        for idx, r in g.iterrows():
            entry_vwaps.append((idx, vwap_map.get(r["entry_date"], np.nan)))
    ev = pd.Series({i: v for i, v in entry_vwaps})
    tr["entry_day_vwap"] = tr.index.map(ev)

    valid = tr.dropna(subset=["entry_day_vwap", "prev_day_vwap"]).copy()
    valid = valid[valid["prev_day_vwap"] > 0]
    print(f"trades with both VWAP values resolvable: {len(valid):,} of {len(tr):,}", flush=True)

    valid["vwap_move_pct"] = (valid["entry_day_vwap"] - valid["prev_day_vwap"]) / valid["prev_day_vwap"] * 100

    below5 = valid[valid["vwap_move_pct"] < 5]
    print(f"\nFLAG: trades with VWAP-close move < 5% under this metric: {len(below5)} ({len(below5)/len(valid)*100:.2f}%)", flush=True)
    if len(below5):
        print(below5[["symbol", "entry_date", "vwap_move_pct"]].sort_values("vwap_move_pct").head(15).to_string(index=False))

    labels = ["5-10%", "10-15%", "15-20%", ">20%"]
    valid["bucket"] = pd.cut(valid["vwap_move_pct"], bins=[5, 10, 15, 20, np.inf], labels=labels, right=False)
    # trades below 5% (if any) or exactly at boundary handled by pd.cut; anything unbucketed (e.g. <5%) -> NaN bucket
    unbucketed = valid[valid["bucket"].isna()]
    if len(unbucketed):
        print(f"\n(also unbucketed, i.e. <5% or otherwise outside range: {len(unbucketed)} trades -- excluded from bucket table, shown separately)")

    has_short = valid["short_exit_type"] != "no_short"

    def block(df, pnl_col, label, require_short=False):
        rows = []
        for lbl in labels:
            sub = df[df["bucket"] == lbl]
            if require_short:
                sub = sub[has_short.loc[sub.index]]
            n = len(sub)
            if n == 0:
                rows.append({"bucket": lbl, "n_trades": 0, "win_rate_pct": np.nan, "total_pnl": np.nan,
                            "avg_pnl": np.nan, "max_profit": np.nan, "max_loss": np.nan})
                continue
            win = sub[pnl_col] > 0
            rows.append({"bucket": lbl, "n_trades": n, "win_rate_pct": round(win.mean() * 100, 2),
                        "total_pnl": round(sub[pnl_col].sum(), 1), "avg_pnl": round(sub[pnl_col].mean(), 2),
                        "max_profit": round(sub[pnl_col].max(), 1), "max_loss": round(sub[pnl_col].min(), 1)})
        R = pd.DataFrame(rows)
        print(f"\n=== {label} ===")
        print(R.to_string(index=False))
        return R

    LONG = block(valid, "long_pnl", "LONG-ONLY (by VWAP-close move bucket)")
    SHORT = block(valid, "short_pnl", "SHORT-ONLY (double-down leg; no_short trades excluded)", require_short=True)
    COMBINED = block(valid, "gross_pnl", "COMBINED (long+short, gross)")

    pd.set_option("display.width", 200)

    with pd.ExcelWriter(OUTDIR / "vwap_close_move_segmentation.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Diagnostic segmentation (no strategy-logic changes) of the locked baseline's 3,436 trades "
                      "by day-over-day VWAP-close move: (entry_day's own VWAP-close - prev_day's VWAP-close) / "
                      "prev_day's VWAP-close * 100. VWAP-close = VWAP of the last 30 one-min candles (15:00-"
                      "15:29), typical price (H+L+C)/3 -- identical formula/window to prepare_data.py's own "
                      "prev_day_vwap_close (the strategy's entry-filter reference), reused not reinvented."},
            {"note": "This is NOT the entry fill price (Category A/B/C mechanics can fill well above/below the "
                      "raw VWAP-close move) -- it is a pure day-over-day VWAP comparison, independent of fill."},
            {"note": f"{len(below5)} of {len(valid)} trades ({len(below5)/len(valid)*100:.2f}%) show a VWAP-close "
                      "move BELOW 5% under this metric, despite the entry filter requiring >=5% under its OWN "
                      "return definition -- confirms the two metrics are NOT identical (the entry filter's return "
                      "is based on 15:00 OPEN vs prev-day VWAP-close, not entry-day's own end-of-day VWAP-close; "
                      "an entry-day pullback after the 3pm trigger can easily erase the move by day's end). "
                      "Flagged as expected definitional divergence, not a data error -- see the Below_5pct sheet."},
            {"note": "LONG-ONLY/SHORT-ONLY/COMBINED all use GROSS pnl (long_pnl, short_pnl, gross_pnl) -- no "
                      "cost allocation is naturally splittable per leg, so net_A/net_B are not broken out here."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        LONG.to_excel(w, sheet_name="Long_Only", index=False)
        SHORT.to_excel(w, sheet_name="Short_Only", index=False)
        COMBINED.to_excel(w, sheet_name="Combined", index=False)
        if len(below5):
            below5.to_excel(w, sheet_name="Below_5pct_Flagged", index=False)
        valid.to_excel(w, sheet_name="All_Trades_with_VWAP_Move", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 28)

    print(f"\nSaved -> {OUTDIR}/vwap_close_move_segmentation.xlsx")


if __name__ == "__main__":
    main()
