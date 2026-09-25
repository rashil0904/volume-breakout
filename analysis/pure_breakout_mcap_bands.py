# -*- coding: utf-8 -*-
"""
pure_breakout_mcap_bands.py
===========================
Pure 52-week-high breakout across the full ₹1,500-50,000 Cr cap range, with a market-cap
band breakup. Entry: mcap in band + NEW 52w high on entry day + daily return >= +3% +
3:15pm entry. NO volume/lookback/target. ₹5L pool / ₹1L per trade.

FIXED EXIT (all trades): conditional split t1=09:30, t2=11:00 next day — return_at_930>0
-> exit @09:30 open; else -> exit @11:00 open (100% at one, not 50/50). No target.
This one fixed exit (the top combo from the 1,500-5,000 pure-breakout sweep) is applied
across every band, so the breakup isolates the market-cap effect, not exit timing.

Bands by ENTRY-DAY market cap (>=lower, <upper): 1500-5000, then 5000-step to 50000.
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "pure_breakout_mcap_bands"
DIAG = rb.RESULTS / "diagnostic_table_mcap1p5k50k.csv"
BASE_POOL, MAX_PER_STOCK, DAILY_POOL = 500_000, 100_000, 500_000
EXP023 = 0.0023
WIN_52, RET_MIN, SMALL = 252, 3.0, 20
HM_0930, HM_1100 = 570, 660
EDGES = [1500, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]


def band_of(mc):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= mc < EDGES[i + 1]:
            return f"{EDGES[i]:,}-{EDGES[i+1]:,}"
    return None


def build():
    diag = pd.read_csv(DIAG, usecols=["symbol", "date", "entry_price_315pm",
                                      "return_pct_vs_prev_close", "market_cap_value"],
                       parse_dates=["date"])
    sig = diag[(diag["return_pct_vs_prev_close"] >= RET_MIN)
               & diag["entry_price_315pm"].notna()].reset_index(drop=True)   # NO volume filter
    n_ret3 = len(sig)

    rows = []; o930 = []; o1100 = []
    n_no_hist = 0
    for sym, grp in sig.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            n_no_hist += len(grp); continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        dh = raw.groupby("date")["high"].max()
        ds = sorted(dh.index); dh = dh.reindex(ds)
        r52 = dh.rolling(WIN_52, min_periods=WIN_52).max().shift(1)    # ends day BEFORE entry
        r52_map, dh_map = r52.to_dict(), dh.to_dict()
        o930m = raw[raw["hm"] == HM_0930].groupby("date")["open"].last().to_dict()
        o1100m = raw[raw["hm"] == HM_1100].groupby("date")["open"].last().to_dict()
        for d, ep, mc in zip(grp["date"], grp["entry_price_315pm"], grp["market_cap_value"]):
            dd = pd.Timestamp(d).date()
            hi = r52_map.get(dd, np.nan)
            if hi is None or hi != hi:
                n_no_hist += 1; continue
            if not (dh_map.get(dd, np.nan) >= hi):
                continue
            j = bisect.bisect_right(ds, dd)
            nd = ds[j] if j < len(ds) else None
            rows.append({"date": dd, "symbol": sym, "entry": float(ep), "mcap": float(mc)})
            o930.append(o930m.get(nd, np.nan) if nd else np.nan)
            o1100.append(o1100m.get(nd, np.nan) if nd else np.nan)

    bk = pd.DataFrame(rows)
    bk["o930"] = o930; bk["o1100"] = o1100
    # pool-split sizing on the full breakout set's own daily counts
    cnt = bk.groupby("date")["symbol"].transform("size").values
    tgt = np.where(cnt <= 5, MAX_PER_STOCK, DAILY_POOL / cnt)
    sh = np.floor(tgt / bk["entry"].values)
    bk = bk[sh > 0].reset_index(drop=True); sh = sh[sh > 0]
    bk["shares"] = sh.astype(int); bk["cap"] = bk["shares"] * bk["entry"]
    # conditional split 09:30 / 11:00
    ret930 = (bk["o930"] - bk["entry"]) / bk["entry"] * 100
    bk["exit_price"] = np.where(ret930 > 0, bk["o930"], bk["o1100"])
    bk = bk[bk["exit_price"].notna()].reset_index(drop=True)
    bk["gross_pnl"] = bk["shares"] * (bk["exit_price"] - bk["entry"])
    bk["gross_ret"] = (bk["exit_price"] - bk["entry"]) / bk["entry"] * 100
    bk["net023_pnl"] = bk["gross_pnl"] - bk["cap"] * EXP023
    bk["net023_ret"] = bk["gross_ret"] - EXP023 * 100
    bk["band"] = bk["mcap"].map(band_of)
    return bk, n_ret3, n_no_hist


def metrics(df, label):
    n = len(df)
    g = df["gross_pnl"].sum()
    w = df[df["gross_pnl"] > 0]; l = df[df["gross_pnl"] <= 0]
    return {
        "mcap_band_cr": label, "n_trades": n,
        "win_rate_pct": round((df["gross_pnl"] > 0).mean() * 100, 2) if n else 0,
        "avg_return_per_trade_pct": round(df["gross_ret"].mean(), 4) if n else np.nan,
        "median_return_per_trade_pct": round(df["gross_ret"].median(), 4) if n else np.nan,
        "total_return_fixedbase_pct": round(g / BASE_POOL * 100, 4),
        "total_pnl_inr": round(g, 0),
        "avg_return_winning_trades_pct": round(w["gross_ret"].mean(), 4) if len(w) else np.nan,
        "avg_return_losing_trades_pct": round(l["gross_ret"].mean(), 4) if len(l) else np.nan,
        "avg_capital_deployed_per_trade": round(df["cap"].mean(), 0) if n else np.nan,
        "small_sample_flag": "n_trades<20" if n < SMALL else "",
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DIAG.exists():
        raise SystemExit(f"Missing {DIAG.name} — build the 1500-50000 diagnostic table first.")
    print("Building pure-breakout trade set over ₹1,500-50,000 Cr …")
    bk, n_ret3, n_no_hist = build()
    n = len(bk); ndays = bk["date"].nunique()
    print(f"  +3% signals (no volume, full band): {n_ret3:,}")
    print(f"  dropped (<252d history / missing parquet): {n_no_hist:,}")
    print(f"  pure-breakout trades: {n:,} over {ndays:,} days | avg {n/ndays:.2f}/day")

    # per-band rows + ALL
    rows = []
    for i in range(len(EDGES) - 1):
        lbl = f"{EDGES[i]:,}-{EDGES[i+1]:,}"
        rows.append(metrics(bk[bk["band"] == lbl], lbl))
    rows.append(metrics(bk, "ALL (1,500-50,000)"))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "pure_breakout_mcap_bands.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "pure_breakout_mcap_bands.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="by_mcap_band", index=False)

    # reconciliation
    bandrows = tbl[tbl["mcap_band_cr"] != "ALL (1,500-50,000)"]
    print(f"\nReconcile: Σ band n_trades = {int(bandrows['n_trades'].sum())} vs ALL = "
          f"{int(tbl[tbl.mcap_band_cr=='ALL (1,500-50,000)'].n_trades.iloc[0])}")

    # ── headline gross + net@0.23% ──
    gtot = bk["gross_pnl"].sum(); ntot = bk["net023_pnl"].sum()
    print("\n=== HEADLINE ₹1,500-50,000 Cr (exit 09:30/11:00) ===")
    print(f"  n_trades {n:,} | win_rate {round((bk.gross_pnl>0).mean()*100,2)}%")
    print(f"  GROSS: total_return {round(gtot/BASE_POOL*100,4)}% | avg {round(bk.gross_ret.mean(),4)}% "
          f"| median {round(bk.gross_ret.median(),4)}%")
    print(f"  NET@0.23%: total_return {round(ntot/BASE_POOL*100,4)}% | avg {round(bk.net023_ret.mean(),4)}% "
          f"| median {round(bk.net023_ret.median(),4)}%")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 130)
    print("PURE 52w-HIGH BREAKOUT BY MARKET-CAP BAND (exit 09:30/11:00; gross)")
    print("=" * 130)
    print(tbl.to_string(index=False))

    # ── charts ──
    bl = bandrows["mcap_band_cr"].tolist()
    small = bandrows["small_sample_flag"].values != ""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#b0b0b0" if s else "#1f4e79" for s in small]
    bars = ax.bar(range(len(bl)), bandrows["avg_return_per_trade_pct"], color=colors, edgecolor="white")
    for b, nt, s in zip(bars, bandrows["n_trades"], small):
        ax.annotate(f"n={nt}", (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7,
                    color="#888" if s else "#333")
    ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl, rotation=40, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("market-cap band (₹ Cr)"); ax.set_ylabel("avg return per trade %")
    ax.set_title("Pure 52w-high breakout — avg return/trade by market-cap band\n"
                 "(exit 09:30/11:00; grey = n<20; bar label = n_trades)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUTDIR / "avg_return_by_band.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(bl)), bandrows["n_trades"], color="#2ca25f", edgecolor="white")
    for i, nt in enumerate(bandrows["n_trades"]):
        ax.annotate(f"{nt}", (i, nt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("market-cap band (₹ Cr)"); ax.set_ylabel("n_trades")
    ax.set_title("Pure 52w-high breakout — signal count by market-cap band", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUTDIR / "n_trades_by_band.png", dpi=130); plt.close(fig)

    flagged = bandrows[bandrows["small_sample_flag"] != ""]
    print(f"\nSmall-sample bands (n<{SMALL}): "
          f"{', '.join(flagged['mcap_band_cr']) if len(flagged) else 'none'}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
