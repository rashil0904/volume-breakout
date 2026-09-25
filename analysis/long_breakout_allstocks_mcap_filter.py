# -*- coding: utf-8 -*-
"""long_breakout_allstocks_mcap_filter.py — post-filters the already-computed all-stocks long-breakout
trades (results/long_breakout_k25_n20_allstocks/..._trades.csv, 17,747 trades) to market cap > 1500 Cr
at entry, using the RAW mcap_cache/*.xlsx snapshots (unrestricted, unlike the project's pre-filtered
1500-5000/5000-10000/etc band tables) and the exact same "nearest preceding snapshot" convention as
prepare_data.py (bisect_right(snap_dates, entry_date) - 1, floored at 0). No re-run of signal generation
needed — the mcap band only relabels/subsets which trades are reported, matching returns_by_mcap_band.py's
existing convention elsewhere in this project.
"""
import sys
from pathlib import Path
from bisect import bisect_right
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

MDIR = rb.BASE / "mcap_cache"
TRADES = rb.RESULTS / "long_breakout_k25_n20_allstocks" / "long_breakout_k25_n20_allstocks_trades.csv"
OUTDIR = rb.RESULTS / "long_breakout_k25_n20_allstocks"; OUTDIR.mkdir(parents=True, exist_ok=True)
MCAP_MIN_CR = 1_500

SNAPS = [
    ("2022-03-31", "mcap_2022-03-31.xlsx"),
    ("2022-12-31", "mcap_2022-12-31.xlsx"),
    ("2023-03-31", "mcap_2023-03-31.xlsx"),
    ("2023-12-31", "mcap_2023-12-31.xlsx"),
    ("2024-03-28", "mcap_2024-03-28.xlsx"),
    ("2024-12-31", "mcap_2024-12-31.xlsx"),
    ("2025-12-31", "mcap_2025-12-31.xlsx"),
]


def load_snapshots():
    snap_dates, snap_maps, snap_labels = [], [], []
    for date_str, cache_file in SNAPS:
        path = MDIR / cache_file
        df = pd.read_excel(path, header=None, skiprows=1, usecols=[1, 3])
        df.columns = ["symbol", "mcap_lakhs"]
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["mcap_cr"] = pd.to_numeric(df["mcap_lakhs"], errors="coerce") / 100
        df = df.dropna(subset=["mcap_cr"])
        snap_dates.append(pd.Timestamp(date_str))
        snap_maps.append(df.set_index("symbol")["mcap_cr"].to_dict())
        snap_labels.append(date_str)
        print(f"  {date_str}: {len(df):,} symbols, mcap range {df.mcap_cr.min():.1f}-{df.mcap_cr.max():.1f} Cr", flush=True)
    return snap_dates, snap_maps, snap_labels


def main():
    print("Loading raw (unrestricted) mcap snapshots...", flush=True)
    snap_dates, snap_maps, snap_labels = load_snapshots()

    T = pd.read_csv(TRADES, parse_dates=["signal_date", "entry_date", "exit_date"])
    print(f"\ntotal trades before mcap filter: {len(T)}", flush=True)

    def lookup(row):
        i = max(bisect_right(snap_dates, row["entry_date"]) - 1, 0)
        return snap_maps[i].get(str(row["symbol"]).upper(), float("nan")), snap_labels[i]

    res = T.apply(lookup, axis=1, result_type="expand")
    T["entry_mcap_cr"] = res[0]; T["mcap_snapshot_used"] = res[1]

    missing = T["entry_mcap_cr"].isna().sum()
    print(f"trades with no mcap match (symbol not in any snapshot): {missing}", flush=True)

    F = T[T["entry_mcap_cr"] > MCAP_MIN_CR].copy()
    print(f"trades with mcap > {MCAP_MIN_CR:,} Cr at entry: {len(F)}", flush=True)

    def summarize(df):
        if df.empty:
            return {"n_trades": 0}
        p = df["net_return"]
        return {"n_trades": len(df), "win_rate_%": round((p > 0).mean() * 100, 2),
                "mean_net_return_%": round(p.mean() * 100, 3), "median_net_return_%": round(p.median() * 100, 3),
                "best_trade_%": round(p.max() * 100, 2), "worst_trade_%": round(p.min() * 100, 2),
                "avg_winner_%": round(df.loc[p > 0, "net_return"].mean() * 100, 2),
                "avg_loser_%": round(df.loc[p <= 0, "net_return"].mean() * 100, 2),
                "avg_holding_days": round(df["holding_days"].mean(), 1),
                "pct_exit_stop": round((df["exit_reason"] == "stop").mean() * 100, 1),
                "pct_exit_time": round((df["exit_reason"] == "time").mean() * 100, 1)}

    overall = summarize(F)
    print("\n=== OVERALL (mcap > 1,500 Cr) ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    yearly = F.copy(); yearly["year"] = yearly["entry_date"].dt.year
    yearly_stats = yearly.groupby("year").apply(
        lambda g: pd.Series({"n_trades": len(g), "win_rate_%": round((g.net_return > 0).mean() * 100, 1),
                              "mean_net_return_%": round(g.net_return.mean() * 100, 3)}),
        include_groups=False).reset_index()

    per_symbol = F.groupby("symbol").apply(
        lambda g: pd.Series({"n_trades": len(g), "win_rate_%": round((g.net_return > 0).mean() * 100, 1),
                              "mean_net_return_%": round(g.net_return.mean() * 100, 3),
                              "total_net_return_%": round(g.net_return.sum() * 100, 2)}),
        include_groups=False).reset_index().sort_values("total_net_return_%", ascending=False)

    extreme = F[F.net_return.abs() > 0.5].sort_values("net_return", ascending=False)

    with pd.ExcelWriter(OUTDIR / "long_breakout_k25_n20_mcap1500cr.xlsx", engine="openpyxl") as w:
        pd.DataFrame([overall]).to_excel(w, sheet_name="Overall", index=False)
        yearly_stats.to_excel(w, sheet_name="By_Year", index=False)
        per_symbol.to_excel(w, sheet_name="By_Symbol", index=False)
        extreme.to_excel(w, sheet_name="Extreme_Trades_gt50pct", index=False)
        F.to_excel(w, sheet_name="Trades", index=False)
    F.to_csv(OUTDIR / "long_breakout_k25_n20_mcap1500cr_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== BY YEAR ===")
    print(yearly_stats.to_string(index=False))
    print("\n=== TOP 15 SYMBOLS ===")
    print(per_symbol.head(15).to_string(index=False))
    print("\n=== BOTTOM 15 SYMBOLS ===")
    print(per_symbol.tail(15).to_string(index=False))
    print(f"\n=== EXTREME TRADES (|net return| > 50%): {len(extreme)} ===")
    print(extreme[["symbol", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "net_return"]].head(15).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/long_breakout_k25_n20_mcap1500cr.xlsx")


if __name__ == "__main__":
    main()
