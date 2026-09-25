# -*- coding: utf-8 -*-
"""check_equity_data_last_date.py — verification-only status check (no new data pulled). Reports the exact
last date/timestamp available across the FULL NSE equity 1-min dataset (master_data/, all 1,609 stocks,
not restricted to the strategy's mcap entry filter), the last date the Volume-Breakout strategy actually
produced trades, and whether tail-end coverage is even across stocks.
"""
import sys, time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"


def last_ts_via_metadata(fn):
    """Try to get max timestamp from parquet row-group statistics (fast, no data load).
    Falls back to a full column read if stats aren't available/usable."""
    try:
        pf = pq.ParquetFile(fn)
        meta = pf.metadata
        col_idx = pf.schema_arrow.get_field_index("timestamp")
        if col_idx == -1:
            return None
        maxes = []
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(col_idx).statistics
            if stats is not None and stats.has_max_min:
                maxes.append(stats.max)
        if maxes:
            return max(maxes)
    except Exception:
        pass
    try:
        df = pd.read_parquet(fn, columns=["timestamp"])
        return df["timestamp"].max()
    except Exception:
        return None


def main():
    files = sorted(rb.MASTER_DIR.glob("*.parquet"))
    print(f"Full NSE equity 1-min dataset: {len(files)} stock files in {rb.MASTER_DIR}", flush=True)

    t0 = time.time()
    rows = []
    for i, fn in enumerate(files, 1):
        mx = last_ts_via_metadata(fn)
        rows.append({"symbol": fn.stem, "last_ts_raw": mx})
        if i % 400 == 0:
            print(f"  ...{i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    print(f"scan complete in {time.time()-t0:.0f}s", flush=True)

    R["last_ts"] = pd.to_datetime(R["last_ts_raw"], utc=True, errors="coerce")
    R["last_ts_ist"] = R["last_ts"].dt.tz_convert(IST).dt.tz_localize(None)
    R["last_date"] = R["last_ts_ist"].dt.date
    valid = R.dropna(subset=["last_date"])

    overall_last_date = valid["last_date"].max()
    overall_last_ts = valid.loc[valid["last_date"] == overall_last_date, "last_ts_ist"].max()
    print(f"\n=== FULL UNIVERSE: overall last date across all {len(valid)} readable stocks: {overall_last_date} ===")
    print(f"latest exact timestamp seen: {overall_last_ts}")

    date_counts = valid["last_date"].value_counts().sort_index(ascending=False)
    print("\n--- distribution of each stock's OWN last available date (top 15 most recent) ---")
    print(date_counts.head(15).to_string())

    n_at_overall_last = int((valid["last_date"] == overall_last_date).sum())
    n_before = len(valid) - n_at_overall_last
    print(f"\nstocks whose data reaches the overall last date ({overall_last_date}): {n_at_overall_last} of {len(valid)}")
    print(f"stocks whose OWN last date is EARLIER than the overall last date: {n_before} of {len(valid)}")
    lagging = valid[valid["last_date"] < overall_last_date].sort_values("last_date", ascending=False)
    print(f"\n--- of those lagging, the most-recent-but-still-behind stocks (top 15) ---")
    print(lagging[["symbol", "last_date"]].head(15).to_string(index=False))
    print(f"\nlagging stocks' last-date range: {lagging['last_date'].min()} .. {lagging['last_date'].max()}" if len(lagging) else "none lagging")

    # ---- Volume Breakout strategy's actual last trade date ----
    trades_fn = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
    if trades_fn.exists():
        tr = pd.read_excel(trades_fn, sheet_name="all_trades", usecols=["entry_date", "exit_date"])
        tr["entry_date"] = pd.to_datetime(tr["entry_date"])
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])
        last_entry = tr["entry_date"].max().date()
        last_exit = tr["exit_date"].max().date()
        print(f"\n=== VOLUME-BREAKOUT STRATEGY: last trade entry_date = {last_entry} | last trade exit_date = {last_exit} ===")
        print(f"gap between raw data's overall last date ({overall_last_date}) and strategy's last entry ({last_entry}): "
              f"{(overall_last_date - last_entry).days} calendar days")
    else:
        print(f"\n(baseline_final_performance.xlsx not found at {trades_fn} -- cannot report last trade date)")

    unreadable = R[R["last_ts_raw"].isna()]
    if len(unreadable):
        print(f"\nFLAG: {len(unreadable)} stock files could not be read at all (corrupt/empty?):")
        print(unreadable["symbol"].tolist()[:20])

    R.to_csv(rb.RESULTS / "equity_data_last_date_check.csv", index=False)
    print(f"\nSaved per-stock detail -> {rb.RESULTS}/equity_data_last_date_check.csv")


if __name__ == "__main__":
    main()
