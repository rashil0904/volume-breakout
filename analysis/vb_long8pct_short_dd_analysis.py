# -*- coding: utf-8 -*-
"""vb_long8pct_short_dd_analysis.py — diagnostic: for main Volume-Breakout long trades exiting >=+8% via
the standard 9:25am exit ('positive_0925') or an early 17% target hit ('target_pre_0925' -- strictly before
9:25am, per spec), examine the corresponding double-down short opened at that same exit point: its P&L
(already in all_trades) and its Max Adverse Excursion (highest price reached before the short is covered,
via the 5% target or the 14:39 fallback). Read-only diagnostic -- no strategy changes.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "long8pct_short_dd_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
RET_THRESH = 8.0
COVER_HM = 14 * 60 + 39   # 14:39 fallback cover


def hm(t):
    h, m = t.split(":"); return int(h) * 60 + int(m)


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    qual = T[(T.long_exit_type.isin(["positive_0925", "target_pre_0925"])) & (T.gross_ret >= RET_THRESH)].copy()
    excluded_edge = T[(T.long_exit_type == "target_0925_1159") & (T.gross_ret >= RET_THRESH)]
    print(f"qualifying trades (positive_0925 / target_pre_0925, gross_ret >= {RET_THRESH}%): {len(qual)}", flush=True)
    print(f"NOTE: {len(excluded_edge)} trade(s) with target hit BETWEEN 9:25-11:59 and gross_ret>={RET_THRESH}% "
          f"exist but are excluded per literal spec (neither 'standard 9:25' nor 'earlier than 9:25') -- flagging, not including.", flush=True)

    rows = []; missing_data = []
    for _, r in qual.iterrows():
        sym = r["symbol"]; ex_date = pd.Timestamp(r["exit_date"])
        open_time_hm = hm(r["exit_time"])
        open_price = float(r["exit_price"])
        cover_price = float(r["cover_price"])
        short_exit_type = r["short_exit_type"]

        fn = MD / f"{sym}.parquet"
        if not fn.exists():
            missing_data.append((sym, ex_date.date(), "no master_data file")); continue
        df = pd.read_parquet(fn, columns=["timestamp", "high", "low"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.assign(ts=ts)
        day = df[df["ts"].dt.normalize() == ex_date.normalize()].sort_values("ts")
        day["mod"] = day["ts"].dt.hour * 60 + day["ts"].dt.minute
        window = day[(day["mod"] > open_time_hm) & (day["mod"] <= COVER_HM)]
        if window.empty:
            missing_data.append((sym, ex_date.date(), "no intraday candles after short-open")); continue

        if short_exit_type == "short_target_5pct":
            target_px = open_price * 0.95
            hit = window[window["low"] <= target_px]
            if hit.empty:
                cover_mod = window["mod"].iloc[-1]  # fallback, shouldn't happen
                missing_data.append((sym, ex_date.date(), "short_target_5pct but no low<=target found -- used last candle"));
            else:
                cover_mod = hit["mod"].iloc[0]
        else:  # short_cover_1439
            cover_mod = COVER_HM

        mae_window = window[window["mod"] <= cover_mod]
        mae_price = float(mae_window["high"].max()) if len(mae_window) else open_price
        mae_pct = (mae_price - open_price) / open_price * 100

        rows.append({
            "date": ex_date.date(), "symbol": sym,
            "long_exit_type": r["long_exit_type"], "long_exit_time": r["exit_time"], "long_exit_return_pct": round(float(r["gross_ret"]), 2),
            "short_open_price": round(open_price, 2), "short_open_time": r["exit_time"],
            "short_cover_price": round(cover_price, 2), "short_cover_time": f"{cover_mod//60:02d}:{cover_mod%60:02d}",
            "short_cover_reason": "5% target" if short_exit_type == "short_target_5pct" else "14:39 fallback",
            "short_pnl": round(float(r["short_pnl"]), 2),
            "short_mae_price": round(mae_price, 2), "short_mae_pct": round(mae_pct, 3),
            "shares": int(r["shares"]),
        })

    R = pd.DataFrame(rows)
    print(f"\nprocessed: {len(R)} | missing/flagged: {len(missing_data)}", flush=True)

    # ---- summary stats ----
    total_short_pnl = R["short_pnl"].sum()
    avg_short_pnl = R["short_pnl"].mean()
    win_rate = (R["short_pnl"] > 0).mean() * 100
    avg_mae = R["short_mae_pct"].mean()

    bins = [-0.001, 0, 0.5, 1, 2, 3, 5, 100]
    labels = ["0% (never adverse)", "0-0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", ">5%"]
    R["mae_bucket"] = pd.cut(R["short_mae_pct"], bins=bins, labels=labels)
    mae_dist = R["mae_bucket"].value_counts().reindex(labels)

    print("\n=== SUMMARY ===")
    print(f"n trades: {len(R)}")
    print(f"total short P&L: {round(total_short_pnl,1)}")
    print(f"avg short P&L/trade: {round(avg_short_pnl,1)}")
    print(f"short win rate: {round(win_rate,1)}%")
    print(f"avg short MAE %: {round(avg_mae,3)}%")
    print(f"median short MAE %: {round(R['short_mae_pct'].median(),3)}%")
    print("\nMAE distribution:")
    print(mae_dist.to_string())

    # ---- correlation: long exit strength vs short MAE/PnL ----
    r_mae, p_mae = np.nan, np.nan
    r_pnl, p_pnl = np.nan, np.nan
    from scipy import stats
    r_mae, p_mae = stats.pearsonr(R["long_exit_return_pct"], R["short_mae_pct"])
    r_pnl, p_pnl = stats.pearsonr(R["long_exit_return_pct"], R["short_pnl"])
    print(f"\ncorrelation long_exit_return_pct vs short_mae_pct: r={r_mae:.3f}, p={p_mae:.3f}")
    print(f"correlation long_exit_return_pct vs short_pnl: r={r_pnl:.3f}, p={p_pnl:.3f}")

    # split target_pre_0925 (strong/early momentum) vs positive_0925 (standard)
    strong = R[R.long_exit_type == "target_pre_0925"]; standard = R[R.long_exit_type == "positive_0925"]
    print(f"\n--- target_pre_0925 (early 17% target hit, n={len(strong)}) ---")
    print(f"avg short P&L: {round(strong.short_pnl.mean(),1) if len(strong) else 'n/a'} | avg MAE%: {round(strong.short_mae_pct.mean(),3) if len(strong) else 'n/a'} | win%: {round((strong.short_pnl>0).mean()*100,1) if len(strong) else 'n/a'}")
    print(f"--- positive_0925 (standard 9:25 exit, n={len(standard)}) ---")
    print(f"avg short P&L: {round(standard.short_pnl.mean(),1)} | avg MAE%: {round(standard.short_mae_pct.mean(),3)} | win%: {round((standard.short_pnl>0).mean()*100,1)}")

    with pd.ExcelWriter(OUTDIR / "long8pct_short_dd_diagnostic.xlsx", engine="openpyxl") as w:
        R.drop(columns=["mae_bucket"]).to_excel(w, sheet_name="Trade_Detail", index=False)
        pd.DataFrame([{"metric": "n_trades", "value": len(R)}, {"metric": "total_short_pnl", "value": round(total_short_pnl, 1)},
                      {"metric": "avg_short_pnl", "value": round(avg_short_pnl, 1)}, {"metric": "short_win_rate_pct", "value": round(win_rate, 1)},
                      {"metric": "avg_mae_pct", "value": round(avg_mae, 3)}, {"metric": "median_mae_pct", "value": round(R['short_mae_pct'].median(), 3)},
                      {"metric": "corr_longret_vs_mae_r", "value": round(r_mae, 3)}, {"metric": "corr_longret_vs_mae_p", "value": round(p_mae, 3)},
                      {"metric": "corr_longret_vs_shortpnl_r", "value": round(r_pnl, 3)}, {"metric": "corr_longret_vs_shortpnl_p", "value": round(p_pnl, 3)}]
                     ).to_excel(w, sheet_name="Summary", index=False)
        mae_dist.reset_index().to_excel(w, sheet_name="MAE_Distribution", index=False)
        if missing_data:
            pd.DataFrame(missing_data, columns=["symbol", "date", "issue"]).to_excel(w, sheet_name="Missing_Flagged", index=False)
        excluded_edge[["symbol", "entry_date", "exit_date", "long_exit_type", "exit_time", "gross_ret"]].to_excel(w, sheet_name="Excluded_Edge_Case", index=False)

    pd.set_option("display.width", 220)
    print("\n--- sample trade rows ---")
    print(R.head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/long8pct_short_dd_diagnostic.xlsx")


if __name__ == "__main__":
    main()
