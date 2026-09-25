# -*- coding: utf-8 -*-
"""
nifty_divergence_backtest.py — trade the FILTERED Nifty RSI hourly divergences (exact-TV + 30/70)
with fixed 1% stop / 2% target (2:1 R:R). Bullish -> LONG, bearish -> SHORT.

ENTRY (flag a): at the CLOSE of the confirmation bar T (= last 15-min close of that hourly bucket),
  no look-ahead. LONG stop=entry*0.99 target=entry*1.02 ; SHORT stop=entry*1.01 target=entry*0.98.
EXIT (flag b): scanned on 15-MIN candles from the candle AFTER entry (1-min Nifty not available in the
  project -> 15-min is the finest series; this makes same-candle stop+target spans a bit more frequent).
  First-touch wins; if a single candle spans BOTH stop and target -> STOP first (conservative). Counted.
TIME EXIT (flag c): DEFAULT hold-till-hit across days; a trade unresolved at data-end exits at last close.
OVERLAP (flag d): one-position-at-a-time — a new signal while a trade is open is ignored (incl. opposite).
COST (flag e): 0.03% round-trip (index-futures ballpark 0.02-0.05%); net = gross - cost.
SERIES (flag f): Nifty-50 SPOT index points (the series the RSI is on); live trading would be futures.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

N15 = rb.BASE / "data" / "nifty_15min_ohlc.csv"
DIVS = rb.RESULTS / "nifty_rsi_divergence_tv" / "divergences_filtered.csv"
OUTDIR = rb.RESULTS / "nifty_divergence_backtest"
IST = "Asia/Kolkata"
SL, TGT = 0.01, 0.02
COST = 0.03                       # % round-trip
BE_WINRATE = 100.0 / 3.0          # 33.3% breakeven for 2:1 R:R
BUCKETS = {(555, 600): "09:07-10", (600, 660): "10-11", (660, 720): "11-12", (720, 780): "12-13",
           (780, 840): "13-14", (840, 900): "14-15", (900, 930): "15-15:30"}
LBL2IDX = {lbl: i + 1 for i, lbl in enumerate(["09:07-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15-15:30"])}


def bucket_idx(hm):
    for (lo, hi), lbl in BUCKETS.items():
        if lo <= hm < hi:
            return LBL2IDX[lbl]
    return None


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(N15)
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw = raw.assign(dt=ts, date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    raw = raw[(raw["hm"] >= 555) & (raw["hm"] < 930)].sort_values("dt").reset_index(drop=True)
    raw["bidx"] = raw["hm"].map(bucket_idx)
    lo15 = raw["low"].values.astype(float); hi15 = raw["high"].values.astype(float)
    cl15 = raw["close"].values.astype(float); ts15 = raw["dt"]      # tz-aware IST Series
    M = len(raw)
    # (date, bucket idx) -> global 15-min index of the LAST candle in that hourly bucket (= entry candle)
    last_of_bucket = raw.groupby(["date", "bidx"]).apply(lambda g: g.index.max()).to_dict()

    D = pd.read_csv(DIVS, parse_dates=["confirm_date"])
    D["confirm_date"] = D["confirm_date"].dt.date
    D["bidx"] = D["confirm_bucket"].map(LBL2IDX)
    D["entry_gidx"] = [last_of_bucket.get((d, b), -1) for d, b in zip(D["confirm_date"], D["bidx"])]
    D = D[D["entry_gidx"] >= 0].sort_values("entry_gidx").reset_index(drop=True)

    def do_exit(gi, entry, direction):
        stop = entry * (1 - SL) if direction == "long" else entry * (1 + SL)
        tgt = entry * (1 + TGT) if direction == "long" else entry * (1 - TGT)
        for j in range(gi + 1, M):
            hs = (lo15[j] <= stop) if direction == "long" else (hi15[j] >= stop)
            ht = (hi15[j] >= tgt) if direction == "long" else (lo15[j] <= tgt)
            if hs and ht:
                return stop, "stop", j, True                 # same-candle span -> stop first
            if hs:
                return stop, "stop", j, False
            if ht:
                return tgt, "target", j, False
        return cl15[M - 1], "open_end", M - 1, False

    trades = []; last_exit = -1; n_skip = 0
    for _, d in D.iterrows():
        gi = int(d["entry_gidx"])
        if gi <= last_exit:                                  # one-at-a-time: ignore while in a trade
            n_skip += 1; continue
        direction = "long" if d["type"] == "bullish" else "short"
        entry = float(cl15[gi])
        xp, xt, xj, amb = do_exit(gi, entry, direction)
        ret = (xp - entry) / entry * 100 if direction == "long" else (entry - xp) / entry * 100
        trades.append({"type": d["type"], "direction": direction,
                       "confirm_date": d["confirm_date"], "confirm_bucket": d["confirm_bucket"],
                       "entry_time": pd.Timestamp(ts15[gi]).strftime("%Y-%m-%d %H:%M"),
                       "entry_price": round(entry, 2),
                       "stop": round(entry * (1 - SL) if direction == "long" else entry * (1 + SL), 2),
                       "target": round(entry * (1 + TGT) if direction == "long" else entry * (1 - TGT), 2),
                       "exit_price": round(xp, 2), "exit_type": xt,
                       "exit_time": pd.Timestamp(ts15[xj]).strftime("%Y-%m-%d %H:%M"),
                       "bars_held_15min": int(xj - gi), "hours_held": round((xj - gi) / 4.0, 2),
                       "stopfirst_ambiguity": amb, "return_pct": round(ret, 4),
                       "net_return_pct": round(ret - COST, 4)})
        last_exit = xj
    T = pd.DataFrame(trades)

    def block(df, label):
        n = len(df)
        if n == 0:
            return {"segment": label, "n_trades": 0}
        wins = int((df["exit_type"] == "target").sum()); losses = int((df["exit_type"] == "stop").sum())
        opn = int((df["exit_type"] == "open_end").sum()); wr = wins / n * 100
        eq = df["net_return_pct"].cumsum().values
        peak = np.maximum.accumulate(eq); dd = (eq - peak).min()
        return {"segment": label, "n_trades": n, "n_wins": wins, "n_losses": losses, "n_open_end": opn,
                "win_rate_pct": round(wr, 1), "breakeven_wr_pct": round(BE_WINRATE, 1),
                "clears_breakeven_net": bool(wr > BE_WINRATE),
                "avg_return_gross_pct": round(df["return_pct"].mean(), 4),
                "avg_return_net_pct": round(df["net_return_pct"].mean(), 4),
                "total_return_gross_pct": round(df["return_pct"].sum(), 2),
                "total_return_net_pct": round(df["net_return_pct"].sum(), 2),
                "expectancy_net_pct": round(df["net_return_pct"].mean(), 4),
                "gross_expectancy_formula_pct": round(wr / 100 * 2 - (1 - wr / 100) * 1, 4),
                "avg_bars_held_15min": round(df["bars_held_15min"].mean(), 1),
                "avg_hours_held": round(df["hours_held"].mean(), 1),
                "max_drawdown_net_pct": round(float(dd), 2),
                "n_stopfirst_ambiguity": int(df["stopfirst_ambiguity"].sum())}
    summary = pd.DataFrame([block(T[T["type"] == "bullish"], "bullish"),
                            block(T[T["type"] == "bearish"], "bearish"),
                            block(T, "combined")])

    # equity curve + streaks (combined, net)
    Tc = T.sort_values("entry_time").reset_index(drop=True)
    Tc["equity_net_cum"] = Tc["net_return_pct"].cumsum().round(3)
    def streaks(x):
        bw = bl = cw = cl = 0
        for v in x:
            if v > 0:
                cw += 1; cl = 0
            else:
                cl += 1; cw = 0
            bw = max(bw, cw); bl = max(bl, cl)
        return bw, bl
    bw, bls = streaks(Tc["net_return_pct"].values)

    with pd.ExcelWriter(OUTDIR / "nifty_divergence_backtest.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        Tc.to_excel(w, sheet_name="all_trades", index=False)
    Tc.to_csv(OUTDIR / "all_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("=" * 100 + "\nNIFTY RSI-DIVERGENCE BACKTEST (filtered TV signals; 1% SL / 2% target; 15-min exits)\n" + "=" * 100)
    print(f"filtered signals: {len(D)} | taken: {len(T)} | skipped (in-trade overlap): {n_skip}")
    print("\n--- SUMMARY (gross & net; breakeven win-rate for 2:1 = 33.3%) ---")
    cols = ["segment", "n_trades", "n_wins", "n_losses", "n_open_end", "win_rate_pct", "clears_breakeven_net",
            "avg_return_net_pct", "total_return_net_pct", "total_return_gross_pct",
            "avg_hours_held", "max_drawdown_net_pct", "n_stopfirst_ambiguity"]
    print(summary[cols].to_string(index=False))
    print(f"\n  combined net win/loss streaks: max win {bw}, max loss {bls}")
    print("\n--- first 8 trades ---")
    print(Tc[["type", "entry_time", "entry_price", "stop", "target", "exit_type", "exit_time",
              "hours_held", "return_pct", "net_return_pct", "equity_net_cum"]].head(8).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
