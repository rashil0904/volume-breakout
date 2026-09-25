# -*- coding: utf-8 -*-
"""
nifty_divergence_backtest_1min.py — trade the FILTERED Nifty RSI hourly divergences
(exact-TV replication + 30/70 filter) with 1% stop / 2% target (2:1 R:R), exits checked
on 1-MINUTE Nifty-spot candles (precise intra-hour). Bullish -> LONG, bearish -> SHORT.

FLAGS
 (a) ENTRY at the CLOSE of the confirmation hourly bar T (= that bucket's hourly close, the value
     the RSI is computed on). NOT the back-dated pivot at T-5. entry_price pulled from the hourly
     RSI series (nifty_hourly_rsi.csv) so it is exactly the signal-series close. No look-ahead.
 (b) EXIT on 1-MIN candles from the FIRST 1-min candle at/after the confirmation bar's hourly close
     (bucket end time). First touch wins, filled at the EXACT stop/target level. If a single 1-min
     candle spans BOTH stop and target -> STOP first (conservative); counted. Gap-through the level
     (candle OPEN already beyond the hit level) counted separately (exact-fill is optimistic there).
 (c) TIME EXIT: default (b) HOLD-TILL-HIT across days; a trade unresolved at data-end exits at last
     1-min close. A sensitivity line reports the (a) same-day-close square-off alternative.
 (d) OVERLAP: one-position-at-a-time — a signal arriving while a trade is open is skipped.
 (e) COST 0.03% round-trip (index-futures ballpark 0.02-0.05%); net = gross - cost.
 (f) Nifty-50 SPOT for both signal and 1-min exits; live trading would be futures/options.
 (g) 1-min exit series = NSE_INDEX|Nifty 50 = SAME instrument the hourly RSI is built on (spot/spot).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

M1 = rb.BASE / "data" / "nifty_1min_ohlc.csv"
HOURLY = rb.RESULTS / "nifty_hourly_rsi" / "nifty_hourly_rsi.csv"
DIVS = rb.RESULTS / "nifty_rsi_divergence_tv" / "divergences_filtered.csv"
OUTDIR = rb.RESULTS / "nifty_divergence_backtest_1min"
IST = "Asia/Kolkata"
SL, TGT = 0.01, 0.02
COST = 0.03
BE_WINRATE = 100.0 / 3.0
BUCKET_END = {"09:07-10": 600, "10-11": 660, "11-12": 720, "12-13": 780,
              "13-14": 840, "14-15": 900, "15-15:30": 930}   # minute-of-day the hourly bar closes


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── 1-min exit series (regular session only; spot, same instrument as RSI) ──
    m = pd.read_csv(M1)
    mts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(IST)
    hm = mts.dt.hour * 60 + mts.dt.minute
    keep = (hm >= 555) & (hm < 930)
    m = m[keep].copy(); mts = mts[keep]
    order = np.argsort(mts.values.astype("datetime64[ns]"))
    m = m.iloc[order].reset_index(drop=True); mts = mts.iloc[order].reset_index(drop=True)
    lo = m["low"].values.astype(float); hi = m["high"].values.astype(float)
    op = m["open"].values.astype(float); cl = m["close"].values.astype(float)
    ts_ns = mts.values.astype("datetime64[ns]").astype("int64")   # UTC ns, sorted
    Mn = len(m)

    # ── entry price = hourly close of the confirmation bar (signal series) ──
    H = pd.read_csv(HOURLY, parse_dates=["date"]); H["date"] = H["date"].dt.date
    close_of = {(r.date, r.hour_bucket): float(r.close) for r in H.itertuples()}

    # ── filtered divergences ──
    D = pd.read_csv(DIVS, parse_dates=["confirm_date"])
    D["confirm_date"] = D["confirm_date"].dt.date

    def start_ns(d, bucket):
        e = BUCKET_END[bucket]
        t = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=e // 60, minute=e % 60, tz=IST)
        return t.value

    rows = []
    for r in D.itertuples():
        key = (r.confirm_date, r.confirm_bucket)
        if key not in close_of:
            continue
        sns = start_ns(r.confirm_date, r.confirm_bucket)
        gi = int(np.searchsorted(ts_ns, sns, "left"))      # first 1-min candle at/after hourly close
        if gi >= Mn:
            continue
        rows.append({"type": r.type, "confirm_date": r.confirm_date, "confirm_bucket": r.confirm_bucket,
                     "entry_gi": gi, "entry_price": close_of[key]})
    D2 = pd.DataFrame(rows).sort_values("entry_gi").reset_index(drop=True)

    def do_exit(gi, entry, direction, stop_gi=None):
        stop = entry * (1 - SL) if direction == "long" else entry * (1 + SL)
        tgt = entry * (1 + TGT) if direction == "long" else entry * (1 - TGT)
        end = Mn if stop_gi is None else min(stop_gi + 1, Mn)
        for j in range(gi, end):
            hs = (lo[j] <= stop) if direction == "long" else (hi[j] >= stop)
            ht = (hi[j] >= tgt) if direction == "long" else (lo[j] <= tgt)
            if hs and ht:
                gap = (op[j] <= stop) if direction == "long" else (op[j] >= stop)
                return stop, "stop", j, True, bool(gap)
            if hs:
                gap = (op[j] < stop) if direction == "long" else (op[j] > stop)
                return stop, "stop", j, False, bool(gap)
            if ht:
                gap = (op[j] > tgt) if direction == "long" else (op[j] < tgt)
                return tgt, "target", j, False, bool(gap)
        # unresolved: time exit at last available close in scan window
        jx = end - 1
        return cl[jx], "time", jx, False, False

    def ret_of(entry, xp, direction):
        return (xp - entry) / entry * 100 if direction == "long" else (entry - xp) / entry * 100

    # same-day-close index for flag (a): last 1-min candle on the confirmation date
    day_last = {}
    dser = mts.dt.date.values
    for d in np.unique(dser):
        idx = np.nonzero(dser == d)[0]
        day_last[d] = int(idx[-1])

    def simulate(concurrent):
        """concurrent=False -> one-position-at-a-time (skip a signal while a trade is open).
        concurrent=True -> trade EVERY divergence independently (overlapping/reverse allowed)."""
        trades = []; last_exit = -1; n_skip = 0
        for r in D2.itertuples():
            gi = int(r.entry_gi)
            if (not concurrent) and gi <= last_exit:
                n_skip += 1; continue
            direction = "long" if r.type == "bullish" else "short"
            entry = float(r.entry_price)
            xp, xt, xj, amb, gap = do_exit(gi, entry, direction)
            gret = ret_of(entry, xp, direction)
            rfill = float(op[xj]) if (gap and xt in ("stop", "target")) else xp   # gap-through -> fill at open
            rret = ret_of(entry, rfill, direction)
            dl = day_last.get(r.confirm_date, gi)                                 # flag(a) same-day square-off
            axp, _, _, _, _ = do_exit(gi, entry, direction, stop_gi=max(dl, gi))
            trades.append({
                "type": r.type, "direction": direction,
                "confirm_date": str(r.confirm_date), "confirm_bucket": r.confirm_bucket,
                "entry_gi": gi, "exit_gi": xj,
                "entry_time": pd.Timestamp(ts_ns[gi]).tz_localize("UTC").tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                "entry_price": round(entry, 2),
                "stop": round(entry * (1 - SL) if direction == "long" else entry * (1 + SL), 2),
                "target": round(entry * (1 + TGT) if direction == "long" else entry * (1 - TGT), 2),
                "exit_price": round(xp, 2), "exit_type": xt,
                "exit_time": pd.Timestamp(ts_ns[xj]).tz_localize("UTC").tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                "hold_1min_bars": int(xj - gi + 1), "hold_hours_wall": round((ts_ns[xj] - ts_ns[gi]) / 3.6e12, 2),
                "same1min_ambiguity": amb, "gap_through_level": gap,
                "return_pct": round(gret, 4), "net_return_pct": round(gret - COST, 4),
                "net_return_realistic_pct": round(rret - COST, 4),
                "net_return_flagA_pct": round(ret_of(entry, axp, direction) - COST, 4)})
            last_exit = max(last_exit, xj)
        return pd.DataFrame(trades), n_skip

    T_seq, n_skip = simulate(False)          # one-position-at-a-time (flag d default)
    T, _ = simulate(True)                    # PRIMARY now: trade overlapping signals too
    alt_a_total = T.net_return_flagA_pct.sum()

    # max simultaneous open positions (capital planning under concurrency)
    ev = [(int(r.entry_gi), +1) for r in T.itertuples()] + [(int(r.exit_gi), -1) for r in T.itertuples()]
    ev.sort(key=lambda x: (x[0], x[1]))      # exits (-1) before entries (+1) at equal index
    cur = mx = 0
    for _, delta in ev:
        cur += delta; mx = max(mx, cur)
    max_concurrent = mx

    def block(df, label):
        n = len(df)
        if n == 0:
            return {"segment": label, "n_trades": 0}
        wins = int((df.exit_type == "target").sum()); stops = int((df.exit_type == "stop").sum())
        tex = int((df.exit_type == "time").sum()); wr = wins / n * 100
        eq = df.sort_values("exit_gi").net_return_pct.cumsum().values   # realized order (exit time)
        dd = float((eq - np.maximum.accumulate(eq)).min())
        return {"segment": label, "n_trades": n, "n_target": wins, "n_stop": stops, "n_time_exit": tex,
                "win_rate_pct": round(wr, 1), "breakeven_wr_pct": round(BE_WINRATE, 1),
                "clears_breakeven_net": bool(wr > BE_WINRATE),
                "avg_return_gross_pct": round(df.return_pct.mean(), 4),
                "avg_return_net_pct": round(df.net_return_pct.mean(), 4),
                "total_return_gross_pct": round(df.return_pct.sum(), 2),
                "total_return_net_pct": round(df.net_return_pct.sum(), 2),
                "gross_expectancy_formula_pct": round(wr / 100 * 2 - (1 - wr / 100) * 1, 4),
                "avg_hold_1min_bars": round(df.hold_1min_bars.mean(), 0),
                "avg_hold_hours_wall": round(df.hold_hours_wall.mean(), 1),
                "max_drawdown_net_pct": round(dd, 2),
                "n_same1min_ambiguity": int(df.same1min_ambiguity.sum()),
                "n_gap_through_level": int(df.gap_through_level.sum())}

    summary = pd.DataFrame([block(T[T.type == "bullish"], "bullish"),
                            block(T[T.type == "bearish"], "bearish"),
                            block(T, "combined")])
    summary_seq = pd.DataFrame([block(T_seq[T_seq.type == "bullish"], "bullish"),
                                block(T_seq[T_seq.type == "bearish"], "bearish"),
                                block(T_seq, "combined")])
    T["equity_net_cum"] = T.net_return_pct.cumsum().round(3)

    def streaks(x):
        bw = bl = cw = cl_ = 0
        for v in x:
            if v > 0:
                cw += 1; cl_ = 0
            else:
                cl_ += 1; cw = 0
            bw = max(bw, cw); bl = max(bl, cl_)
        return bw, bl
    bw, bls = streaks(T.net_return_pct.values)

    with pd.ExcelWriter(OUTDIR / "nifty_divergence_backtest_1min.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary_concurrent", index=False)
        summary_seq.to_excel(w, sheet_name="summary_one_at_a_time", index=False)
        T.to_excel(w, sheet_name="all_trades_concurrent", index=False)
        T_seq.to_excel(w, sheet_name="all_trades_one_at_a_time", index=False)
    T.to_csv(OUTDIR / "all_trades_concurrent.csv", index=False)

    pd.set_option("display.width", 260)
    print("=" * 104)
    print("NIFTY RSI-DIVERGENCE BACKTEST — 1-MIN EXITS — TRADE OVERLAPPING SIGNALS (concurrent, flag d off)")
    print("=" * 104)
    print(f"1-min candles: {Mn:,} | filtered signals: {len(D2)} | concurrent taken: {len(T)} "
          f"(vs one-at-a-time {len(T_seq)}; the {n_skip} previously-skipped overlaps are now traded)")
    print(f"max simultaneous OPEN positions under concurrency: {max_concurrent}  "
          f"(=> capital must cover {max_concurrent} legs at once)")
    print("\n--- SUMMARY  [CONCURRENT — all overlapping signals traded]  (breakeven 2:1 = 33.3%) ---")
    cols = ["segment", "n_trades", "n_target", "n_stop", "n_time_exit", "win_rate_pct", "clears_breakeven_net",
            "avg_return_net_pct", "total_return_net_pct", "total_return_gross_pct",
            "avg_hold_hours_wall", "max_drawdown_net_pct", "n_same1min_ambiguity", "n_gap_through_level"]
    print(summary[cols].to_string(index=False))
    print("\n--- COMPARISON: one-at-a-time (flag d default) vs concurrent (overlaps traded) ---")
    cmp = pd.DataFrame([
        {"mode": "one_at_a_time", **{k: summary_seq[summary_seq.segment == "combined"][k].values[0]
                                     for k in ["n_trades", "win_rate_pct", "total_return_net_pct",
                                               "avg_return_net_pct", "max_drawdown_net_pct"]}},
        {"mode": "concurrent", **{k: summary[summary.segment == "combined"][k].values[0]
                                  for k in ["n_trades", "win_rate_pct", "total_return_net_pct",
                                            "avg_return_net_pct", "max_drawdown_net_pct"]}}])
    print(cmp.to_string(index=False))
    print(f"\n  combined net win/loss streaks (concurrent, entry order): max win {bw}, max loss {bls}")
    print(f"  flag(a) sensitivity — same-day-close square-off, combined total net: {alt_a_total:.2f}% "
          f"(vs hold-till-hit {T.net_return_pct.sum():.2f}%)")
    print(f"  gap-through realism  — if the {int(T.gap_through_level.sum())} gapped exits fill at the 1-min OPEN "
          f"instead of the exact level: combined total net {T.net_return_realistic_pct.sum():.2f}% "
          f"(bull {T[T.type=='bullish'].net_return_realistic_pct.sum():.2f}%, "
          f"bear {T[T.type=='bearish'].net_return_realistic_pct.sum():.2f}%)")
    print("\n--- first 10 trades ---")
    print(T[["type", "entry_time", "entry_price", "stop", "target", "exit_type", "exit_time",
             "hold_hours_wall", "net_return_pct", "equity_net_cum"]].head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
