# -*- coding: utf-8 -*-
"""ipo_1week_breakout_improve_returns.py — tests concrete levers to improve the "IPO 1 Week Breakout" strategy,
motivated directly by what the last two runs flagged:
  (1) ipo_1week_breakout_backtest.py: the plain (no-SL) average return was almost entirely a tail-outlier effect
      (a handful of sub-Rs20 penny stocks multiplying 5-15x) -- median stayed flat/negative throughout.
  (2) ipo_1week_breakout_backtest_sl.py: a stop at the exact breakout level was too tight (47% of trades stopped
      out on the breakout day itself) and made every metric worse.
LEVERS TESTED (each isolated first, then the best combined):
  A) PRICE FLOOR -- drop breakout entries below a minimum price (removes penny-stock noise/outlier risk).
  B) LOOSER STOP-LOSS -- a % buffer below the breakout level instead of the exact level (close < ref_high*(1-buf)).
  C) VOLUME-CONFIRMED BREAKOUT -- require the breakout day's volume to be a multiple of the 5-day reference
     window's average daily volume (a weak, low-conviction breakout is more likely to fail).
Same universe/reference-window/touch-based-entry/gap-fill logic as the original script, reused unmodified where
possible. New file; both prior scripts untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUT = rb.RESULTS / "ipo_1week_breakout_improve_returns"; OUT.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"
REF_DAYS = 5
MONTHS = 3
CAPS = [5, 10, 15, 20]
FULL_CAPS = list(range(1, 21))


def daily_ohlcv(sym):
    fn = rb.MASTER_DIR / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.assign(date=ts.dt.date).sort_values("timestamp")
    g = df.groupby("date")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                       "close": g["close"].last(), "volume": g["volume"].sum()})
    return d.sort_index()


def precompute(sym, listing_date):
    D = daily_ohlcv(sym)
    if D is None or len(D) < REF_DAYS + 1:
        return None
    tdays = list(D.index)
    hi = D["high"].values; op = D["open"].values; cl = D["close"].values; vol = D["volume"].values
    ref_hi = float(hi[:REF_DAYS].max()); ref_avg_vol = float(vol[:REF_DAYS].mean())
    end3 = (pd.Timestamp(listing_date) + pd.DateOffset(months=MONTHS)).date()
    f = None
    for j in range(REF_DAYS, len(tdays)):
        if tdays[j] >= end3:
            break
        if hi[j] > ref_hi + 1e-9:
            f = j; break
    if f is None:
        return None
    entry_px = ref_hi if op[f] <= ref_hi + 1e-9 else float(op[f])
    return {"symbol": sym, "listing_date": listing_date, "ref_hi": ref_hi, "ref_avg_vol": ref_avg_vol,
            "f": f, "entry_px": entry_px, "breakout_date": tdays[f], "breakout_vol": float(vol[f]),
            "tdays": tdays, "cl": cl}


def build_universe():
    U = pd.read_csv(rb.RESULTS / "newly_listed_universe_combined.csv", parse_dates=["first_date"])
    U["first_date"] = U["first_date"].dt.date
    P = []
    for i, r in enumerate(U.itertuples(), 1):
        p = precompute(r.symbol, r.first_date)
        if p: P.append(p)
        if i % 200 == 0: print(f"  precompute ...{i}/{len(U)}", flush=True)
    print(f"universe: {len(U)} | qualifying breakouts precomputed: {len(P)}", flush=True)
    return P


def trades_for(P, price_floor=0.0, sl_buffer=None, vol_mult=None, cap=20):
    """sl_buffer: None = no SL; 0.0 = exact level (tight); e.g. 0.10 = close < ref_hi*0.90.
       vol_mult: None = no filter; e.g. 1.5 = breakout day volume >= 1.5x the 5-day reference average volume."""
    rows = []
    for p in P:
        if p["entry_px"] < price_floor:
            continue
        if vol_mult is not None and not (p["ref_avg_vol"] > 0 and p["breakout_vol"] >= vol_mult * p["ref_avg_vol"]):
            continue
        f, entry_px, tdays, cl = p["f"], p["entry_px"], p["tdays"], p["cl"]
        cap_idx = f + cap
        xi = None
        if sl_buffer is not None:
            stop_level = p["ref_hi"] * (1 - sl_buffer)
            for k in range(f, min(cap_idx, len(tdays) - 1) + 1):
                if cl[k] < stop_level - 1e-9:
                    xi = k; break
        if xi is None:
            if cap_idx < len(tdays):
                xi = cap_idx
            else:
                continue
        ret = (float(cl[xi]) - entry_px) / entry_px * 100.0
        rows.append({"symbol": p["symbol"], "breakout_date": p["breakout_date"], "entry_price": round(entry_px, 3),
                     "exit_date": tdays[xi], "actual_days_held": xi - f, "return_pct": round(ret, 3)})
    return pd.DataFrame(rows)


def stats(T):
    if T.empty:
        return {"n": 0}
    x = T["return_pct"]
    return {"n": len(T), "avg_pct": round(x.mean(), 3), "median_pct": round(x.median(), 3), "win_pct": round((x > 0).mean() * 100, 1),
            "p10": round(x.quantile(.1), 2), "p90": round(x.quantile(.9), 2)}


def main():
    P = build_universe()

    print("\n=== BASELINE (no floor, no SL, no volume filter) ===")
    base = {N: stats(trades_for(P, cap=N)) for N in CAPS}
    Bdf = pd.DataFrame(base).T.reset_index().rename(columns={"index": "cap"})
    print(Bdf.to_string(index=False))

    print("\n=== LEVER A: PRICE FLOOR (no SL) ===")
    A_rows = []
    for floor in [0, 10, 20, 50, 100]:
        for N in CAPS:
            A_rows.append({"price_floor": floor, "cap": N, **stats(trades_for(P, price_floor=floor, cap=N))})
    Adf = pd.DataFrame(A_rows)
    print(Adf.pivot(index="price_floor", columns="cap", values="avg_pct").to_string())
    print("median:\n" + Adf.pivot(index="price_floor", columns="cap", values="median_pct").to_string())
    print("win%:\n" + Adf.pivot(index="price_floor", columns="cap", values="win_pct").to_string())
    print("n:\n" + Adf.pivot(index="price_floor", columns="cap", values="n").to_string())

    print("\n=== LEVER B: SL BUFFER (no floor) ===")
    B_rows = []
    for buf, lbl in [(None, "no_SL"), (0.0, "0%_tight"), (0.05, "5%"), (0.10, "10%"), (0.15, "15%"), (0.20, "20%")]:
        for N in CAPS:
            B_rows.append({"sl_buffer": lbl, "cap": N, **stats(trades_for(P, sl_buffer=buf, cap=N))})
    Bufdf = pd.DataFrame(B_rows)
    print(Bufdf.pivot(index="sl_buffer", columns="cap", values="avg_pct").to_string())
    print("median:\n" + Bufdf.pivot(index="sl_buffer", columns="cap", values="median_pct").to_string())
    print("win%:\n" + Bufdf.pivot(index="sl_buffer", columns="cap", values="win_pct").to_string())

    print("\n=== LEVER C: VOLUME-CONFIRMED BREAKOUT (no floor, no SL) ===")
    C_rows = []
    for vm, lbl in [(None, "no_filter"), (1.0, ">=1x_ref_avg"), (1.5, ">=1.5x"), (2.0, ">=2x"), (3.0, ">=3x")]:
        for N in CAPS:
            C_rows.append({"vol_mult": lbl, "cap": N, **stats(trades_for(P, vol_mult=vm, cap=N))})
    Cdf = pd.DataFrame(C_rows)
    print(Cdf.pivot(index="vol_mult", columns="cap", values="avg_pct").to_string())
    print("median:\n" + Cdf.pivot(index="vol_mult", columns="cap", values="median_pct").to_string())
    print("win%:\n" + Cdf.pivot(index="vol_mult", columns="cap", values="win_pct").to_string())
    print("n:\n" + Cdf.pivot(index="vol_mult", columns="cap", values="n").to_string())

    # ---- combined: pick the best-looking single setting from each lever (by median improvement, not just avg) and stack them ----
    COMBO = {"price_floor": 20, "sl_buffer": 0.15, "vol_mult": 1.5}
    print(f"\n=== COMBINED ({COMBO}) vs BASELINE, full T+1..T+20 sweep ===")
    comb_rows, base_rows = [], []
    for N in FULL_CAPS:
        comb_rows.append({"cap": N, **stats(trades_for(P, price_floor=COMBO["price_floor"], sl_buffer=COMBO["sl_buffer"], vol_mult=COMBO["vol_mult"], cap=N))})
        base_rows.append({"cap": N, **stats(trades_for(P, cap=N))})
    Combdf = pd.DataFrame(comb_rows); Basedf = pd.DataFrame(base_rows)
    cmp = Combdf.merge(Basedf, on="cap", suffixes=("_combined", "_baseline"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 30)
    print(cmp[["cap", "n_combined", "avg_pct_combined", "median_pct_combined", "win_pct_combined",
               "n_baseline", "avg_pct_baseline", "median_pct_baseline", "win_pct_baseline"]].to_string(index=False))

    detail = trades_for(P, price_floor=COMBO["price_floor"], sl_buffer=COMBO["sl_buffer"], vol_mult=COMBO["vol_mult"], cap=15).sort_values("return_pct", ascending=False)
    print(f"\ncombined-filter trades at cap=15 (n={len(detail)}):\n" + detail.to_string(index=False), flush=True)

    fn = OUT / "ipo_1week_breakout_improve_returns.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Three levers tested in isolation (price floor, looser SL buffer, volume-confirmed breakout), then combined "
                                "at illustrative settings (price_floor=Rs20, sl_buffer=15%, vol_mult=1.5x). Judge levers on MEDIAN and win rate, "
                                "not just average -- the original no-filter average was shown to be a tail-outlier artifact."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        Bdf.to_excel(w, sheet_name="Baseline", index=False)
        Adf.to_excel(w, sheet_name="Lever_A_price_floor", index=False)
        Bufdf.to_excel(w, sheet_name="Lever_B_SL_buffer", index=False)
        Cdf.to_excel(w, sheet_name="Lever_C_volume_confirm", index=False)
        cmp.to_excel(w, sheet_name="Combined_vs_Baseline", index=False)
        detail.to_excel(w, sheet_name="Combined_trades_cap15", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
