# -*- coding: utf-8 -*-
"""pead_mfe.py — favorable excursion (MFE) over the ACTUAL SL-aware holding window for finalized PEAD
trades (VWAP-ref reaction, K=4, LONG T+10/SHORT T+3, TRIGGER SL, during->next-day). LONG mfe=max HIGH,
SHORT mfe=min LOW, from T+1 to the real exit day. Reports avg/median mfe_pct & mfe_day, mfe-vs-realized
gap, mfe_day histogram, split by exit_reason.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_mfe"
K, N_LONG, N_SHORT, COST = 4.0, 10, 3, 0.20


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})

    rows = []
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, op, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        p = pn if r.announcement_session in ("post_market", "during_market") else (ps if ps is not None else pn)
        if p is None or p < 1:
            continue
        entry = cl[p]; prev_vwap = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and prev_vwap == prev_vwap and prev_vwap > 0):
            continue
        rr = (entry - prev_vwap) / prev_vwap * 100.0
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K:
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):
            continue
        # trigger-SL exit -> actual exit day
        exit_price, reason, days = cl[p + n], "holding_end", n
        for j in range(1, n + 1):
            k = p + j
            if direction == "long" and lo[k] <= sl:
                exit_price = min(sl, op[k]); reason = "gap_through" if op[k] < sl else "sl_trigger"; days = j; break
            if direction == "short" and hi[k] >= sl:
                exit_price = max(sl, op[k]); reason = "gap_through" if op[k] > sl else "sl_trigger"; days = j; break
        realized = (exit_price - entry) / entry * 100 if direction == "long" else (entry - exit_price) / entry * 100
        # MFE over T+1..T+days (SL-aware)
        if direction == "long":
            seg = hi[p + 1:p + 1 + days]; mfe = (seg.max() - entry) / entry * 100; mfe_day = int(seg.argmax()) + 1
        else:
            seg = lo[p + 1:p + 1 + days]; mfe = (entry - seg.min()) / entry * 100; mfe_day = int(seg.argmin()) + 1
        rows.append({"symbol": r.symbol, "quarter": r.quarter, "T0_date": str(dates[p]), "direction": direction,
                     "reaction_return_pct": round(rr, 2), "entry_price": round(entry, 2), "sl_price": round(sl, 2),
                     "holding_target": n, "exit_date": str(dates[p + days]), "exit_price": round(exit_price, 2),
                     "exit_reason": reason, "days_held": days, "realized_return_pct": round(realized, 3),
                     "mfe_pct": round(mfe, 3), "mfe_day": mfe_day, "mfe_minus_realized": round(mfe - realized, 3),
                     "went_green": bool(mfe > 0), "year": int(str(dates[p])[:4])})
    T = pd.DataFrame(rows).sort_values("T0_date").reset_index(drop=True)
    print(f"trades {len(T)} | long {int((T.direction=='long').sum())} short {int((T.direction=='short').sum())}")

    def agg(df, lbl):
        return {"segment": lbl, "n_trades": len(df), "avg_mfe_pct": round(df.mfe_pct.mean(), 3), "median_mfe_pct": round(df.mfe_pct.median(), 3),
                "avg_mfe_day": round(df.mfe_day.mean(), 2), "median_mfe_day": int(df.mfe_day.median()),
                "avg_realized_pct": round(df.realized_return_pct.mean(), 3), "median_realized_pct": round(df.realized_return_pct.median(), 3),
                "avg_mfe_minus_realized": round(df.mfe_minus_realized.mean(), 3), "pct_went_green": round(df.went_green.mean() * 100, 1),
                "avg_days_held": round(df.days_held.mean(), 1)}
    L, S = T[T.direction == "long"], T[T.direction == "short"]
    SUM = pd.DataFrame([agg(L, "LONG (T+10)"), agg(S, "SHORT (T+3)")])
    # split by exit_reason
    er = []
    for dr, sub in [("long", L), ("short", S)]:
        for rn, g in sub.groupby(sub["exit_reason"].map(lambda x: "sl_stop" if x in ("sl_trigger", "gap_through") else "holding_end")):
            er.append({"direction": dr, "exit_group": rn, "n": len(g), "avg_mfe_pct": round(g.mfe_pct.mean(), 3),
                       "avg_mfe_day": round(g.mfe_day.mean(), 2), "avg_realized_pct": round(g.realized_return_pct.mean(), 3),
                       "pct_went_green": round(g.went_green.mean() * 100, 1)})
    ER = pd.DataFrame(er)
    # mfe_day distribution
    dist = []
    for dr, sub, nn in [("long", L, N_LONG), ("short", S, N_SHORT)]:
        vc = sub["mfe_day"].value_counts().reindex(range(1, nn + 1), fill_value=0)
        for day, c in vc.items():
            dist.append({"direction": dr, "mfe_day": day, "n": int(c), "pct": round(c / len(sub) * 100, 1)})
    DIST = pd.DataFrame(dist)

    with pd.ExcelWriter(OUTDIR / "pead_mfe.xlsx", engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="summary", index=False)
        ER.to_excel(w, sheet_name="by_exit_reason", index=False)
        DIST.to_excel(w, sheet_name="mfe_day_distribution", index=False)
        T.to_excel(w, sheet_name="per_trade", index=False)
    T.to_csv(OUTDIR / "pead_mfe_per_trade.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (dr, sub, nn) in enumerate([("long", L, N_LONG), ("short", S, N_SHORT)]):
        ax[i].hist(sub["mfe_day"], bins=range(1, nn + 2), align="left", color="#4472C4", edgecolor="white")
        ax[i].axvline(sub["mfe_day"].mean(), color="red", ls="--", label=f"avg T+{sub['mfe_day'].mean():.1f}")
        ax[i].set_title(f"{dr.upper()} — day of favorable peak (mfe_day)"); ax[i].set_xlabel("T+n"); ax[i].set_ylabel("n trades"); ax[i].legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "mfe_day_hist.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 92 + "\nPEAD FAVORABLE EXCURSION (MFE) — SL-aware holding window (T+1..exit)\n" + "=" * 92)
    print(SUM.to_string(index=False))
    print("\n--- BY EXIT GROUP ---"); print(ER.to_string(index=False))
    print("\n--- MFE_DAY DISTRIBUTION ---")
    print(DIST.pivot(index="mfe_day", columns="direction", values="n").fillna(0).astype(int).to_string())
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
