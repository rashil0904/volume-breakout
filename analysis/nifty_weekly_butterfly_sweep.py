# -*- coding: utf-8 -*-
"""nifty_weekly_butterfly_sweep.py — PROFIT-TARGET sweep on the NIFTY Weekly 2-Day Breakout Butterfly.
Base strategy unchanged (entries/ATM/limits/close-based entry spot); ONLY the max-profit target % is swept:
50,55,...,90%. Each butterfly's value series is computed ONCE, then every threshold is applied to it.
Reports per-threshold trades/win/P&L/%target-hit/avg-time/P&L-by-exit, PUT vs CALL, and IS/OOS. GROSS points.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_butterfly_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; FB_MOD = 14 * 60 + 30; SQ = 15 * 60 + 15
THRESH = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]; OOS_SPLIT = pd.Timestamp("2025-08-29")


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def fly_series(folder, atm, direction, entry_time, e_cur):
    ot = "PE" if direction == "PUT" else "CE"; sgn = -1 if direction == "PUT" else 1
    Kw1, Kb, Kw2 = atm, atm + sgn * WIDTH, atm + sgn * 2 * WIDTH; t1 = e_cur + pd.Timedelta(hours=15, minutes=16)
    w1 = leg_series(folder, Kw1, ot, entry_time, t1); bd = leg_series(folder, Kb, ot, entry_time, t1); w2 = leg_series(folder, Kw2, ot, entry_time, t1)
    if w1 is None or bd is None or w2 is None: return None
    e1, eb, e2 = w1.asof(entry_time), bd.asof(entry_time), w2.asof(entry_time)
    if any(np.isnan(v) for v in (e1, eb, e2)): return None
    net_debit = float(e1 + e2 - 2 * eb)
    if net_debit <= 0 or net_debit >= WIDTH: return None
    both = pd.concat([w1.rename("w1"), bd.rename("bd"), w2.rename("w2")], axis=1).ffill().dropna()
    both = both[both.index > entry_time]
    if both.empty: return None
    fv = (both["w1"] + both["w2"] - 2 * both["bd"]).values; bt = both.index.values
    sq = both[both.index <= e_cur + pd.Timedelta(hours=15, minutes=15)]
    if sq.empty: return None
    exit1515 = float(sq["w1"].iloc[-1] + sq["w2"].iloc[-1] - 2 * sq["bd"].iloc[-1])
    return {"net_debit": net_debit, "max_profit": WIDTH - net_debit, "fv": fv, "bt": bt, "exit1515": exit1515}


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(h=("high", "max"), l=("low", "min"))
    daily["p2h"] = daily["h"].shift(1).rolling(2).max(); daily["p2l"] = daily["l"].shift(1).rolling(2).min()
    tdays = list(daily.index); expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    flies = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        win_days = [d for d in tdays if e_prev < d <= e_cur]
        if not win_days or win_days[0] > spot_end or e_cur > spot_end: continue
        folder = e_cur.strftime("%Y%m%d"); put_e = None; call_e = None; fb_done = False
        for di, D in enumerate(win_days):
            p2h = daily.loc[D, "p2h"]; p2l = daily.loc[D, "p2l"]
            if np.isnan(p2h): continue
            ed = sp[sp["date"] == D]; hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values; d_open = ed["open"].iloc[0]
            for k in range(len(ed)):
                if di == 0 and md[k] >= FB_MOD and not fb_done and put_e is None and call_e is None:
                    if cl[k] < d_open: put_e = (pd.Timestamp(ets[k]), cl[k], "fallback-2:30")
                    else: call_e = (pd.Timestamp(ets[k]), cl[k], "fallback-2:30")
                    fb_done = True
                if put_e is None and lo[k] <= p2l: put_e = (pd.Timestamp(ets[k]), cl[k], "2d-low breach")
                if call_e is None and hi[k] >= p2h: call_e = (pd.Timestamp(ets[k]), cl[k], "2d-high breach")
            if put_e is not None and call_e is not None: break
        for direction, ent in [("PUT", put_e), ("CALL", call_e)]:
            if ent is None: continue
            et, espot, trig = ent; atm = round(espot / 50) * 50
            fs = fly_series(folder, atm, direction, et, e_cur)
            if fs is None: continue
            fs.update({"direction": direction, "entry_date": win_days[0], "entry_time": et, "expiry": e_cur.date()})
            flies.append(fs)

    def exit_at(f, T):
        live = f["fv"] - f["net_debit"]; hit = np.where(live >= T * f["max_profit"])[0]
        if len(hit):
            j = hit[0]; return f["fv"][j] - f["net_debit"], "target", (pd.Timestamp(f["bt"][j]) - f["entry_time"]).total_seconds() / 86400
        return f["exit1515"] - f["net_debit"], "expiry", np.nan

    def agg(sub, T):
        res = [exit_at(f, T) for f in sub]; pnl = np.array([r[0] for r in res]); typ = [r[1] for r in res]; tm = [r[2] for r in res]
        thit = [i for i, t in enumerate(typ) if t == "target"]; texp = [i for i, t in enumerate(typ) if t == "expiry"]
        return {"target_%": int(T * 100), "trades": len(sub), "win_%": round((pnl > 0).mean() * 100, 1) if len(sub) else 0,
                "total_pnl": round(pnl.sum(), 1), "avg_pnl": round(pnl.mean(), 2) if len(sub) else 0,
                "pct_target_hit": round(len(thit) / len(sub) * 100, 1) if len(sub) else 0,
                "avg_days_to_target": round(np.nanmean([tm[i] for i in thit]), 2) if thit else np.nan,
                "avg_pnl_target": round(pnl[thit].mean(), 2) if thit else np.nan,
                "avg_pnl_expiry": round(pnl[texp].mean(), 2) if texp else np.nan}

    groups = {"ALL": flies, "PUT": [f for f in flies if f["direction"] == "PUT"], "CALL": [f for f in flies if f["direction"] == "CALL"],
              "IS": [f for f in flies if f["entry_date"] < OOS_SPLIT], "OOS": [f for f in flies if f["entry_date"] >= OOS_SPLIT]}
    tables = {g: pd.DataFrame([agg(sub, T) for T in THRESH]) for g, sub in groups.items()}

    v = tables["ALL"]["total_pnl"].values; nb = np.array([np.mean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
    bi = int(np.argmax(nb)); center = int(THRESH[bi] * 100); region = [int(THRESH[j] * 100) for j in range(max(0, bi - 1), min(len(THRESH), bi + 2))]
    best_single = int(THRESH[int(np.argmax(v))] * 100)

    with pd.ExcelWriter(OUTDIR / "butterfly_target_sweep.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "NIFTY Weekly 2-Day Breakout Butterfly — PROFIT-TARGET sweep (only target % varies); GROSS points"},
            {"metric": "Thresholds", "value": str([int(t * 100) for t in THRESH])}, {"metric": "Flies", "value": len(flies)},
            {"metric": "OOS split", "value": str(OOS_SPLIT.date())},
            {"metric": "Best single target", "value": f"{best_single}% = {v[int(np.argmax(v))]} pts"},
            {"metric": "BEST STABLE REGION (3-neighbour)", "value": f"center {center}% | region {region}% | pts {[tables['ALL'].total_pnl.iloc[j] for j in range(max(0,bi-1),min(len(THRESH),bi+2))]}"},
            {"metric": "Note", "value": "entry/ATM/limits unchanged; expiry 15:15 catch-all applies at every threshold"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        for g, tb in tables.items(): tb.to_excel(w, sheet_name=f"Sweep_{g}", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for g, col in [("ALL", "#1F6FB2"), ("PUT", "#2E8B57"), ("CALL", "#C0392B")]:
        ax.plot([int(t * 100) for t in THRESH], tables[g]["total_pnl"], "-o", ms=4, label=g, color=col)
    ax.axvspan(min(region), max(region), color="#999", alpha=.12, label=f"best region ~{center}%")
    ax.set_title("NIFTY Weekly Butterfly — profit-target sweep (total gross P&L)"); ax.set_xlabel("Target % of max profit"); ax.set_ylabel("Total P&L (points)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "target_sweep_curve.png", dpi=130); plt.close(fig)
    tables["ALL"].to_csv(OUTDIR / "butterfly_target_sweep_all.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nBUTTERFLY PROFIT-TARGET SWEEP | flies " + str(len(flies)) + " | OOS split " + str(OOS_SPLIT.date()) + "\n" + "=" * 96)
    print("\n--- ALL ---"); print(tables["ALL"].to_string(index=False))
    print("\n--- PUT ---"); print(tables["PUT"][["target_%", "trades", "win_%", "total_pnl", "pct_target_hit", "avg_days_to_target"]].to_string(index=False))
    print("\n--- CALL ---"); print(tables["CALL"][["target_%", "trades", "win_%", "total_pnl", "pct_target_hit", "avg_days_to_target"]].to_string(index=False))
    print("\n--- IS vs OOS (total_pnl) ---")
    m = tables["IS"][["target_%", "total_pnl"]].merge(tables["OOS"][["target_%", "total_pnl"]], on="target_%", suffixes=("_IS", "_OOS")); print(m.to_string(index=False))
    print(f"\nbest single {best_single}% | BEST REGION center {center}% region {region}%")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
