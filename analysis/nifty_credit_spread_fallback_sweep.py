# -*- coding: utf-8 -*-
"""nifty_credit_spread_fallback_sweep.py — sweep the FALLBACK CHECK TIME (14:30..15:29, 1-min, 60 vals) of the
NIFTY Weekly Credit Spread. Breakout entries are UNCHANGED (touch-based, fire whenever 3-day level breached).
At swept time T: if no breach by T -> fallback using candle-T OPEN (red->CCS, green->PCS). As T rises, cycles
whose breach falls in (14:30,T] SWITCH from fallback to breakout -> trade population shifts; reported per T.
Reuses the credit-spread trade mechanics (net credit, 90% target 1-min, DTE-0 settlement). GROSS points.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from nifty_weekly_credit_spread import leg_series

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "credit_spread_fallback_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MODS = list(range(870, 930)); BASE_MOD = 870; OOS_SPLIT = pd.Timestamp("2025-08-29")


def compute_trade(cache, folder, entry_ts, entry_spot, ccs, e_cur, settle_spot, t0, t1):
    atm = round(entry_spot / 50) * 50
    sK, lK, ot = (atm, atm + WIDTH, "CE") if ccs else (atm, atm - WIDTH, "PE")

    def getleg(K):
        key = (K, ot)
        if key not in cache: cache[key] = leg_series(folder, K, ot, t0, t1)
        return cache[key]
    ss = getleg(sK); ls = getleg(lK)
    if ss is None or ls is None: return None
    s0 = ss.asof(entry_ts); l0 = ls.asof(entry_ts)
    if np.isnan(s0) or np.isnan(l0): return None
    nc = float(s0 - l0)
    if nc <= 0: return None
    both = pd.concat([ss.rename("s"), ls.rename("l")], axis=1).ffill().dropna(); both = both[both.index > entry_ts]
    sv = (both["s"] - both["l"]).values; bt = both.index.values; thr = TARGET_FRAC * nc
    hit = np.where(sv <= thr)[0]
    if len(hit): return nc - float(sv[hit[0]]), "target"
    if np.isnan(settle_spot): return None
    ed = float(np.clip(settle_spot - atm, 0, WIDTH)) if ccs else float(np.clip(atm - settle_spot, 0, WIDTH))
    return nc - ed, "settlement"


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean(); tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    cycles = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: continue
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3: continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()
        ed = sp[sp["date"] == entry_day]; hi = ed["high"].values; lo = ed["low"].values; op = ed["open"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
        d_open = ed["open"].iloc[0]; folder = e_cur.strftime("%Y%m%d"); ss = settle.get(e_cur, np.nan)
        t0 = pd.Timestamp(entry_day) + pd.Timedelta(hours=9); t1 = e_cur + pd.Timedelta(hours=15, minutes=35); cache = {}
        # ---- first breach (touch) ----
        fb_mod = None; brk = None
        for k in range(len(ed)):
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                ccs = (bl and (not bh or abs(d_open - d3l) <= abs(d_open - d3h)))     # low-breach=bearish=CCS
                brk = compute_trade(cache, folder, pd.Timestamp(ets[k]), cl[k], ccs, e_cur, ss, t0, t1)
                fb_mod = int(md[k]); brk_lbl = "3d-low(CCS)" if ccs else "3d-high(PCS)"; break
        # ---- fallback per T (only where no breach by T) ----
        opens = {int(m): float(o) for m, o in zip(md, op)}; fb = {}
        for T in FB_MODS:
            if fb_mod is not None and fb_mod <= T: continue                          # breakout already fired by T
            if T not in opens: continue
            px = opens[T]; ccs = px < d_open                                         # red -> CCS
            r = compute_trade(cache, folder, pd.Timestamp(entry_day) + pd.Timedelta(minutes=T), px, ccs, e_cur, ss, t0, t1)
            if r is not None: fb[T] = r[0]
        cycles.append({"entry_day": entry_day, "fb_mod": fb_mod, "brk": (brk[0] if brk else None), "brk_lbl": brk_lbl if brk else None, "fb": fb})

    def agg(T, subset):
        pnl = []; nbrk = 0; nfb = 0; nsw = 0
        for c in subset:
            if c["fb_mod"] is not None and c["fb_mod"] <= T:
                if c["brk"] is None: continue
                pnl.append(c["brk"]); nbrk += 1
                if BASE_MOD < c["fb_mod"] <= T: nsw += 1                              # would be fallback at 14:30, breakout now
            else:
                if T not in c["fb"]: continue
                pnl.append(c["fb"][T]); nfb += 1
        p = np.array(pnl)
        return {"time": f"{T//60}:{T%60:02d}", "trades": len(p), "win_%": round((p > 0).mean() * 100, 1) if len(p) else 0,
                "total_pnl": round(p.sum(), 1), "avg_pnl": round(p.mean(), 2) if len(p) else 0,
                "breakout": nbrk, "fallback": nfb, "switched_from_fb": nsw}

    ISs = [c for c in cycles if c["entry_day"] < OOS_SPLIT]; OOSs = [c for c in cycles if c["entry_day"] >= OOS_SPLIT]
    MAIN = pd.DataFrame([agg(T, cycles) for T in FB_MODS])
    ISd = pd.DataFrame([{"time": f"{T//60}:{T%60:02d}", "IS_pnl": agg(T, ISs)["total_pnl"], "OOS_pnl": agg(T, OOSs)["total_pnl"]} for T in FB_MODS])
    v = MAIN["total_pnl"].values; nb = np.array([np.mean(v[max(0, i - 2):i + 3]) for i in range(len(v))]); bi = int(np.argmax(nb))
    region = [MAIN.time.iloc[j] for j in range(max(0, bi - 2), min(len(v), bi + 3))]; best_single = MAIN.time.iloc[int(np.argmax(v))]
    base_row = MAIN.iloc[0]

    with pd.ExcelWriter(OUTDIR / "fallback_time_sweep.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "Credit Spread — FALLBACK check-time sweep 14:30..15:29 (breakout entries unchanged). GROSS points"},
            {"metric": "Cycles", "value": len(cycles)}, {"metric": "OOS split", "value": str(OOS_SPLIT.date())},
            {"metric": "BASELINE 2:30pm", "value": f"total {base_row.total_pnl} | win {base_row['win_%']}% | trades {base_row.trades} | breakout/fallback {base_row.breakout}/{base_row.fallback}"},
            {"metric": "Best single time", "value": f"{best_single} = {v[int(np.argmax(v))]} pts"},
            {"metric": "BEST STABLE REGION (5-neighbour)", "value": f"center {MAIN.time.iloc[bi]} | region {region}"},
            {"metric": "POPULATION-SHIFT NOTE", "value": f"breakout mix rises {base_row.breakout}->{MAIN.breakout.iloc[-1]} and fallback falls {base_row.fallback}->{MAIN.fallback.iloc[-1]} across 14:30->15:29; NOT apples-to-apples"},
            {"metric": "Switched fb->breakout by 15:29", "value": int(MAIN.switched_from_fb.iloc[-1])},
        ]).to_excel(w, sheet_name="Summary", index=False)
        MAIN.to_excel(w, sheet_name="Sweep", index=False); ISd.to_excel(w, sheet_name="IS_OOS", index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5)); xs = list(range(len(FB_MODS)))
    ax.plot(xs, MAIN.total_pnl, "-o", ms=3, color="#1F6FB2", label="total P&L")
    ax.axhline(base_row.total_pnl, color="#C0392B", ls="--", lw=1.2, label=f"2:30 baseline ({base_row.total_pnl})")
    ax.axvspan(max(0, bi - 2), min(len(v) - 1, bi + 2), color="#2E8B57", alpha=.12, label=f"best region ~{MAIN.time.iloc[bi]}")
    ax2 = ax.twinx(); ax2.plot(xs, MAIN.breakout, color="#888", lw=1, alpha=.6, label="breakout count"); ax2.set_ylabel("breakout trade count", color="#888")
    ax.set_xticks(xs[::5]); ax.set_xticklabels([MAIN.time.iloc[i] for i in xs[::5]], rotation=45, fontsize=8)
    ax.set_title("Credit Spread — fallback check-time sweep (P&L) + breakout-mix shift"); ax.set_xlabel("fallback check time"); ax.set_ylabel("total P&L (pts)"); ax.grid(alpha=.25); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "fallback_sweep_curve.png", dpi=130); plt.close(fig); MAIN.to_csv(OUTDIR / "fallback_time_sweep.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + f"\nCREDIT SPREAD — FALLBACK CHECK-TIME SWEEP (14:30-15:29) | cycles {len(cycles)}\n" + "=" * 96)
    print(f"\nBASELINE 2:30: {base_row.total_pnl} pts | win {base_row['win_%']}% | trades {base_row.trades} | breakout/fallback {base_row.breakout}/{base_row.fallback}")
    print("\n--- sweep (every 5 min) ---"); print(MAIN.iloc[::5][["time", "trades", "win_%", "total_pnl", "breakout", "fallback", "switched_from_fb"]].to_string(index=False))
    print(f"\npopulation shift: breakout {base_row.breakout}->{MAIN.breakout.iloc[-1]} | fallback {base_row.fallback}->{MAIN.fallback.iloc[-1]} | switched by 15:29 {int(MAIN.switched_from_fb.iloc[-1])}")
    print(f"best single {best_single} | BEST REGION center {MAIN.time.iloc[bi]} region {region}")
    print(f"IS/OOS at baseline: {ISd.IS_pnl.iloc[0]}/{ISd.OOS_pnl.iloc[0]} | at best-region center: {ISd.IS_pnl.iloc[bi]}/{ISd.OOS_pnl.iloc[bi]}")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
