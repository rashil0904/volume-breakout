# -*- coding: utf-8 -*-
"""nifty_credit_spread_width_sweep.py — BUY-LEG (hedge) DISTANCE sweep on the NIFTY Weekly Credit Spread.
Sell leg ALWAYS at ATM; only the buy/hedge leg distance from ATM is swept: 100/150/200/250/300 (200 = base).
Everything else identical to nifty_weekly_credit_spread.py (entry = first session after prev expiry, DTE on
calendar days, touch-basis 3d-high/low breakout, 14:30 fallback, ATM=round(spot/50)*50, 1 trade/week, exit =
90% target [1-min] or DTE-0 spot-settlement intrinsic, no stop). Entry/direction/ATM are width-independent, so
each week is entered ONCE and only the hedge leg + net credit + exit vary per width. Max loss/trade = width -
net_credit (no stop, held to settlement). GROSS premium points, 1 spread. IS/OOS split 2025-08-29.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTHS = [100, 150, 200, 250, 300]; BASE_WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 30
OOS_SPLIT = pd.Timestamp("2025-08-29")


def dd_stats(pnl_series):
    """Max & avg drawdown (points) on the realized cumulative-P&L curve (already chronological)."""
    eq = np.concatenate([[0.0], np.cumsum(pnl_series)]); peak = np.maximum.accumulate(eq); dd = eq - peak
    depths = []; in_dd = False; tv = 0.0
    for k in range(len(eq)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]
            if dd[k] >= -1e-9: depths.append(abs(tv)); in_dd = False
    if in_dd: depths.append(abs(tv))
    if not depths: return 0.0, 0.0, 0
    return round(max(depths), 1), round(float(np.mean(depths)), 1), len(depths)


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    rows = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: skipped.append((e_cur.date(), "outside spot window")); continue
        dte = (e_cur - entry_day).days
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3: continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()

        ed = sp[sp["date"] == entry_day]
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
        d_open = ed["open"].iloc[0]; trig_i = None; typ = None; label = None
        for k in range(len(ed)):
            if md[k] > FB_MOD: break
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                if bh and bl: typ, label = ("PCS", "3d-high") if abs(d_open - d3h) <= abs(d_open - d3l) else ("CCS", "3d-low")
                elif bh: typ, label = "PCS", "3d-high"
                else: typ, label = "CCS", "3d-low"
                trig_i = k; break
        if trig_i is None:
            fb = ed[ed["mod"] >= FB_MOD]
            if fb.empty: continue
            trig_i = ed.index.get_loc(fb.index[0]); px = cl[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:30"
        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / 50) * 50
        trig_kind = "fallback" if label == "fallback-2:30" else "breakout"
        folder = e_cur.strftime("%Y%m%d"); ot = "CE" if typ == "CCS" else "PE"
        t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
        sser = leg_series(folder, atm, ot, entry_time, t_end)                      # sell leg @ ATM (shared across widths)
        if sser is None: skipped.append((e_cur.date(), f"missing sell {atm}{ot}")); continue
        try: s0 = float(sser.asof(entry_time))
        except Exception: s0 = np.nan
        if np.isnan(s0): skipped.append((e_cur.date(), "no sell premium")); continue
        S = settle.get(e_cur, np.nan)
        if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan

        for wd in WIDTHS:
            lK = atm + wd if typ == "CCS" else atm - wd
            lser = leg_series(folder, lK, ot, entry_time, t_end)
            if lser is None: skipped.append((e_cur.date(), f"w{wd} missing hedge {lK}{ot}")); continue
            try: l0 = float(lser.asof(entry_time))
            except Exception: l0 = np.nan
            if np.isnan(l0): skipped.append((e_cur.date(), f"w{wd} no hedge premium")); continue
            net_credit = s0 - l0
            if net_credit <= 0: skipped.append((e_cur.date(), f"w{wd} non-pos credit")); continue
            both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna()
            both = both[both.index > entry_time]; sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; bt = both.index.values
            thr = TARGET_FRAC * net_credit; hit = np.where(spread_val <= thr)[0]
            if len(hit):
                j = hit[0]; short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"; exit_time = pd.Timestamp(bt[j])
            else:
                if np.isnan(S): skipped.append((e_cur.date(), f"w{wd} no settle")); continue
                if typ == "CCS": short_x = float(max(0.0, S - atm)); long_x = float(max(0.0, S - lK))
                else: short_x = float(max(0.0, atm - S)); long_x = float(max(0.0, lK - S))
                reason = "DTE0 settlement"; exit_time = t_end
            pnl = net_credit - (short_x - long_x)
            rows.append({"width": wd, "entry_date": entry_day.date(), "DTE": dte, "type": ("CCS" if typ == "CCS" else "PCS"),
                         "trigger": label, "trig_kind": trig_kind, "ATM": int(atm), "leg": ot, "short_K": int(atm), "long_K": int(lK),
                         "net_credit": round(net_credit, 2), "max_loss": round(wd - net_credit, 2), "exit_reason": reason,
                         "pnl_points": round(pnl, 2), "oos": pd.Timestamp(entry_day) >= OOS_SPLIT})

    R = pd.DataFrame(rows)

    def stats(sub):
        n = len(sub)
        return {"trades": n, "win_%": round((sub.pnl_points > 0).mean() * 100, 1) if n else 0,
                "total_pnl": round(sub.pnl_points.sum(), 1), "avg_pnl": round(sub.pnl_points.mean(), 2) if n else 0,
                "breakout_%": round((sub.trig_kind == "breakout").mean() * 100, 1) if n else 0,
                "fallback_%": round((sub.trig_kind == "fallback").mean() * 100, 1) if n else 0,
                "avg_DTE": round(sub.DTE.mean(), 2) if n else 0, "avg_credit": round(sub.net_credit.mean(), 1) if n else 0,
                "avg_maxloss": round(sub.max_loss.mean(), 1) if n else 0, "worst_trade": round(sub.pnl_points.min(), 1) if n else 0,
                "best_trade": round(sub.pnl_points.max(), 1) if n else 0,
                "ret_per_maxloss": round(sub.pnl_points.sum() / sub.max_loss.mean(), 2) if n and sub.max_loss.mean() else 0}
    def wstats(wd):
        sub = R[R.width == wd].sort_values("entry_date"); mdd, add, nep = dd_stats(sub.pnl_points.values)
        s = stats(sub); s["max_dd"] = mdd; s["avg_dd"] = add; s["dd_episodes"] = nep
        s["ret_per_maxdd"] = round(s["total_pnl"] / mdd, 2) if mdd else 0
        return {"width": wd, **s}
    PER = pd.DataFrame([wstats(wd) for wd in WIDTHS])

    # best stable region on total_pnl (3-neighbour mean)
    v = PER.set_index("width")["total_pnl"].values; nb = np.array([np.nanmean(v[max(0, k - 1):k + 2]) for k in range(len(v))])
    bi = int(np.nanargmax(nb)); region = [WIDTHS[j] for j in range(max(0, bi - 1), min(len(WIDTHS), bi + 2))]; best_single = WIDTHS[int(np.nanargmax(v))]

    CMP = pd.DataFrame([{"width": wd, "total_pnl": round(R[R.width == wd].pnl_points.sum(), 1),
                         "avg_pnl": round(R[R.width == wd].pnl_points.mean(), 2), "win_%": round((R[R.width == wd].pnl_points > 0).mean() * 100, 1),
                         "avg_credit": round(R[R.width == wd].net_credit.mean(), 1), "avg_maxloss": round(R[R.width == wd].max_loss.mean(), 1),
                         "baseline": "<== 200 BASELINE" if wd == BASE_WIDTH else "", "best_region": "*" if wd in region else ""} for wd in WIDTHS])

    IS = pd.DataFrame([{"width": wd, **{f"IS_{k}": val for k, val in stats(R[(R.width == wd) & (~R.oos)]).items() if k in ("trades", "total_pnl", "avg_pnl", "win_%")},
                        **{f"OOS_{k}": val for k, val in stats(R[(R.width == wd) & (R.oos)]).items() if k in ("trades", "total_pnl", "avg_pnl", "win_%")}} for wd in WIDTHS])

    # dedicated drawdown comparison (overall + IS + OOS), on chronological realized cum-P&L
    def ddrow(wd):
        sub = R[R.width == wd].sort_values("entry_date")
        mo, ao, no = dd_stats(sub.pnl_points.values)
        mi, ai, ni = dd_stats(sub[~sub.oos].sort_values("entry_date").pnl_points.values)
        mx, ax, nx = dd_stats(sub[sub.oos].sort_values("entry_date").pnl_points.values)
        return {"width": wd, "total_pnl": round(sub.pnl_points.sum(), 1), "avg_maxloss": round(sub.max_loss.mean(), 1),
                "max_dd": mo, "avg_dd": ao, "dd_episodes": no, "ret_per_maxdd": round(sub.pnl_points.sum() / mo, 2) if mo else 0,
                "IS_max_dd": mi, "IS_avg_dd": ai, "OOS_max_dd": mx, "OOS_avg_dd": ax,
                "baseline": "<== 200" if wd == BASE_WIDTH else "", "best_region": "*" if wd in region else ""}
    DD = pd.DataFrame([ddrow(wd) for wd in WIDTHS])

    summary = pd.DataFrame([
        {"metric": "Sweep", "value": "buy/hedge-leg distance from ATM: 100/150/200/250/300 (sell leg fixed at ATM). 200 = existing baseline."},
        {"metric": "Base strategy", "value": "unchanged: entry, 3d-high/low touch breakout + 14:30 fallback, ATM, 90%-target/DTE0-settlement exit, 1 trade/week"},
        {"metric": "Window", "value": f"{R.entry_date.min()} .. {R.entry_date.max()}"},
        {"metric": "Best total-P&L region (stable)", "value": f"widths {region} (center {WIDTHS[bi]}); best single {best_single}"},
        {"metric": "RISK/REWARD TRADEOFF", "value": "narrower width caps max loss (=width-credit) but the hedge is bought closer/costlier -> LOWER net credit & lower target reward. Wider width = more credit but larger tail loss. See avg_credit vs avg_maxloss per width."},
        {"metric": "OOS split", "value": str(OOS_SPLIT.date()) + " (IS before / OOS after)"},
    ])
    with pd.ExcelWriter(OUTDIR / "credit_spread_width_sweep.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        PER.to_excel(w, sheet_name="Per_Width", index=False)
        CMP.to_excel(w, sheet_name="Comparison", index=False)
        DD.to_excel(w, sheet_name="Drawdown_Compare", index=False)
        IS.to_excel(w, sheet_name="IS_OOS", index=False)
        R.to_excel(w, sheet_name="All_Trades", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    R.to_csv(OUTDIR / "credit_spread_width_sweep_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("=" * 96 + "\nNIFTY WEEKLY CREDIT SPREAD - BUY-LEG (HEDGE) WIDTH SWEEP\n" + "=" * 96)
    print(f"window {R.entry_date.min()} .. {R.entry_date.max()} | {len(R)} trade-rows across {len(WIDTHS)} widths\n")
    print("--- PER WIDTH ---"); print(PER.to_string(index=False))
    print(f"\nBest stable region: widths {region} (center {WIDTHS[bi]}) | best single {best_single} | 200=baseline")
    print("\n--- COMPARISON (width vs total P&L) ---"); print(CMP.to_string(index=False))
    print("\n--- DRAWDOWN COMPARE (realized cum-P&L; max & avg) ---"); print(DD.to_string(index=False))
    print("\n--- IS / OOS ---"); print(IS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
