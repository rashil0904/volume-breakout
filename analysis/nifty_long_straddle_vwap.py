# -*- coding: utf-8 -*-
"""nifty_long_straddle_vwap.py — LONG ATM straddle VWAP strategy on NIFTY weekly expiries (DTE 6..0). SEPARATE
from the short straddle strategy; NO leg-shift. Reuses the SAME conventions: VWAP on ATM straddle (CE+PE),
vol = CE_vol+PE_vol, accumulates from 09:15, resets daily; ATM = chain-implied min|CE-PE| among REAL strikes.
BUY when straddle CLOSE > VWAP (1-min close); EXIT when straddle CLOSE < VWAP OR 3:20pm forced; re-enter on a
later close>VWAP (new ATM). No min-premium filter (unlike short). P&L = exit_total - entry_total (long). India
VIX at entry added. GROSS straddle premium points, 1 lot/leg."""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from straddle_vwap_backtest import build_front_map, DTE_MAX

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"; VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "long_straddle_vwap"; OUTDIR.mkdir(parents=True, exist_ok=True)
FORCE_MOD = 15 * 60 + 20; VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    if v < 9: return "<9"
    if v >= 20: return ">20"
    return f"{int(v)}-{int(v)+1}"


def run_day(day_df, dte):
    ce = day_df[day_df.option_type == "CE"]; pe = day_df[day_df.option_type == "PE"]
    times = np.array(sorted(day_df["timestamp"].unique())); strikes = np.array(sorted(day_df["strike"].unique()))
    ti = {t: i for i, t in enumerate(times)}; si = {s: j for j, s in enumerate(strikes)}; nT, nS = len(times), len(strikes)
    ri = ce["timestamp"].map(ti).values; ci = ce["strike"].map(si).values; rj = pe["timestamp"].map(ti).values; cj = pe["strike"].map(si).values

    def mat(sub, col, r, c):
        a = np.full((nT, nS), np.nan); a[r, c] = sub[col].values; return a
    CEc, CEv = mat(ce, "close", ri, ci), mat(ce, "volume", ri, ci); PEc, PEv = mat(pe, "close", rj, cj), mat(pe, "volume", rj, cj)
    CEsyn = np.ones((nT, nS), bool); CEsyn[ri, ci] = ce["is_synthetic"].values
    PEsyn = np.ones((nT, nS), bool); PEsyn[rj, cj] = pe["is_synthetic"].values
    real = (~CEsyn) & (~PEsyn) & ~np.isnan(CEc) & ~np.isnan(PEc)
    Sc = CEc + PEc; Sv = np.nan_to_num(CEv) + np.nan_to_num(PEv); absdiff = np.abs(CEc - PEc)
    mod = np.array([pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute for t in times])
    _w = np.where(mod <= FORCE_MOD)[0]; force_i = int(_w[-1]) if len(_w) else nT - 1

    def chain_atm(i):
        cand = real[i]
        if not cand.any(): return None
        return int(np.argmin(np.where(cand, absdiff[i], np.inf)))

    a0 = chain_atm(0)
    if a0 is None:
        for i in range(1, nT):
            a0 = chain_atm(i)
            if a0 is not None: break

    def walk(rolling):
        segs = []; cumPV = 0.0; cumV = 0.0; atm = a0; held = None; pos = 0; entry = np.nan; entry_t = None; had = False; typ = "initial entry"
        for i in range(nT):
            feed = chain_atm(i) if rolling else atm                 # rolling: VWAP+signal on re-centered ATM ; same: fixed strike
            if feed is not None and not np.isnan(Sc[i, feed]):
                cumPV += Sc[i, feed] * Sv[i, feed]; cumV += Sv[i, feed]
            vwap = (cumPV / cumV) if cumV > 0 else (Sc[i, feed] if feed is not None else np.nan)
            if i == force_i:
                if pos == 1:
                    xp = Sc[i, held] if not np.isnan(Sc[i, held]) else entry
                    segs.append(_seg(entry_t, entry, strikes[held], times[i], xp, "3:20pm forced", dte, typ, rolling))
                break
            if pos == 0:
                if feed is not None and not np.isnan(Sc[i, feed]) and cumV > 0 and Sc[i, feed] > vwap:
                    k = feed if rolling else chain_atm(i)            # entry strike = ATM at entry
                    if k is not None and not np.isnan(Sc[i, k]):
                        held = k; atm = k; entry = Sc[i, k]; entry_t = times[i]; pos = 1
                        typ = "re-entry" if had else "initial entry"; had = True
            else:
                sig = Sc[i, feed] if feed is not None else np.nan    # rolling ATM close (or fixed strike close)
                if not np.isnan(sig) and sig < vwap:                 # exit signal
                    xp = Sc[i, held] if not np.isnan(Sc[i, held]) else entry   # fill = HELD strike price
                    segs.append(_seg(entry_t, entry, strikes[held], times[i], xp, "VWAP-exit", dte, typ, rolling)); pos = 0
        return segs
    return walk(False) + walk(True)                                 # same-strike + rolling-ATM


def _seg(t_in, entry, strike, t_out, exit_total, reason, dte, typ, rolling):
    return {"variant": "rolling_atm" if rolling else "same_strike", "date": pd.Timestamp(t_in).date(), "DTE": dte, "segment_type": typ,
            "entry_time": pd.Timestamp(t_in).strftime("%H:%M"), "entry_strike": int(strike), "entry_total": round(float(entry), 2),
            "exit_time": pd.Timestamp(t_out).strftime("%H:%M"), "exit_reason": reason, "exit_total": round(float(exit_total), 2),
            "pnl_points": round(float(exit_total) - float(entry), 2)}


def process_expiry(args):
    exp, days = args; fs = list((OPTDIR / exp).glob("*.parquet"))
    cols = ["strike", "option_type", "timestamp", "DTE", "close", "volume", "is_synthetic"]; parts = []
    for f in fs:
        try:
            dd = pd.read_parquet(f, columns=cols); parts.append(dd[dd["DTE"].isin(range(0, DTE_MAX + 1))])
        except Exception:
            pass
    if not parts: return []
    chain = pd.concat(parts, ignore_index=True); chain["timestamp"] = pd.to_datetime(chain["timestamp"]); chain["day"] = chain["timestamp"].dt.normalize()
    out = []
    for day, dte in days:
        sub = chain[chain["day"] == pd.Timestamp(day)]
        if sub.empty: continue
        out += run_day(sub.drop(columns=["day"]), dte)
    return out


def main():
    front = build_front_map(); byexp = {}
    for day, (exp, dte) in front.items(): byexp.setdefault(exp, []).append((day, dte))
    print(f"front-weekly days {len(front)} | expiries {len(byexp)}")
    allsegs = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(process_expiry, (e, d)): e for e, d in byexp.items()}
        done = 0
        for f in as_completed(futs):
            allsegs += f.result(); done += 1
            if done % 20 == 0 or done == len(byexp): print(f"  {done}/{len(byexp)} expiries")

    T = pd.DataFrame(allsegs).sort_values(["variant", "date", "entry_time"]).reset_index(drop=True); T.insert(0, "seg_no", range(1, len(T) + 1))
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts).set_index("ts")["close"].sort_index()
    ent_ts = pd.to_datetime(T["date"].astype(str) + " " + T["entry_time"])
    T["entry_vix"] = [round(float(vser.asof(t)), 2) if pd.notna(vser.asof(t)) else np.nan for t in ent_ts]
    T["vix_bucket"] = T["entry_vix"].map(vbucket)
    em = pd.to_datetime(T["entry_time"], format="%H:%M"); xm = pd.to_datetime(T["exit_time"], format="%H:%M")
    T["hold_min"] = ((xm - em).dt.total_seconds() / 60).round(0)

    def vsum(V):
        D = V.groupby("date").agg(entries=("seg_no", "size"), re_entries=("segment_type", lambda s: int((s == "re-entry").sum())), day_pnl=("pnl_points", "sum")).reset_index()
        return {"segments": len(V), "win_%": round((V.pnl_points > 0).mean() * 100, 1), "total_points": round(V.pnl_points.sum(), 1),
                "avg_points": round(V.pnl_points.mean(), 2), "avg_day_pnl": round(D.day_pnl.mean(), 2), "profitable_days_%": round((D.day_pnl > 0).mean() * 100, 1),
                "entries/day": round(D.entries.mean(), 2), "re_entries/day": round(D.re_entries.mean(), 2), "avg_hold_min": round(V.hold_min.mean(), 1),
                "VWAP_exits": int((V.exit_reason == "VWAP-exit").sum()), "forced_320": int((V.exit_reason == "3:20pm forced").sum())}, D

    variants = ["same_strike", "rolling_atm"]; subs = {v: T[T.variant == v] for v in variants}; sm = {v: vsum(subs[v]) for v in variants}
    CMP = pd.DataFrame([{"metric": k, "same_strike": sm["same_strike"][0][k], "rolling_atm": sm["rolling_atm"][0][k]} for k in sm["same_strike"][0]])
    dtes = list(range(0, DTE_MAX + 1))
    DTEc = pd.DataFrame({"DTE": dtes, **{f"{v}_total": [round(subs[v][subs[v].DTE == b].pnl_points.sum(), 1) for b in dtes] for v in variants},
                         **{f"{v}_win%": [round((subs[v][subs[v].DTE == b].pnl_points > 0).mean() * 100, 1) if (subs[v].DTE == b).any() else 0 for b in dtes] for v in variants}})
    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    VIXc = pd.DataFrame({"vix_bucket": present, **{f"{v}_segments": [int((subs[v].vix_bucket == b).sum()) for b in present] for v in variants},
                         **{f"{v}_total": [round(subs[v][subs[v].vix_bucket == b].pnl_points.sum(), 1) for b in present] for v in variants},
                         **{f"{v}_win%": [round((subs[v][subs[v].vix_bucket == b].pnl_points > 0).mean() * 100, 1) if (subs[v].vix_bucket == b).any() else 0 for b in present] for v in variants}})

    with pd.ExcelWriter(OUTDIR / "long_straddle_vwap.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"metric": "Strategy", "value": "NIFTY LONG ATM straddle VWAP (buy>VWAP; no leg-shift); TWO VARIANTS. GROSS premium points, 1 lot/leg"},
                      {"metric": "same_strike", "value": "VWAP + exit-signal close track the FIXED entered strike"},
                      {"metric": "rolling_atm", "value": "VWAP + exit-signal close track the ROLLING nearest-spot ATM (re-centered each min); held position=entry strike (no shift), fill=held strike"},
                      {"metric": "Conventions", "value": "vol=CE+PE, ATM=chain-implied min|CE-PE| real strikes, VWAP from 09:15 daily reset; NO min-premium filter"},
                      {"metric": "Period", "value": f"{T.date.min()} .. {T.date.max()}"}]).to_excel(w, sheet_name="About", index=False)
        CMP.to_excel(w, sheet_name="Summary_Compare", index=False); DTEc.to_excel(w, sheet_name="DTE_Compare", index=False); VIXc.to_excel(w, sheet_name="VIX_Compare", index=False)
        subs["same_strike"].to_excel(w, sheet_name="SameStrike_Trades", index=False); subs["rolling_atm"].to_excel(w, sheet_name="RollingATM_Trades", index=False)
        sm["same_strike"][1].to_excel(w, sheet_name="SameStrike_Daily", index=False); sm["rolling_atm"][1].to_excel(w, sheet_name="RollingATM_Daily", index=False)
    T.to_csv(OUTDIR / "long_straddle_vwap_segments.csv", index=False)

    pd.set_option("display.width", 210)
    print("=" * 96 + "\nNIFTY LONG ATM STRADDLE VWAP — SAME-STRIKE vs ROLLING-ATM VWAP\n" + "=" * 96)
    print(f"period {T.date.min()}..{T.date.max()}\n"); print(CMP.to_string(index=False))
    print("\n--- DTE (total P&L) ---"); print(DTEc[["DTE", "same_strike_total", "rolling_atm_total"]].to_string(index=False))
    print("\n--- VIX buckets (total P&L) ---"); print(VIXc[["vix_bucket", "same_strike_total", "rolling_atm_total"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
