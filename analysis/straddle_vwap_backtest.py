# -*- coding: utf-8 -*-
"""straddle_vwap_backtest.py — NIFTY intraday rolling ATM SHORT-STRADDLE, VWAP-of-straddle based.
Uses the pulled NIFTY options 1-min chain (Oct 2024 - 31 Jul 2026). P&L in STRADDLE PREMIUM POINTS
(1 lot/leg placeholder; sizing NOT defined -> NOT INR). GROSS (no cost).

CONFIRMED CONVENTIONS (per user):
- VWAP weight per minute = CE_volume + PE_volume (SUM). Price = CE_close + PE_close. Accumulates from
  09:15, RESETS daily, CONTINUES across shifts (new straddle appended, no reset). VWAP includes the
  current candle before the compare. If cumulative vol == 0 -> VWAP = current straddle price (no false trigger).
- ATM = CHAIN-IMPLIED: strike minimizing |CE_close - PE_close| at each DECISION POINT (entry/re-entry/
  shift). Between decision points the strike is FIXED and feeds the running VWAP. No spot needed (full period).
- Leg-SL is CLOSE-based on 1-min (CE_close>=leg_SL OR PE_close>=leg_SL). All exits fill both legs at CLOSE.
- MIN PREMIUM FILTER: never enter/re-enter/shift-into an ATM straddle whose total < 25 points.

RULES: entry when straddle_close < VWAP AND straddle_close >= 25 -> SHORT (sell CE+PE at close). entry_total=
CE+PE; leg_SL=entry_total*0.67. Each candle, IN ORDER: (B) if straddle_close > VWAP -> VWAP-EXIT (close both
@close), then watch re-entry; (A) elif CE_close>=leg_SL OR PE_close>=leg_SL -> SHIFT (close old @close, open
fresh ATM @close if >=25, else go flat), new leg_SL; else hold. Force-exit at 15:20. Front weekly, DTE 0..6.
"""
import sys, os, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "straddle_vwap"; OUTDIR.mkdir(parents=True, exist_ok=True)
SL_MULT = 0.69; FORCE_MOD = 15 * 60 + 20        # leg_SL = entry_total * 0.69 ; 15:20 forced exit
DTE_MAX = 6; MIN_STRIKES = 15                    # expiries with fewer strikes = incomplete pull -> excluded (flagged)
MIN_PREMIUM = float(os.environ.get("MINPREM", "50"))   # do NOT trade if ATM straddle total < this (default 50)


def build_front_map():
    """day -> (front_expiry_folder, DTE). front = smallest non-negative DTE across expiries with a real chain."""
    front = {}; skipped_exp = []
    for exp in sorted(os.listdir(OPTDIR)):
        fs = list((OPTDIR / exp).glob("*.parquet"))
        if not fs:
            continue
        nstrk = len(set(re.search(r"NIFTY_(\d+)_", f.name).group(1) for f in fs))
        if nstrk < MIN_STRIKES:                  # incomplete chain -> do not let it be a front expiry
            skipped_exp.append((exp, nstrk)); continue
        df = pd.read_parquet(fs[len(fs) // 2], columns=["timestamp", "DTE"])
        dmap = df.assign(day=pd.to_datetime(df["timestamp"]).dt.normalize()).groupby("day")["DTE"].first()
        for day, dte in dmap.items():
            if 0 <= dte <= DTE_MAX and (day not in front or dte < front[day][1]):
                front[day] = (exp, int(dte))
    if skipped_exp:
        print(f"  EXCLUDED {len(skipped_exp)} incomplete expiries (< {MIN_STRIKES} strikes): {skipped_exp}")
    return front


def run_day(day_df, dte):
    """intraday state machine for one day's front-expiry chain. returns (segments, daysummary)."""
    ce = day_df[day_df.option_type == "CE"]; pe = day_df[day_df.option_type == "PE"]
    times = np.array(sorted(day_df["timestamp"].unique())); strikes = np.array(sorted(day_df["strike"].unique()))
    ti = {t: i for i, t in enumerate(times)}; si = {s: j for j, s in enumerate(strikes)}
    nT, nS = len(times), len(strikes)

    ri = ce["timestamp"].map(ti).values; ci = ce["strike"].map(si).values
    rj = pe["timestamp"].map(ti).values; cj = pe["strike"].map(si).values

    def mat(sub, col, r, c):
        a = np.full((nT, nS), np.nan); a[r, c] = sub[col].values; return a
    CEc, CEv = mat(ce, "close", ri, ci), mat(ce, "volume", ri, ci)     # leg-SL is now CLOSE-based -> no 'high' needed
    PEc, PEv = mat(pe, "close", rj, cj), mat(pe, "volume", rj, cj)
    CEsyn = np.ones((nT, nS), bool); CEsyn[ri, ci] = ce["is_synthetic"].values      # True = synthetic/missing
    PEsyn = np.ones((nT, nS), bool); PEsyn[rj, cj] = pe["is_synthetic"].values
    real = (~CEsyn) & (~PEsyn) & ~np.isnan(CEc) & ~np.isnan(PEc)                     # both legs REAL (non-synthetic) -> valid ATM candidate
    if real.any(axis=0).sum() < 8:                                                   # too few real strikes all day -> untradeable (thin/incomplete chain)
        return [], {"date": pd.Timestamp(times[0]).date(), "DTE": dte, "entries": 0, "initial": 0, "re_entries": 0,
                    "shifts": 0, "segments": 0, "day_pnl_points": 0.0, "day_end": "skipped-thin-chain"}
    Sc = CEc + PEc                                   # straddle close per strike
    Sv = np.nan_to_num(CEv) + np.nan_to_num(PEv)     # straddle volume per strike (SUM)
    absdiff = np.abs(CEc - PEc)
    mod = np.array([pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute for t in times])
    _w = np.where(mod <= FORCE_MOD)[0]; force_i = int(_w[-1]) if len(_w) else nT - 1   # nT-1 = special/evening session (e.g. Muhurat)

    def chain_atm(i):
        cand = real[i]
        if not cand.any():
            return None
        return int(np.argmin(np.where(cand, absdiff[i], np.inf)))   # min|CE-PE| among REAL strikes only

    segs = []; cumPV = 0.0; cumV = 0.0
    atm = chain_atm(0)
    if atm is None:
        for i in range(1, nT):
            atm = chain_atm(i)
            if atm is not None:
                break
    pos = None; n_init = 0; n_re = 0; n_shift = 0; had_entry = False
    for i in range(nT):
        # ---- running VWAP on current atm straddle (includes this candle) ----
        if atm is not None and not np.isnan(Sc[i, atm]):
            s_now, v_now = Sc[i, atm], Sv[i, atm]
            cumPV += s_now * v_now; cumV += v_now
        vwap = (cumPV / cumV) if cumV > 0 else (Sc[i, atm] if atm is not None else np.nan)

        if i == force_i:                              # ---- 15:20 forced exit ----
            if pos is not None:
                k = pos["k"]
                segs.append(_seg(pos, times[i], strikes[k], CEc[i, k], PEc[i, k], "3:20pm forced", dte, vwap))
                pos = None
            break

        if pos is None:                               # ---- flat: entry / re-entry (straddle<VWAP AND >= MIN_PREMIUM) ----
            if atm is not None and not np.isnan(Sc[i, atm]) and cumV > 0 and Sc[i, atm] < vwap and Sc[i, atm] >= MIN_PREMIUM:
                k = chain_atm(i)
                if k is not None and not np.isnan(Sc[i, k]) and Sc[i, k] >= MIN_PREMIUM:
                    atm = k; ce0, pe0 = CEc[i, k], PEc[i, k]; et = ce0 + pe0
                    typ = "re-entry" if had_entry else "initial entry"
                    if had_entry: n_re += 1
                    else: n_init += 1; had_entry = True
                    pos = {"k": k, "ce": ce0, "pe": pe0, "entry_total": et, "sl": et * SL_MULT, "t_in": times[i], "typ": typ, "vwap_in": vwap}
        else:                                         # ---- in position: B (VWAP-exit) then A (leg-SL, CLOSE-based) ----
            k = pos["k"]; sc = Sc[i, k]; sl = pos["sl"]
            checkB = (not np.isnan(sc)) and (sc > vwap)
            checkA = (CEc[i, k] >= sl) or (PEc[i, k] >= sl)      # CLOSE-based leg SL (was intraday-high touch)
            if checkB:
                segs.append(_seg(pos, times[i], strikes[k], CEc[i, k], PEc[i, k], "VWAP-exit", dte, vwap)); pos = None
            elif checkA:
                segs.append(_seg(pos, times[i], strikes[k], CEc[i, k], PEc[i, k], "leg-shift", dte, vwap)); n_shift += 1   # both legs @ close
                nk = chain_atm(i)                     # open fresh ATM at this candle's close (only if straddle >= MIN_PREMIUM)
                if nk is not None and not np.isnan(Sc[i, nk]) and Sc[i, nk] >= MIN_PREMIUM:
                    atm = nk; ce0, pe0 = CEc[i, nk], PEc[i, nk]; et = ce0 + pe0
                    pos = {"k": nk, "ce": ce0, "pe": pe0, "entry_total": et, "sl": et * SL_MULT, "t_in": times[i], "typ": "shift", "vwap_in": vwap}
                else:
                    pos = None
            # else hold

    day_pnl = round(sum(s["pnl_points"] for s in segs), 2)
    end = segs[-1]["exit_reason"] if segs else "no-trade"
    day_end = "forced 3:20pm" if end == "3:20pm forced" else ("VWAP-exit standing" if end == "VWAP-exit" else end)
    summary = {"date": pd.Timestamp(times[0]).date(), "DTE": dte, "entries": n_init + n_re, "initial": n_init,
               "re_entries": n_re, "shifts": n_shift, "segments": len(segs), "day_pnl_points": day_pnl, "day_end": day_end}
    return segs, summary


def _seg(pos, t_out, strike_out, ce_exit, pe_exit, reason, dte, exit_vwap):
    exit_total = float(ce_exit) + float(pe_exit)
    return {"date": pd.Timestamp(pos["t_in"]).date(), "DTE": dte, "segment_type": pos["typ"],
            "entry_time": pd.Timestamp(pos["t_in"]).strftime("%H:%M"), "entry_strike": int(strike_out),
            "ce_entry": round(float(pos["ce"]), 2), "pe_entry": round(float(pos["pe"]), 2),
            "entry_total": round(pos["entry_total"], 2), "entry_vwap": round(float(pos["vwap_in"]), 2),
            "exit_time": pd.Timestamp(t_out).strftime("%H:%M"), "exit_reason": reason,
            "ce_exit": round(float(ce_exit), 2), "pe_exit": round(float(pe_exit), 2), "exit_total": round(exit_total, 2),
            "exit_vwap": round(float(exit_vwap), 2), "pnl_points": round(pos["entry_total"] - exit_total, 2)}


def process_expiry(args):
    exp, days = args
    fs = list((OPTDIR / exp).glob("*.parquet"))
    cols = ["strike", "option_type", "timestamp", "DTE", "close", "volume", "is_synthetic"]
    parts = []
    for f in fs:
        try:
            d = pd.read_parquet(f, columns=cols)
            parts.append(d[d["DTE"].isin(range(0, DTE_MAX + 1))])
        except Exception:
            pass
    if not parts:
        return [], []
    chain = pd.concat(parts, ignore_index=True)
    chain["timestamp"] = pd.to_datetime(chain["timestamp"]); chain["day"] = chain["timestamp"].dt.normalize()
    allsegs, allsum = [], []
    for day, dte in days:
        sub = chain[chain["day"] == pd.Timestamp(day)]
        if sub.empty:
            continue
        segs, summ = run_day(sub.drop(columns=["day"]), dte)
        allsegs += segs; allsum.append(summ)
    return allsegs, allsum


def main():
    front = build_front_map()
    byexp = {}
    for day, (exp, dte) in front.items():
        byexp.setdefault(exp, []).append((day, dte))
    print(f"front-weekly trading days: {len(front)} | expiries used: {len(byexp)} | DTE 0..{DTE_MAX}")
    allsegs, allsum = [], []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(process_expiry, (e, d)): e for e, d in byexp.items()}
        done = 0
        for f in as_completed(futs):
            s, u = f.result(); allsegs += s; allsum += u; done += 1
            if done % 20 == 0 or done == len(byexp): print(f"  {done}/{len(byexp)} expiries")

    T = pd.DataFrame(allsegs)
    T = T.sort_values(["date", "entry_time"]).reset_index(drop=True); T.insert(0, "seg_no", range(1, len(T) + 1))
    D = pd.DataFrame(allsum).sort_values("date").reset_index(drop=True)

    tot = round(T["pnl_points"].sum(), 1); win = round((T["pnl_points"] > 0).mean() * 100, 1)
    ndays = len(D)
    dte_rows = []
    for b in range(0, DTE_MAX + 1):
        sub = T[T.DTE == b]
        dte_rows.append({"DTE": b, "segments": len(sub), "win_%": round((sub.pnl_points > 0).mean() * 100, 1) if len(sub) else 0,
                         "total_points": round(sub.pnl_points.sum(), 1), "avg_points": round(sub.pnl_points.mean(), 2) if len(sub) else 0})
    DTEB = pd.DataFrame(dte_rows)
    by_reason = T.groupby("exit_reason").agg(segments=("pnl_points", "size"), total_pts=("pnl_points", "sum"),
                                             win_pct=("pnl_points", lambda x: round((x > 0).mean() * 100, 1))).reset_index()

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY intraday rolling ATM SHORT STRADDLE, VWAP-of-straddle (points, 1 lot placeholder)"},
        {"metric": "SIZING FLAG", "value": "P&L in STRADDLE PREMIUM POINTS, NOT INR (sizing not defined; 1 lot/leg placeholder). GROSS, no cost."},
        {"metric": "Period", "value": f"{D.date.min()} .. {D.date.max()}"},
        {"metric": "Conventions", "value": "VWAP vol=CE+PE(sum); ATM=chain-implied min|CE-PE| at decision pts; leg-SL CLOSE-based (both legs exit @close)"},
        {"metric": "Min-premium filter", "value": f"no entry/re-entry/shift into an ATM straddle < {int(MIN_PREMIUM)} points"},
        {"metric": "Trading days (front weekly, DTE 0..6)", "value": ndays},
        {"metric": "Total segments (entry->exit/shift)", "value": len(T)},
        {"metric": "Win rate % (segments)", "value": win},
        {"metric": "TOTAL P&L (straddle points)", "value": tot},
        {"metric": "Avg points / segment", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Avg day P&L (points)", "value": round(D.day_pnl_points.mean(), 2)},
        {"metric": "Profitable days %", "value": round((D.day_pnl_points > 0).mean() * 100, 1)},
        {"metric": "Avg shifts / day", "value": round(D.shifts.mean(), 2)},
        {"metric": "Avg re-entries / day", "value": round(D.re_entries.mean(), 2)},
        {"metric": "Avg entries / day (incl re-entries)", "value": round(D.entries.mean(), 2)},
        {"metric": "Days ending forced 3:20pm", "value": int((D.day_end == "forced 3:20pm").sum())},
        {"metric": "Days ending VWAP-exit standing", "value": int((D.day_end == "VWAP-exit standing").sum())},
    ])

    T.to_csv(OUTDIR / "straddle_vwap_segments.csv", index=False); D.to_csv(OUTDIR / "straddle_vwap_daily.csv", index=False)
    xlsx = OUTDIR / "straddle_vwap.xlsx"
    try:
        wr = pd.ExcelWriter(xlsx, engine="openpyxl")
    except PermissionError:
        xlsx = OUTDIR / "straddle_vwap_new.xlsx"; wr = pd.ExcelWriter(xlsx, engine="openpyxl")
        print(f"  (straddle_vwap.xlsx was locked/open -> wrote {xlsx.name} instead)")
    with wr as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        DTEB.to_excel(w, sheet_name="DTE_Buckets", index=False)
        by_reason.to_excel(w, sheet_name="By_Exit_Reason", index=False)
        D.to_excel(w, sheet_name="Daily_Summary", index=False)
        T.to_excel(w, sheet_name="Trades_Segments", index=False)

    # ---- SEPARATE workbook with CE & PE leg prices at entry AND exit ----
    legcols = ["seg_no", "date", "DTE", "segment_type", "entry_time", "entry_strike",
               "ce_entry", "pe_entry", "entry_total", "entry_vwap",
               "exit_time", "exit_reason", "ce_exit", "pe_exit", "exit_total", "exit_vwap", "pnl_points"]
    Tl = T[[c for c in legcols if c in T.columns]]
    legx = OUTDIR / "straddle_vwap_legs.xlsx"
    try:
        wr2 = pd.ExcelWriter(legx, engine="openpyxl")
    except PermissionError:
        legx = OUTDIR / "straddle_vwap_legs_new.xlsx"; wr2 = pd.ExcelWriter(legx, engine="openpyxl")
        print(f"  (straddle_vwap_legs.xlsx locked -> wrote {legx.name})")
    with wr2 as w:
        Tl.to_excel(w, sheet_name="Segments_CE_PE", index=False)
    Tl.to_csv(OUTDIR / "straddle_vwap_legs.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 96 + "\nNIFTY ROLLING ATM SHORT STRADDLE — VWAP-of-straddle (points; 1-lot placeholder, NOT INR)\n" + "=" * 96)
    print(f"period {D.date.min()} .. {D.date.max()} | days {ndays} | segments {len(T)}")
    print(f"win {win}% | TOTAL {tot:,} straddle-points | avg/seg {round(T.pnl_points.mean(),2)} | avg day {round(D.day_pnl_points.mean(),2)} | profitable days {round((D.day_pnl_points>0).mean()*100,1)}%")
    print(f"avg shifts/day {round(D.shifts.mean(),2)} | avg re-entries/day {round(D.re_entries.mean(),2)} | forced-3:20 days {int((D.day_end=='forced 3:20pm').sum())} | vwap-standing {int((D.day_end=='VWAP-exit standing').sum())}")
    print("\n--- DTE BUCKETS ---"); print(DTEB.to_string(index=False))
    print("\n--- BY EXIT REASON ---"); print(by_reason.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
