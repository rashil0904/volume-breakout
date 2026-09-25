# -*- coding: utf-8 -*-
"""nifty_3strike_rolling_straddle.py — "Intraday 3 Strike ATM Short Straddle" with rolling shorts + a per-side
hedge state machine. Confirmed conventions (user): ATM & breakeven = nearest 50; short/roll strikes move in
100s (window = center-100/center/center+100, center anchored to entry ATM on a 100-grid); multi-level move in
one candle => SINGLE JUMP to new center; intraday fills = crossing-minute CLOSE; roll detection = CLOSE-based.
Entry 09:16 candle OPEN; exit 15:15 CLOSE. GROSS points, lot-weighted (short legs 1 lot, hedge 3 lots).

FLAGS (not silently resolved): parity/off-grid hedge (catch impossible), entry-time catch, both-hedges-caught,
roll/jump opening a short at a hedge strike, missing option price. See simulate_day docstring for the state
machine (STATE1 normal toward=fixed / away=shift-by-roll maintaining offset; STATE2 caught: Sub-case A roll-off
-> back to NORMAL; Sub-case B reversal -> if |strike-ATM|>round(live_prem,100) unwind+rebuild fresh breakeven).
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_3strike_rolling_straddle"; OUTDIR.mkdir(parents=True, exist_ok=True)
ENTRY_MOD = 9 * 60 + 16; EXIT_MOD = 15 * 60 + 15; SHORT_LOT = 1; HEDGE_LOT = 3
R50 = lambda x: int(round(x / 50.0) * 50); R100 = lambda x: int(round(x / 100.0) * 100)


class Book:
    """lazy per-day option price access for one expiry folder."""
    def __init__(self, folder, day): self.folder = folder; self.day = pd.Timestamp(day).normalize(); self.c = {}
    def series(self, strike, ot):
        k = (int(strike), ot)
        if k in self.c: return self.c[k]
        fs = glob.glob(str(OPTDIR / self.folder / f"NIFTY_{int(strike)}_{ot}_*.parquet")); s = None
        if fs:
            o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
            o = o[o.ts.dt.normalize() == self.day].set_index("ts").sort_index()
            if len(o): s = o
        self.c[k] = s; return s
    def px(self, strike, ot, ts, field="close"):
        s = self.series(strike, ot)
        if s is None: return np.nan
        v = s[field].asof(ts)
        return float(v) if pd.notna(v) else np.nan


def simulate_day(D, folder, ed):
    """ed: spot rows for the day (columns mod, open, close, ts). Returns (events, pnl, stats, flags)."""
    bk = Book(folder, D)
    rows = ed.sort_values("mod"); r0 = rows.iloc[0]
    t_en = pd.Timestamp(r0["ts"])
    A0 = R50(r0["open"]); center = A0
    ce0 = bk.px(A0, "CE", t_en, "open"); pe0 = bk.px(A0, "PE", t_en, "open")
    flags = []; events = []; pnl = {"short": 0.0, "hedge": 0.0}
    if np.isnan(ce0) or np.isnan(pe0):
        return None, None, [{"date": D.date(), "flag": "no_entry_premium", "detail": f"ATM {A0}"}], []
    entry_prem = ce0 + pe0
    BE_CE = R50(A0 + entry_prem); BE_PE = R50(A0 - entry_prem)

    shorts = {}                                        # strike -> {"ce":entry, "pe":entry}
    def open_short(s, ts, tag):
        ce = bk.px(s, "CE", ts); pe = bk.px(s, "PE", ts)
        if np.isnan(ce) or np.isnan(pe): flags.append({"date": D.date(), "flag": "missing_price_short", "detail": f"{s} @ {pd.Timestamp(ts).strftime('%H:%M')}"})
        shorts[s] = {"ce": ce, "pe": pe}
        events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": tag, "leg": "short straddle", "strike": s, "qty": -SHORT_LOT, "ce_px": ce, "pe_px": pe})
    def close_short(s, ts, tag):
        d = shorts.pop(s); ce = bk.px(s, "CE", ts); pe = bk.px(s, "PE", ts)
        pnl["short"] += ((d["ce"] - ce) + (d["pe"] - pe)) * SHORT_LOT              # short: entry-exit
        events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": tag, "leg": "short straddle", "strike": s, "qty": SHORT_LOT, "ce_px": ce, "pe_px": pe})

    for s in (A0 - 100, A0, A0 + 100): open_short(s, t_en, "ENTRY-SELL")

    # hedges: long HEDGE_LOT
    def hedge_new(side, strike, ts, tag):
        ot = "CE" if side == "CE" else "PE"; p = bk.px(strike, ot, ts, "open" if tag == "ENTRY-HEDGE" else "close")
        if np.isnan(p): flags.append({"date": D.date(), "flag": "missing_price_hedge", "detail": f"{side} {strike} @ {pd.Timestamp(ts).strftime('%H:%M')}"})
        events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": tag, "leg": f"hedge {side}", "strike": strike, "qty": HEDGE_LOT, "ce_px": p if side == "CE" else "", "pe_px": p if side == "PE" else ""})
        return {"side": side, "strike": strike, "entry": p, "state": "NORMAL", "caught_dir": 0}
    def hedge_close(H, ts, tag):
        ot = H["side"]; p = bk.px(H["strike"], ot, ts, "close")
        pnl["hedge"] += (p - H["entry"]) * HEDGE_LOT
        events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": tag, "leg": f"hedge {H['side']}", "strike": H["strike"], "qty": -HEDGE_LOT, "ce_px": p if ot == "CE" else "", "pe_px": p if ot == "PE" else ""})

    HCE = hedge_new("CE", BE_CE, t_en, "ENTRY-HEDGE"); HPE = hedge_new("PE", BE_PE, t_en, "ENTRY-HEDGE")
    # parity / entry-catch flags
    for H in (HCE, HPE):
        if H["strike"] in shorts:
            H["state"] = "CAUGHT"; H["caught_dir"] = 1 if H["side"] == "CE" else -1
            flags.append({"date": D.date(), "flag": "entry_time_catch", "detail": f"{H['side']} hedge {H['strike']} == short strike"})
        else:
            catchable = ((H["strike"] - A0) % 100) == (100 - 100) if False else (((H["strike"] - (A0 - 100)) % 100) == 0)
            if not catchable: flags.append({"date": D.date(), "flag": "hedge_off_parity_uncatchable", "detail": f"{H['side']} hedge {H['strike']} off shorts' grid (parity)"})

    n_catch = sum(H["state"] == "CAUGHT" for H in (HCE, HPE)); n_jump = 0; both_flagged = False

    def roll(new_center, ts):
        nonlocal center, n_catch, n_jump
        direction = 1 if new_center > center else -1; roll_amt = new_center - center
        if abs(roll_amt) > 100: flags.append({"date": D.date(), "flag": "multi_level_jump", "detail": f"{center}->{new_center} ({pd.Timestamp(ts).strftime('%H:%M')}) single-jump"})
        new_win = {new_center - 100, new_center, new_center + 100}
        closed = set()
        for s in [x for x in shorts if x not in new_win]: close_short(s, ts, "ROLL-SQUAREOFF"); closed.add(s)
        for s in sorted(new_win - set(shorts)):
            open_short(s, ts, "ROLL-SELL")            # a new short reaching a fixed hedge strike IS the normal catch (handled below)
        for H in (HCE, HPE):
            if H["state"] == "CAUGHT":
                if (H["strike"] in closed) or (H["strike"] not in shorts):       # Sub-case A: caught leg rolled off
                    H["state"] = "NORMAL"; H["caught_dir"] = 0
                    events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": "CATCH-RESOLVE(A)", "leg": f"hedge {H['side']}", "strike": H["strike"], "qty": 0, "ce_px": "", "pe_px": ""})
                elif direction == -H["caught_dir"]:                               # Sub-case B: reversal
                    lp = bk.px(new_center, "CE", ts) + bk.px(new_center, "PE", ts); thr = R100(lp); dist = abs(H["strike"] - new_center)
                    if not np.isnan(lp) and dist > thr:
                        hedge_close(H, ts, "HEDGE-UNWIND(B)")
                        newBE = R50(new_center + lp) if H["side"] == "CE" else R50(new_center - lp)
                        if newBE in shorts: flags.append({"date": D.date(), "flag": "hedge_jump_onto_short", "detail": f"{H['side']} jump BE {newBE} == live short"})
                        nh = hedge_new(H["side"], newBE, ts, "HEDGE-JUMP(B)"); H.update(nh); n_jump += 1
                # else continuation while caught -> hold fixed
            else:                                                                # STATE 1 normal
                toward = abs(new_center - H["strike"]) < abs(center - H["strike"])
                if (not toward) and roll_amt != 0:                               # away -> shift maintaining offset
                    hedge_close(H, ts, "HEDGE-SHIFT-close"); nh = hedge_new(H["side"], H["strike"] + roll_amt, ts, "HEDGE-SHIFT-open"); H.update(nh)
                if H["state"] == "NORMAL" and H["strike"] in shorts:            # catch check
                    H["state"] = "CAUGHT"; H["caught_dir"] = direction; n_catch += 1
                    events.append({"date": D.date(), "time": pd.Timestamp(ts).strftime("%H:%M"), "event": "CATCH", "leg": f"hedge {H['side']}", "strike": H["strike"], "qty": 0, "ce_px": "", "pe_px": ""})
        center = new_center
        nonlocal both_flagged
        if HCE["state"] == "CAUGHT" and HPE["state"] == "CAUGHT" and not both_flagged:
            both_flagged = True; flags.append({"date": D.date(), "flag": "both_hedges_caught", "detail": f"first @ {pd.Timestamp(ts).strftime('%H:%M')}"})

    centers = [center]
    for _, r in rows.iloc[1:].iterrows():
        if r["mod"] > EXIT_MOD: break
        if r["mod"] == EXIT_MOD: break
        nc = A0 + 100 * int(np.floor((r["close"] - A0) / 100.0 + 0.5))
        if nc != center: roll(nc, pd.Timestamp(r["ts"])); centers.append(center)

    # exit 15:15 squareoff
    exrow = rows[rows["mod"] == EXIT_MOD]
    t_ex = pd.Timestamp(exrow.iloc[0]["ts"]) if len(exrow) else pd.Timestamp(rows.iloc[-1]["ts"])
    for s in list(shorts): close_short(s, t_ex, "EXIT-SQUAREOFF")
    for H in (HCE, HPE): hedge_close(H, t_ex, "EXIT-SQUAREOFF")

    net = pnl["short"] + pnl["hedge"]
    stats = {"date": D.date(), "A0": A0, "entry_prem": round(entry_prem, 1), "BE_CE": BE_CE, "BE_PE": BE_PE,
             "short_pnl": round(pnl["short"], 1), "hedge_pnl": round(pnl["hedge"], 1), "net_pnl": round(net, 1),
             "n_rolls": len(centers) - 1, "n_catch": n_catch, "n_jump": n_jump,
             "atm_range": int(max(centers) - min(centers)), "n_flags": len(flags)}
    return events, stats, flags, []


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    sp = sp[(sp["mod"] >= ENTRY_MOD) & (sp["mod"] <= EXIT_MOD)]
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); exp_dates = [e.normalize() for e in expiries]
    FLOOR = pd.Timestamp("2024-10-01")
    days = [d for d in sorted(sp["date"].unique()) if pd.Timestamp(d) >= FLOOR]
    lim = None
    if len(sys.argv) > 1: lim = int(sys.argv[1])                # test on first N days

    all_ev = []; all_stats = []; all_flags = []; done = 0
    for D in days:
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")                 # front expiry >= day
        ed = sp[sp["date"] == D]
        if ed["mod"].min() > ENTRY_MOD or len(ed) < 5: continue
        ev, st, fl, _ = simulate_day(pd.Timestamp(D), folder, ed)
        if st is None: all_flags += fl; continue
        all_ev += ev; all_stats.append(st); all_flags += fl; done += 1
        if lim and done >= lim: break

    S = pd.DataFrame(all_stats); EV = pd.DataFrame(all_ev); FL = pd.DataFrame(all_flags)
    win = round((S.net_pnl > 0).mean() * 100, 1); tot = round(S.net_pnl.sum(), 1)
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Intraday 3-Strike ATM Short Straddle + rolling hedge state machine"},
        {"metric": "Conventions", "value": "ATM/BE nearest 50; strikes shift 100; single-jump; close-based rolls; fills=crossing close; entry 09:16 open / exit 15:15 close; GROSS lot-weighted pts (short 1 / hedge 3)"},
        {"metric": "Window", "value": f"{S.date.min()} .. {S.date.max()}"},
        {"metric": "Trading days", "value": len(S)},
        {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / day", "value": round(S.net_pnl.mean(), 2)},
        {"metric": "Median P&L / day", "value": round(S.net_pnl.median(), 2)},
        {"metric": "Short-leg total / avg", "value": f"{round(S.short_pnl.sum(),1)} / {round(S.short_pnl.mean(),2)}"},
        {"metric": "Hedge-leg total / avg", "value": f"{round(S.hedge_pnl.sum(),1)} / {round(S.hedge_pnl.mean(),2)}"},
        {"metric": "Max profit / loss day", "value": f"{round(S.net_pnl.max(),1)} / {round(S.net_pnl.min(),1)}"},
        {"metric": "Avg rolls / day", "value": round(S.n_rolls.mean(), 2)},
        {"metric": "Avg catch events / day", "value": round(S.n_catch.mean(), 2)},
        {"metric": "Avg hedge-jump (Sub-case B) / day", "value": round(S.n_jump.mean(), 2)},
        {"metric": "Avg ATM range traveled / day (pts)", "value": round(S.atm_range.mean(), 1)},
        {"metric": "Days with >=1 catch", "value": int((S.n_catch > 0).sum())},
        {"metric": "Days with >=1 hedge-jump", "value": int((S.n_jump > 0).sum())},
        {"metric": "Flag rows (review)", "value": len(FL)},
    ])
    with pd.ExcelWriter(OUTDIR / "nifty_3strike_rolling_straddle.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        S.to_excel(w, sheet_name="Daily_PnL", index=False)
        (FL if len(FL) else pd.DataFrame([{"flag": "none"}])).to_excel(w, sheet_name="Flags", index=False)
        EV.head(4000).to_excel(w, sheet_name="Position_Log_sample", index=False)
    EV.to_csv(OUTDIR / "position_log_full.csv", index=False); S.to_csv(OUTDIR / "daily_pnl.csv", index=False)
    if len(FL): FL.to_csv(OUTDIR / "flags.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY INTRADAY 3-STRIKE ROLLING SHORT STRADDLE\n" + "=" * 96)
    print(f"days {len(S)} | win {win}% | TOTAL {tot:,} pts | avg/day {round(S.net_pnl.mean(),2)} | median {round(S.net_pnl.median(),2)}")
    print(f"short-leg {round(S.short_pnl.sum(),1)} | hedge-leg {round(S.hedge_pnl.sum(),1)}")
    print(f"avg rolls {round(S.n_rolls.mean(),2)} | avg catch {round(S.n_catch.mean(),2)} | avg jump {round(S.n_jump.mean(),2)} | avg ATM range {round(S.atm_range.mean(),1)}")
    print(f"days w/ catch {int((S.n_catch>0).sum())} | days w/ jump {int((S.n_jump>0).sum())} | flag rows {len(FL)}")
    if len(FL): print("\nflag types:"); print(FL['flag'].value_counts().to_string())
    print("\nfirst 5 days:"); print(S.head(5).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
