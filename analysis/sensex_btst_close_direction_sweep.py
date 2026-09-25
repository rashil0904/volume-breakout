# -*- coding: utf-8 -*-
"""sensex_btst_close_direction_sweep.py — SENSEX adaptation of the NIFTY Daily Close-Direction BTST, with a
full OFFSET x EXIT-TIME grid sweep. Entry FIXED at 15:15 (spot vs day-open): RED(spot<open)=+2 ATM PE / -1
(ATM-offset) PE ; GREEN=+2 ATM CE / -1 (ATM+offset) CE. ATM=round(spot/100)*100 (SENSEX step 100). Never
DTE-0: contract = nearest expiry STRICTLY AFTER entry day (current wk normal days, next wk on expiry day),
from the ACTUAL SENSEX expiry calendar (folder names; weekday Fri->Tue@2025-01-07->Thu@2025-09-04). Entry
price = option CLOSE at 15:15; EXIT = next-day minute OPEN. Sweep: offset {400,500,600,700,800} x exit-min
{09:16..10:00} = 225 cells. No VIX filter; India VIX (NIFTY vol index, used as general proxy) segregation only.
P&L = OPTION PREMIUM points (2xlong - 1xshort), GROSS. Lot size 20 (reference; points unaffected).
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "sensex_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "SENSEX"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "sensex_btst_close_direction"; OUTDIR.mkdir(parents=True, exist_ok=True)
ENTRY_MOD = 15 * 60 + 15; STEP = 100; OFFSETS = [400, 500, 600, 700, 800]
EXIT_MODS = list(range(9 * 60 + 16, 10 * 60 + 1))          # 09:16 .. 10:00 inclusive (45)
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def leg_series(folder, strike, ot, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"SENSEX_{int(strike)}_{ot}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts").sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first(); spot_entry = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    rows = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot_entry.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; is_exp = D in expset; folder = E.strftime("%Y%m%d")
        espot = float(spot_entry.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / STEP) * STEP
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        exit_ts = [pd.Timestamp(Dn) + pd.Timedelta(minutes=m) for m in EXIT_MODS]
        t0 = t_en - pd.Timedelta(minutes=10); t1 = pd.Timestamp(Dn) + pd.Timedelta(hours=10, minutes=5)
        # long leg (ATM) shared across offsets
        Lo = leg_series(folder, atm, ot, t0, t1)
        if Lo is None: skipped += 1; continue
        L_en = Lo["close"].asof(t_en); L_ex = Lo["open"].reindex(exit_ts, method="ffill").values
        if np.isnan(L_en): skipped += 1; continue
        vixv = round(float(vix_ent.get(D, np.nan)), 2)
        for off in OFFSETS:
            sK = atm - off if direction == "RED" else atm + off
            So = leg_series(folder, sK, ot, t0, t1)
            if So is None: skipped += 1; continue
            S_en = So["close"].asof(t_en); S_ex = So["open"].reindex(exit_ts, method="ffill").values
            if np.isnan(S_en): skipped += 1; continue
            entry_cost = 2 * L_en - S_en
            pnl = (2 * L_ex - S_ex) - entry_cost                    # per exit-minute vector
            for k, m in enumerate(EXIT_MODS):
                if np.isnan(pnl[k]): continue
                rows.append({"date": D.date(), "offset": off, "exit_mod": m, "direction": direction,
                             "is_expiry_day": is_exp, "entry_vix": vixv, "pnl": round(float(pnl[k]), 2)})
    R = pd.DataFrame(rows)
    R["exit_time"] = R["exit_mod"].map(lambda m: f"{m//60:02d}:{m%60:02d}")

    # ---- grids: offset x exit-time ----
    def grid(val, aggfunc):
        g = R.pivot_table(index="offset", columns="exit_time", values="pnl", aggfunc=aggfunc)
        return g.reindex(index=OFFSETS)
    tot = grid("pnl", "sum").round(1)
    winr = R.assign(w=(R.pnl > 0)).pivot_table(index="offset", columns="exit_time", values="w", aggfunc="mean").reindex(OFFSETS).mul(100).round(1)
    cnt = grid("pnl", "size")

    # best stable region: 3x3 neighbourhood mean over the total-P&L grid
    A = tot.values.astype(float); best = None
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            sub = A[max(0, r-1):r+2, max(0, c-1):c+2]; mnb = np.nanmean(sub)
            if best is None or mnb > best[0]: best = (mnb, r, c)
    br, bc = best[1], best[2]
    reg_off = [OFFSETS[x] for x in range(max(0, br-1), min(len(OFFSETS), br+2))]
    reg_ext = list(tot.columns[max(0, bc-1):bc+3])
    best_single = np.unravel_index(np.nanargmax(A), A.shape)

    # ---- VIX segregation (informational; NOT filtered) — at base offset 500 across all exit-times pooled ----
    def vixtab(sub):
        present = [b for b in VIX_BUCKETS if (sub.bucket == b).any()]
        return pd.DataFrame([{"vix_bucket": b, "trades": int((sub.bucket == b).sum()),
                              "win_%": round((sub[sub.bucket == b].pnl > 0).mean() * 100, 1),
                              "total_pnl": round(sub[sub.bucket == b].pnl.sum(), 1),
                              "avg_pnl": round(sub[sub.bucket == b].pnl.mean(), 2)} for b in present])
    R["bucket"] = R["entry_vix"].map(vbucket)
    VIX_all = vixtab(R); VIX_500 = vixtab(R[R.offset == 500])

    # per-cell long table + per-offset & per-exit marginal summaries
    marg_off = R.groupby("offset").agg(trades=("pnl", "size"), win_pct=("pnl", lambda x: round((x > 0).mean()*100, 1)),
                                       total_pnl=("pnl", lambda x: round(x.sum(), 1)), avg=("pnl", lambda x: round(x.mean(), 2))).reset_index()
    marg_ext = R.groupby("exit_time").agg(trades=("pnl", "size"), win_pct=("pnl", lambda x: round((x > 0).mean()*100, 1)),
                                          total_pnl=("pnl", lambda x: round(x.sum(), 1)), avg=("pnl", lambda x: round(x.mean(), 2))).reset_index()

    ndays = R["date"].nunique()
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "SENSEX Daily Close-Direction BTST — offset x exit-time sweep. Entry 15:15 (spot vs open); never DTE-0 (nearest expiry after entry day)."},
        {"metric": "Structure", "value": "RED(spot<open): +2 ATM PE / -1 (ATM-off) PE ; GREEN: +2 ATM CE / -1 (ATM+off) CE ; 1 trade/day"},
        {"metric": "Entry/exit price", "value": "entry = option CLOSE @15:15 ; EXIT = next-day minute OPEN"},
        {"metric": "ATM / strike step", "value": "round(spot/100)*100 ; SENSEX step 100 (verified)"},
        {"metric": "Offsets swept", "value": str(OFFSETS)}, {"metric": "Exit minutes swept", "value": f"09:16..10:00 (45)"},
        {"metric": "P&L unit", "value": "OPTION PREMIUM points (2xlong-1xshort), GROSS. Lot=20 (ref; points unaffected)."},
        {"metric": "Expiry weekday FLAG", "value": "SENSEX weekly NOT constant-Tuesday: Fri -> Tue(2025-01-07) -> Thu(2025-09-04). Actual calendar used."},
        {"metric": "VIX FLAG", "value": "India VIX = NSE/NIFTY vol index (no SENSEX intraday vol index pulled); used as general proxy for segregation ONLY, no filtering."},
        {"metric": "Trading days used", "value": ndays}, {"metric": "Cells (offset x exit)", "value": f"{len(OFFSETS)} x {len(EXIT_MODS)} = {len(OFFSETS)*len(EXIT_MODS)}"},
        {"metric": "Skipped (missing leg/minute)", "value": skipped},
        {"metric": "BEST stable region", "value": f"offsets {reg_off} x exit {reg_ext[0]}..{reg_ext[-1]} (3x3 nbhd mean {round(best[0],1)})"},
        {"metric": "Best single cell", "value": f"offset {OFFSETS[best_single[0]]} @ {tot.columns[best_single[1]]} = {round(A[best_single],1)} pts"},
        {"metric": "Grand total P&L (all cells)", "value": round(R.pnl.sum(), 1)},
    ])

    with pd.ExcelWriter(OUTDIR / "sensex_btst_sweep.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        tot.reset_index().to_excel(w, sheet_name="Grid_TotalPnL", index=False)
        winr.reset_index().to_excel(w, sheet_name="Grid_WinRate", index=False)
        cnt.reset_index().to_excel(w, sheet_name="Grid_TradeCount", index=False)
        marg_off.to_excel(w, sheet_name="By_Offset", index=False); marg_ext.to_excel(w, sheet_name="By_ExitTime", index=False)
        VIX_all.to_excel(w, sheet_name="VIX_all_offsets", index=False); VIX_500.to_excel(w, sheet_name="VIX_offset500", index=False)
    R.drop(columns=["bucket"]).to_csv(OUTDIR / "sensex_btst_sweep_trades.csv", index=False)

    pd.set_option("display.width", 250)
    print("=" * 100 + "\nSENSEX DAILY CLOSE-DIRECTION BTST — OFFSET x EXIT-TIME SWEEP\n" + "=" * 100)
    print(f"days {ndays} | cells {len(OFFSETS)}x{len(EXIT_MODS)} | skipped {skipped} | grand total {round(R.pnl.sum(),1)} pts")
    print("\n--- TOTAL P&L grid (offset rows x exit-time cols) [every 3rd minute] ---")
    print(tot[[c for k, c in enumerate(tot.columns) if k % 3 == 0]].to_string())
    print("\n--- WIN% grid [every 3rd minute] ---")
    print(winr[[c for k, c in enumerate(winr.columns) if k % 3 == 0]].to_string())
    print(f"\nBEST stable region: offsets {reg_off} x exit {reg_ext[0]}..{reg_ext[-1]} | best single offset {OFFSETS[best_single[0]]} @ {tot.columns[best_single[1]]} = {round(A[best_single],1)}")
    print("\n--- BY OFFSET (pooled over exit-times) ---"); print(marg_off.to_string(index=False))
    print("\n--- BY EXIT-TIME (pooled over offsets; every 5th) ---"); print(marg_ext.iloc[::5].to_string(index=False))
    print("\n--- VIX buckets (ALL offsets; informational) ---"); print(VIX_all.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
