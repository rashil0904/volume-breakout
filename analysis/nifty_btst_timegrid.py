# -*- coding: utf-8 -*-
"""nifty_btst_timegrid.py — BTST Close-Direction: ENTRY x EXIT time-grid sweep + India VIX 17-19 filter.
NEW variant (baseline 3:15/9:25 kept for reference). Entry minute 15:00..15:29 (30) x exit minute
09:16..10:00 next day (45) = 1350 cells. Entry price / direction / ATM from that MINUTE's OPEN (spot open
for dir/ATM; option-leg opens for value). Offset fixed +-200. DTE-0 -> next-week contract. VIX = India VIX
open at the entry minute; filter excludes trades with VIX in [17,19]. OPTION PREMIUM points, GROSS.
Reports full grids (P&L / win% / count) with & without the VIX filter, best stable region, baseline cell, IS/OOS.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"; VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "btst_timegrid"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MODS = list(range(900, 930)); EXIT_MODS = list(range(556, 601))     # 15:00-15:29 ; 09:16-10:00 ; sell +-300
VIX_LO, VIX_HI = 17.0, 19.0; OOS_SPLIT = pd.Timestamp("2025-08-29")
BASE_ENTRY = 915; BASE_EXIT = 565                                                        # 3:15pm / 9:25am baseline


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max(); day_open = sp.groupby("date")["open"].first()
    so = sp[sp["mod"].isin(ENTRY_MODS)].pivot_table(index="date", columns="mod", values="open")   # spot open at entry minutes
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(ts=vts); vx["date"] = vts.dt.normalize(); vx["mod"] = vts.dt.hour * 60 + vts.dt.minute
    vo = vx[vx["mod"].isin(ENTRY_MODS)].pivot_table(index="date", columns="mod", values="open")     # VIX open at entry minutes
    tdays = sorted(day_open.index); expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); first_exp = expiries[0]

    dpnl = []; dvix = []; ddate = []                                                     # per-day: pnl[30x45], vix[30], date
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in so.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        folder = expiries[j].strftime("%Y%m%d"); dop = float(day_open.loc[D])
        cache = {}
        def leg(strike, ot):
            if (strike, ot) not in cache:
                fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
                if not fs: cache[(strike, ot)] = None
                else:
                    o = pd.read_parquet(fs[0], columns=["timestamp", "open"]); o["ts"] = pd.to_datetime(o["timestamp"])
                    s = o[(o.ts >= pd.Timestamp(D) + pd.Timedelta(hours=14)) & (o.ts <= pd.Timestamp(Dn) + pd.Timedelta(hours=10, minutes=5))].set_index("ts")["open"].sort_index()
                    cache[(strike, ot)] = s if len(s) else None
            return cache[(strike, ot)]
        pnl = np.full((len(ENTRY_MODS), len(EXIT_MODS)), np.nan); vrow = np.full(len(ENTRY_MODS), np.nan)
        exit_ts = [pd.Timestamp(Dn) + pd.Timedelta(minutes=xm) for xm in EXIT_MODS]
        for ei, em in enumerate(ENTRY_MODS):
            sop = so.loc[D, em] if em in so.columns else np.nan
            if np.isnan(sop): continue
            direction = "RED" if sop < dop else "GREEN"; ot = "PE" if direction == "RED" else "CE"; sgn = -1 if direction == "RED" else 1
            atm = round(sop / 50) * 50; ls = leg(atm, ot); ss = leg(atm + sgn * OFFSET, ot)
            if ls is None or ss is None: continue
            et = pd.Timestamp(D) + pd.Timedelta(minutes=em)
            le = ls.asof(et); se = ss.asof(et)
            if np.isnan(le) or np.isnan(se): continue
            entry_val = 2 * le - se
            lx = np.array([ls.asof(t) for t in exit_ts]); sx = np.array([ss.asof(t) for t in exit_ts])
            pnl[ei] = (2 * lx - sx) - entry_val
            vrow[ei] = vo.loc[D, em] if (D in vo.index and em in vo.columns) else np.nan
        dpnl.append(pnl); dvix.append(vrow); ddate.append(D)

    P = np.array(dpnl); V = np.array(dvix); dates = pd.to_datetime(ddate)                 # P: (days,30,45)
    nE, nX = len(ENTRY_MODS), len(EXIT_MODS)

    def grids(daymask):
        Pm = P[daymask]; Vm = V[daymask]
        # no filter
        tot = np.nansum(Pm, axis=0); cnt = np.sum(~np.isnan(Pm), axis=0)
        win = np.where(cnt > 0, np.nansum(Pm > 0, axis=0) / np.maximum(cnt, 1) * 100, np.nan)
        # VIX filter: per entry-row, drop days with VIX in [VIX_LO,VIX_HI]
        keep = ~((Vm >= VIX_LO) & (Vm <= VIX_HI))                                         # (days,30)
        Pf = np.where(keep[:, :, None], Pm, np.nan)
        totf = np.nansum(Pf, axis=0); cntf = np.sum(~np.isnan(Pf), axis=0)
        winf = np.where(cntf > 0, np.nansum(Pf > 0, axis=0) / np.maximum(cntf, 1) * 100, np.nan)
        return tot, win, cnt, totf, winf, cntf

    tot, win, cnt, totf, winf, cntf = grids(np.ones(len(P), bool))
    ei0 = ENTRY_MODS.index(BASE_ENTRY); xi0 = EXIT_MODS.index(BASE_EXIT)

    def best_region(g):
        nb = np.full(g.shape, -1e18)
        for a in range(g.shape[0]):
            for b in range(g.shape[1]):
                sub = g[max(0, a - 1):a + 2, max(0, b - 1):b + 2]; nb[a, b] = np.nanmean(sub)
        a, b = np.unravel_index(np.nanargmax(nb), nb.shape)
        return a, b, round(float(nb[a, b]), 1)

    ba, bb, bmean = best_region(tot); baf, bbf, bmeanf = best_region(totf)
    lab = lambda em: f"{em//60}:{em%60:02d}"
    ism = np.asarray(dates < OOS_SPLIT); oosm = ~ism
    tIS = grids(ism)[3 if False else 0]; tOOS = grids(oosm)[0]                            # no-filter IS/OOS totals
    aIS, bIS, _ = best_region(tIS); aO, bO, _ = best_region(tOOS)

    idxE = [lab(e) for e in ENTRY_MODS]; idxX = [lab(x) for x in EXIT_MODS]
    def df(g): return pd.DataFrame(np.round(g, 1), index=idxE, columns=idxX)

    with pd.ExcelWriter(OUTDIR / "btst_timegrid.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": f"BTST close-direction ENTRY x EXIT time grid + VIX 17-19 filter (option premium points, GROSS, offset +-{OFFSET})"},
            {"metric": "Grid", "value": f"entry {lab(900)}-{lab(929)} x exit {lab(556)}-{lab(600)} = {nE}x{nX} = {nE*nX} cells; price = minute OPEN"},
            {"metric": "VIX filter", "value": f"exclude trades with India VIX (entry-minute open) in [{VIX_LO},{VIX_HI}]"},
            {"metric": "Days", "value": len(P)}, {"metric": "OOS split", "value": str(OOS_SPLIT.date())},
            {"metric": "BASELINE 3:15/9:25 — total P&L (no filter)", "value": round(float(tot[ei0, xi0]), 1)},
            {"metric": "BASELINE 3:15/9:25 — win% / trades", "value": f"{round(float(win[ei0,xi0]),1)} / {int(cnt[ei0,xi0])}"},
            {"metric": "BASELINE 3:15/9:25 — total P&L (VIX-filtered)", "value": round(float(totf[ei0, xi0]), 1)},
            {"metric": "BASELINE 3:15/9:25 — VIX-filt win% / trades", "value": f"{round(float(winf[ei0,xi0]),1)} / {int(cntf[ei0,xi0])}"},
            {"metric": "VIX filter removed (baseline)", "value": f"{int(cnt[ei0,xi0]-cntf[ei0,xi0])} trades"},
            {"metric": "BEST REGION no-filter (3x3)", "value": f"entry {idxE[ba]} x exit {idxX[bb]} | cell {round(float(tot[ba,bb]),1)} | neigh-mean {bmean}"},
            {"metric": "BEST REGION VIX-filter (3x3)", "value": f"entry {idxE[baf]} x exit {idxX[bbf]} | cell {round(float(totf[baf,bbf]),1)} | neigh-mean {bmeanf}"},
            {"metric": "OOS stability (no-filter best region)", "value": f"IS best entry {idxE[aIS]} x exit {idxX[bIS]} | OOS best entry {idxE[aO]} x exit {idxX[bO]}"},
            {"metric": "OVERFIT FLAG", "value": "1350 cells searched -> read the REGION not the single best cell; confirm IS/OOS agreement before trusting"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        df(tot).to_excel(w, sheet_name="PnL_noVIX"); df(totf).to_excel(w, sheet_name="PnL_VIXfilter")
        df(win).to_excel(w, sheet_name="Win_noVIX"); df(cnt).to_excel(w, sheet_name="TradeCount")
        df(tIS).to_excel(w, sheet_name="PnL_IS_noVIX"); df(tOOS).to_excel(w, sheet_name="PnL_OOS_noVIX")

    for g, tag in [(tot, "noVIX"), (totf, "VIXfilter")]:
        fig, ax = plt.subplots(figsize=(13, 8)); im = ax.imshow(g, aspect="auto", cmap="RdYlGn", origin="upper")
        ax.set_xticks(range(0, nX, 3)); ax.set_xticklabels([idxX[i] for i in range(0, nX, 3)], rotation=90, fontsize=7)
        ax.set_yticks(range(nE)); ax.set_yticklabels(idxE, fontsize=7)
        ax.scatter([xi0], [ei0], marker="*", s=180, color="black", label="baseline 3:15/9:25")
        ax.set_xlabel("EXIT minute (next day)"); ax.set_ylabel("ENTRY minute"); ax.set_title(f"BTST entry x exit total P&L ({tag}) — * = 3:15/9:25 baseline")
        fig.colorbar(im, ax=ax, label="total P&L (premium pts)"); ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(OUTDIR / f"timegrid_{tag}.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 240)
    print("=" * 96 + f"\nBTST ENTRY x EXIT TIME GRID ({nE}x{nX}={nE*nX}) + VIX 17-19 filter | days {len(P)}\n" + "=" * 96)
    print(f"\nBASELINE 3:15/9:25: no-filter {round(float(tot[ei0,xi0]),1)} pts ({round(float(win[ei0,xi0]),1)}% win, {int(cnt[ei0,xi0])} tr) | "
          f"VIX-filtered {round(float(totf[ei0,xi0]),1)} pts ({round(float(winf[ei0,xi0]),1)}% win, {int(cntf[ei0,xi0])} tr) | removed {int(cnt[ei0,xi0]-cntf[ei0,xi0])}")
    print(f"\nGRID no-filter: best cell {round(float(np.nanmax(tot)),1)} | BEST REGION entry {idxE[ba]} x exit {idxX[bb]} (neigh-mean {bmean})")
    print(f"GRID VIX-filter: best cell {round(float(np.nanmax(totf)),1)} | BEST REGION entry {idxE[baf]} x exit {idxX[bbf]} (neigh-mean {bmeanf})")
    print(f"OOS: IS best entry {idxE[aIS]}x{idxX[bIS]} | OOS best entry {idxE[aO]}x{idxX[bO]} -> {'AGREE-ish' if abs(aIS-aO)<=3 and abs(bIS-bO)<=6 else 'DIVERGE (overfit risk)'}")
    print("\ncorner check (total P&L, no filter): entry-rows x a few exit cols")
    show = df(tot).iloc[[0, 7, 15, 22, 29], [0, 9, 20, 30, 44]]; print(show.to_string())
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
