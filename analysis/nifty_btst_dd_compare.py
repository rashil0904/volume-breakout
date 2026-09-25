# -*- coding: utf-8 -*-
"""nifty_btst_dd_compare.py — compare Max/Avg DRAWDOWN of the BTST Close-Direction strategy at sell-leg
offsets +-200 vs +-300 (everything else identical). DD on realized cumulative option-premium P&L."""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from nifty_btst_close_direction import leg_at, drawdown_episodes

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction"; ENTRY_MOD = 15 * 60 + 15; OFFS = [200, 300]


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max(); day_open = sp.groupby("date")["open"].first(); spot1515 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index); expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); first_exp = expiries[0]

    rows = []
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1515.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; folder = E.strftime("%Y%m%d")
        espot = float(spot1515.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        ot = "PE" if espot < dopen else "CE"; sgn = -1 if espot < dopen else 1
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=25)
        Le, Lx = leg_at(folder, atm, ot, t_en, t_ex)
        if Le is None or Lx is None or np.isnan(Le) or np.isnan(Lx): continue
        rec = {"date": D}
        ok = True
        for off in OFFS:
            Se, Sx = leg_at(folder, atm + sgn * off, ot, t_en, t_ex)
            if Se is None or Sx is None or np.isnan(Se) or np.isnan(Sx): ok = False; break
            rec[off] = (2 * Lx - Sx) - (2 * Le - Se)
        if ok: rows.append(rec)

    R = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = []
    for off in OFFS:
        cum = R[off].cumsum().values; eq = np.concatenate([[0.0], cum]); tm = np.concatenate([[pd.Timestamp(R.date.iloc[0])], pd.to_datetime(R.date).values])
        DE = drawdown_episodes(eq, tm); mx = DE.drawdown_points.max(); tot = round(R[off].sum(), 1)
        out.append({"sell_offset": f"+-{off}", "trades": len(R), "total_pnl": tot, "win_%": round((R[off] > 0).mean() * 100, 1),
                    "max_DD": round(mx, 1), "avg_DD": round(DE.drawdown_points.mean(), 1), "n_DD_episodes": len(DE),
                    "max_DD_duration_days": int(DE.days_peak_to_recovery.max()), "return_over_maxDD": round(tot / mx, 2)})
    CMP = pd.DataFrame(out)

    x = OUTDIR / "btst_close_direction.xlsx"
    try:
        with pd.ExcelWriter(x, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            CMP.to_excel(w, sheet_name="DD_Offset_Compare", index=False)
        saved = x.name
    except Exception:
        CMP.to_csv(OUTDIR / "dd_offset_compare.csv", index=False); saved = "dd_offset_compare.csv (excel locked)"

    pd.set_option("display.width", 200)
    print("=" * 80 + "\nBTST close-direction — DRAWDOWN: +-200 vs +-300 sell offset\n" + "=" * 80)
    print(CMP.to_string(index=False))
    print(f"\nSaved -> {saved}")


if __name__ == "__main__":
    main()
