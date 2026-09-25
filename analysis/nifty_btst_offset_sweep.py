# -*- coding: utf-8 -*-
"""nifty_btst_offset_sweep.py — SELL-LEG OFFSET sweep on the NIFTY Close-Direction BTST. Base unchanged
(15:15 in / 09:25-next-day out; RED=+2 ATM PE, GREEN=+2 ATM CE; DTE-0 -> next-week contract). ONLY the sell
leg distance from ATM is swept: 100/150/200/250/300. Buy leg stays 2x ATM. OPTION PREMIUM points, GROSS.
Long leg computed once/day; short leg varied. Reports per-offset splits (RED/GREEN, normal/expiry), IS/OOS.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
VIX = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "btst_offset_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSETS = [100, 150, 200, 250, 300]; ENTRY_MOD = 15 * 60 + 15; OOS_SPLIT = pd.Timestamp("2025-08-29")
VIX_BUCKETS = ["<8"] + [f"{i}-{i+1}" for i in range(8, 20)] + [">20"]        # <8, 8-9, ... 19-20, >20


def vix_bucket(v):
    if np.isnan(v): return "n/a"
    if v < 8: return "<8"
    if v >= 20: return ">20"
    return f"{int(v)}-{int(v)+1}"


def leg_at(folder, strike, otype, t_entry, t_exit):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None, None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    s = o[(o.ts >= t_entry - pd.Timedelta(minutes=5)) & (o.ts <= t_exit + pd.Timedelta(minutes=5))].set_index("ts")["close"].sort_index()
    if s.empty: return None, None
    return s.asof(t_entry), s.asof(t_exit)


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max(); day_open = sp.groupby("date")["open"].first(); spot1515 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIX); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(ts=vts); vx["date"] = vts.dt.normalize(); vx["mod"] = vts.dt.hour * 60 + vts.dt.minute
    vix1515 = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()        # India VIX at 15:15 (entry)
    tdays = sorted(day_open.index); expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    recs = []
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1515.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; is_exp = D in expset; folder = E.strftime("%Y%m%d")
        espot = float(spot1515.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"; sgn = -1 if direction == "RED" else 1
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=25)
        Le, Lx = leg_at(folder, atm, ot, t_en, t_ex)                       # ATM long leg (once)
        if Le is None or Lx is None or np.isnan(Le) or np.isnan(Lx): continue
        shorts = {}
        for off in OFFSETS:
            Se, Sx = leg_at(folder, atm + sgn * off, ot, t_en, t_ex)
            shorts[off] = None if (Se is None or Sx is None or np.isnan(Se) or np.isnan(Sx)) else (2 * Lx - Sx) - (2 * Le - Se)
        vixv = float(vix1515.get(D, np.nan)); recs.append({"date": D, "direction": direction, "is_exp": is_exp, "pnl": shorts, "vix": vixv, "vix_bucket": vix_bucket(vixv)})

    def agg(sub, off):
        p = np.array([r["pnl"][off] for r in sub if r["pnl"][off] is not None])
        return {"trades": len(p), "win_%": round((p > 0).mean() * 100, 1) if len(p) else 0, "total_pnl": round(p.sum(), 1), "avg_pnl": round(p.mean(), 2) if len(p) else 0}

    def sub(cond): return [r for r in recs if cond(r)]
    groups = {"ALL": recs, "RED": sub(lambda r: r["direction"] == "RED"), "GREEN": sub(lambda r: r["direction"] == "GREEN"),
              "normal": sub(lambda r: not r["is_exp"]), "expiry": sub(lambda r: r["is_exp"]),
              "IS": sub(lambda r: r["date"] < OOS_SPLIT), "OOS": sub(lambda r: r["date"] >= OOS_SPLIT)}

    MAIN = pd.DataFrame([{"offset": off, **{f"{g}_{k}": agg(groups[g], off)[k] for g in ["ALL", "RED", "GREEN", "normal", "expiry"] for k in ["total_pnl"]},
                          "ALL_trades": agg(recs, off)["trades"], "ALL_win%": agg(recs, off)["win_%"], "ALL_avg": agg(recs, off)["avg_pnl"]} for off in OFFSETS])
    ISOOS = pd.DataFrame([{"offset": off, "IS_total": agg(groups["IS"], off)["total_pnl"], "OOS_total": agg(groups["OOS"], off)["total_pnl"],
                           "IS_avg": agg(groups["IS"], off)["avg_pnl"], "OOS_avg": agg(groups["OOS"], off)["avg_pnl"]} for off in OFFSETS])

    v = MAIN["ALL_total_pnl"].values; nb = np.array([np.mean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
    bi = int(np.argmax(nb)); center = OFFSETS[bi]; region = OFFSETS[max(0, bi - 1):min(len(OFFSETS), bi + 2)]; best_single = OFFSETS[int(np.argmax(v))]

    with pd.ExcelWriter(OUTDIR / "btst_offset_sweep.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "BTST close-direction — SELL-LEG offset sweep (buy leg fixed 2x ATM); OPTION PREMIUM points, GROSS"},
            {"metric": "Offsets", "value": str(OFFSETS)}, {"metric": "Baseline", "value": "200 (existing spec)"},
            {"metric": "Trades", "value": len(recs)}, {"metric": "OOS split", "value": str(OOS_SPLIT.date())},
            {"metric": "Best single offset", "value": f"{best_single} = {v[int(np.argmax(v))]} pts"},
            {"metric": "BEST STABLE REGION (3-neighbour)", "value": f"center {center} | region {region} | pts {[MAIN.ALL_total_pnl.iloc[OFFSETS.index(o)] for o in region]}"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        MAIN.to_excel(w, sheet_name="Sweep_splits", index=False); ISOOS.to_excel(w, sheet_name="IS_OOS", index=False)
        for g in ["ALL", "RED", "GREEN", "normal", "expiry"]:
            pd.DataFrame([{"offset": off, **agg(groups[g], off)} for off in OFFSETS]).to_excel(w, sheet_name=f"By_{g}", index=False)
        # ---- VIX segregation: summary matrix (bucket x offset total P&L) + one sheet per bucket ----
        present = [b for b in VIX_BUCKETS if any(r["vix_bucket"] == b for r in recs)]
        vsum = []
        for b in present:
            bsub = [r for r in recs if r["vix_bucket"] == b]
            row = {"vix_bucket": b, "n_trades": len(bsub), "avg_vix": round(np.mean([r["vix"] for r in bsub]), 2)}
            for off in OFFSETS: row[f"total_pnl_{off}"] = agg(bsub, off)["total_pnl"]
            row["best_offset"] = OFFSETS[int(np.argmax([agg(bsub, off)["total_pnl"] for off in OFFSETS]))]
            vsum.append(row)
        pd.DataFrame(vsum).to_excel(w, sheet_name="VIX_Summary", index=False)
        for b in present:
            bsub = [r for r in recs if r["vix_bucket"] == b]
            pd.DataFrame([{"offset": off, **agg(bsub, off), "avg_vix": round(np.mean([r["vix"] for r in bsub]), 2)} for off in OFFSETS]).to_excel(w, sheet_name=f"VIX_{b}"[:31], index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for g, col in [("ALL", "#1F6FB2"), ("RED", "#C0392B"), ("GREEN", "#2E8B57")]:
        ax.plot(OFFSETS, [agg(groups[g], o)["total_pnl"] for o in OFFSETS], "-o", ms=5, label=g, color=col)
    ax.axvline(200, color="#888", ls="--", lw=1, label="baseline 200"); ax.axvspan(min(region), max(region), color="#999", alpha=.12, label=f"best region ~{center}")
    ax.set_title("BTST — sell-leg offset sweep (total option-premium P&L)"); ax.set_xlabel("Sell-leg offset from ATM (pts)"); ax.set_ylabel("Total P&L (premium points)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "offset_sweep_curve.png", dpi=130); plt.close(fig)
    MAIN.to_csv(OUTDIR / "btst_offset_sweep.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nBTST SELL-LEG OFFSET SWEEP | trades " + str(len(recs)) + " | OOS split " + str(OOS_SPLIT.date()) + "\n" + "=" * 96)
    print("\n--- per offset: total P&L split (ALL / RED / GREEN / normal / expiry) ---")
    print(MAIN[["offset", "ALL_total_pnl", "ALL_win%", "ALL_avg", "RED_total_pnl", "GREEN_total_pnl", "normal_total_pnl", "expiry_total_pnl"]].to_string(index=False))
    print("\n--- IS vs OOS (total P&L) ---"); print(ISOOS.to_string(index=False))
    print(f"\nbaseline 200 = {MAIN.ALL_total_pnl.iloc[OFFSETS.index(200)]} pts | best single {best_single} | BEST REGION center {center} region {region}")
    print("\n--- BY VIX BUCKET (n trades | total P&L at offset 100/200/300 | best offset) ---")
    for b in [x for x in VIX_BUCKETS if any(r["vix_bucket"] == x for r in recs)]:
        bs = [r for r in recs if r["vix_bucket"] == b]
        print(f"  VIX {b:>6}: n {len(bs):3d} | avgVIX {np.mean([r['vix'] for r in bs]):5.2f} | pnl@100 {agg(bs,100)['total_pnl']:8.1f} @200 {agg(bs,200)['total_pnl']:8.1f} @300 {agg(bs,300)['total_pnl']:8.1f} | best {OFFSETS[int(np.argmax([agg(bs,o)['total_pnl'] for o in OFFSETS]))]}")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
