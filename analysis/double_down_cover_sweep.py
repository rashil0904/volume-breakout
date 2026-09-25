# -*- coding: utf-8 -*-
"""
double_down_cover_sweep.py
==========================
Sweep the SHORT square-off (cover) time for the full double-down. Long side + short-open
prices are the existing double-down; only the cover timing varies. Cover grid: 12:15 ..
15:00 (12 candle opens). A short can only cover AFTER its own open (target/9:45/12:00);
since the grid starts at 12:15 > 12:00, all shorts are eligible at all grid times.

short_pnl(cover T) = shares × (short_open_price − open_at_T). combined = long + short.
net_A = long@0.23% + short@0.10%; net_B = long@0.38% + short@0.10% (per-leg notional).

Scenario 1: single cover time (12 rows). Scenario 2: split cover on the SHORT's own
profitability at t1 (short_ret_t1>0 -> cover@t1, else hold to t2), 66 (t1<t2) pairs.

Flags: (a) split condition uses short profitability at t1 (price fell below short-open ->
cover winning short early). (b) cover only after the short's own open. (c) cover at the
cover candle's OPEN.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "double_down_cover_sweep"
BASE_POOL, TARGET = 500_000, 14.0
LONG_023, LONG_038, SHORT_RATE = 0.0023, 0.0038, 0.0010
COVER_HMS = list(range(735, 901, 15))                  # 12:15 .. 15:00
CLABEL = [f"{h//60:02d}:{h%60:02d}" for h in COVER_HMS]
NC = len(COVER_HMS)


def build():
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    opens, highs, _ = fpr.fetch_ohlc_and_exitday(base)
    pct_high = (highs - e[:, None]) / e[:, None] * 100
    t1, t2 = pts.HM_0945, pts.HM_1200
    ot1, ot2 = opens[:, pts.HCOL[t1]], opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - e) / e * 100
    pre_hms = [hm for hm in pts.CANDLE_HMS if hm < t1]
    bet_hms = [hm for hm in pts.CANDLE_HMS if t1 < hm < t2]
    N = len(base)
    so_hm = np.full(N, np.nan); so_px = np.full(N, np.nan); long_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        ph = pct_high[i]; a = None
        for h in pre_hms:
            v = ph[pts.HCOL[h]]
            if not np.isnan(v) and v >= TARGET:
                a, hm, px = 1, h, e[i]*(1+TARGET/100); break
        if a is None:
            if not np.isnan(ot1[i]) and ret_t1[i] > 0:
                a, hm, px = 1, t1, ot1[i]
            else:
                for h in bet_hms:
                    v = ph[pts.HCOL[h]]
                    if not np.isnan(v) and v >= TARGET:
                        a, hm, px = 1, h, e[i]*(1+TARGET/100); break
                if a is None and not np.isnan(ot2[i]):
                    a, hm, px = 1, t2, ot2[i]
        if a is not None:
            so_hm[i], so_px[i], long_px[i], valid[i] = hm, px, px, True
    vi = np.where(valid)[0]
    cover_opens = opens[np.ix_(vi, [pts.HCOL[h] for h in COVER_HMS])]   # (n, 12)
    return dict(e=e[vi], sh=sh[vi], cap=cap[vi], so_hm=so_hm[vi], so_px=so_px[vi],
                long_pnl=sh[vi]*(long_px[vi]-e[vi]), cover=cover_opens)


def metrics_from_short(D, short_pnl, label_dict):
    long_pnl = D["long_pnl"]; cap = D["cap"]; sh = D["sh"]; so = D["so_px"]
    combined = long_pnl + short_pnl
    short_notl = sh * so
    gross = combined
    netA = combined - LONG_023*cap - SHORT_RATE*short_notl
    netB = combined - LONG_038*cap - SHORT_RATE*short_notl
    def tr(p): return round(p.sum()/BASE_POOL*100, 4)
    return {**label_dict, "n_trades": len(combined),
            "gross_total_return_pct": tr(gross),
            "netA_total_return_pct": tr(netA), "netB_total_return_pct": tr(netB),
            "win_rate_pct": round((gross > 0).mean()*100, 2),
            "avg_return_per_trade_pct": round((gross/cap*100).mean(), 4),
            "median_return_per_trade_pct": round((gross/cap*100).median() if hasattr(gross/cap, 'median')
                                                 else np.median(gross/cap*100), 4),
            "total_short_pnl_inr": round(short_pnl.sum(), 0)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D = build()
    sh, so, so_hm, cover = D["sh"], D["so_px"], D["so_hm"], D["cover"]
    n = len(sh)
    print(f"Shorted trades: {n:,}")

    # eligibility per cover time (cover hm must be after short-open hm)
    elig = np.array([[COVER_HMS[k] > so_hm[i] for k in range(NC)] for i in range(n)])
    print("  ineligible per cover time (cover<=short-open):",
          {CLABEL[k]: int((~elig[:, k]).sum()) for k in range(NC)})

    # ── Scenario 1: single cover ──
    rows = []
    for k, hm in enumerate(COVER_HMS):
        cpx = cover[:, k]
        ok = elig[:, k] & ~np.isnan(cpx)
        sp = np.where(ok, sh*(so - cpx), np.nan)
        # metrics over covered trades only
        Dk = {kk: (vv[ok] if isinstance(vv, np.ndarray) and vv.ndim == 1 else vv) for kk, vv in D.items()}
        Dk = {"long_pnl": D["long_pnl"][ok], "cap": D["cap"][ok], "sh": D["sh"][ok], "so_px": D["so_px"][ok]}
        m = metrics_from_short(Dk, sp[ok], {"cover_time": CLABEL[k]})
        m["n_eligible"] = int(ok.sum())
        rows.append(m)
    s1 = pd.DataFrame(rows).sort_values("gross_total_return_pct", ascending=False).reset_index(drop=True)

    # ── Scenario 2: split cover (t1<t2), condition on short profitability at t1 ──
    rows2 = []
    for i1 in range(NC):
        o_t1 = cover[:, i1]; sret_t1 = (so - o_t1) / so * 100
        for i2 in range(i1+1, NC):
            o_t2 = cover[:, i2]
            profitable = sret_t1 > 0
            cover_px = np.where(profitable, o_t1, o_t2)
            ok = (np.array([COVER_HMS[i1] > h for h in so_hm])) & ~np.isnan(cover_px)
            sp = sh*(so - cover_px)
            Dk = {"long_pnl": D["long_pnl"][ok], "cap": D["cap"][ok], "sh": D["sh"][ok], "so_px": D["so_px"][ok]}
            m = metrics_from_short(Dk, sp[ok], {"cover_t1": CLABEL[i1], "cover_t2": CLABEL[i2]})
            rows2.append(m)
    s2 = pd.DataFrame(rows2).sort_values("gross_total_return_pct", ascending=False).reset_index(drop=True)
    s2.insert(0, "rank", range(1, len(s2)+1))

    with pd.ExcelWriter(OUTDIR / "double_down_cover_sweep.xlsx", engine="openpyxl") as w:
        s1.to_excel(w, sheet_name="full_cover_sweep", index=False)
        s2.to_excel(w, sheet_name="split_cover_sweep", index=False)

    # heatmap scenario 2
    pos = {lab: k for k, lab in enumerate(CLABEL)}
    grid = np.full((NC, NC), np.nan)
    for _, r in s2.iterrows():
        grid[pos[r["cover_t1"]], pos[r["cover_t2"]]] = r["gross_total_return_pct"]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(NC)); ax.set_xticklabels(CLABEL, rotation=90, fontsize=8)
    ax.set_yticks(range(NC)); ax.set_yticklabels(CLABEL, fontsize=8)
    ax.set_xlabel("cover t2"); ax.set_ylabel("cover t1")
    ax.set_title("Double-down split-cover — gross total_return_fixedbase_pct (t1<t2)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    br, bc = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(bc, br, "r*", markersize=16, markeredgecolor="white")
    fig.tight_layout(); fig.savefig(OUTDIR / "split_cover_heatmap.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 210)
    base3pm = {"gross": 708.6810, "netA": 506.1139, "netB": 414.2884}
    print("\n" + "=" * 110 + "\nSCENARIO 1 — FULL COVER (single square-off time)\n" + "=" * 110)
    print(s1.to_string(index=False))
    print("\n" + "=" * 110 + "\nSCENARIO 2 — SPLIT COVER: TOP 10 (t1<t2)\n" + "=" * 110)
    show = ["rank", "cover_t1", "cover_t2", "n_trades", "gross_total_return_pct",
            "netA_total_return_pct", "netB_total_return_pct", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct", "total_short_pnl_inr"]
    print(s2[show].head(10).to_string(index=False))
    print("\n" + "=" * 110 + "\nBASELINE (cover all @3pm, current double-down) & BEST\n" + "=" * 110)
    print(f"  3pm-cover baseline : gross {base3pm['gross']} | net_A {base3pm['netA']} | net_B {base3pm['netB']}")
    b1 = s1.iloc[0]; b2 = s2.iloc[0]
    print(f"  best FULL cover    : {b1['cover_time']} -> gross {b1['gross_total_return_pct']} | "
          f"net_A {b1['netA_total_return_pct']} | net_B {b1['netB_total_return_pct']}")
    print(f"  best SPLIT cover   : {b2['cover_t1']}/{b2['cover_t2']} -> gross {b2['gross_total_return_pct']} | "
          f"net_A {b2['netA_total_return_pct']} | net_B {b2['netB_total_return_pct']}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
