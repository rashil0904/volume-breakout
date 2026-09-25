# -*- coding: utf-8 -*-
"""
quarterly_compounding_breakdown.py
==================================
Standalone audit/presentation of the COMPOUNDING mechanism (quarterly AND yearly),
using FULL COMPOUNDING: pool & per-trade allocation carry the prior period's ENDING value
every period — scaling UP after a gain and DOWN after a loss (the loss is baked into the
pool). No reset to base, and no flat carry-forward. final_performance_report.xlsx uses the
same rule, so the two files agree on the mechanism.

Reuses the exact same trade set + scaling helper from final_performance_report.
GROSS + NET figures (separate self-contained chains).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "final_report"
XLSX = OUTDIR / "quarterly_compounding_breakdown.xlsx"
BP, BA = fpr.BASE_POOL, fpr.BASE_ALLOC


def qlabel(q):
    return f"{q[:4]}-Q{q[5:]}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = fpr.build_trades()

    # ══════════ QUARTERLY (full compounding — up on gain, down on loss) ══════════
    quarters = sorted(T["quarter"].unique())
    pool, alloc = BP, BA
    prev_ret = prev_pool = prev_q = None
    qrows = []
    for i, q in enumerate(quarters):
        starting_pool = BP if i == 0 else prev_pool
        if i == 0:
            trig, scale = "N/A", None
        else:
            trig, scale = (prev_ret > 0), 1 + prev_ret / 100   # always compound (down if <=0)
        g = T[T["quarter"] == q]; n = len(g)
        cg, _ = fpr.scaled(g, alloc)
        ret_q = cg / pool * 100
        ending = pool + cg
        qrows.append({
            "quarter": qlabel(q),
            "prior_quarter": "N/A — first quarter" if i == 0 else qlabel(prev_q),
            "prior_quarter_compounded_return_pct": "" if i == 0 else round(prev_ret, 4),
            "compounding_trigger_met": trig,
            "starting_pool_inr": round(starting_pool, 0),
            "scale_factor_applied": "" if scale is None else round(scale, 6),
            "pool_for_this_quarter_inr": round(pool, 0),
            "per_trade_allocation_used_inr": round(alloc, 0),
            "n_trades_this_quarter": n,
            "gross_pnl_this_quarter_inr": round(cg, 0),
            "compounded_return_pct_this_quarter": round(ret_q, 4),
            "ending_pool_value_inr": round(ending, 0),
            "cumulative_growth_multiple": round(ending / BP, 4),
        })
        prev_pool, prev_ret, prev_q = pool, ret_q, q
        f = 1 + ret_q / 100; pool *= f; alloc *= f     # always compound ending pool (up or down)
    qdetail = pd.DataFrame(qrows)

    # ══════════ QUARTERLY NET (full compounding; driven by prior quarter NET return) ══════════
    pool, alloc = BP, BA
    prev_ret = prev_pool = prev_q = None
    qnrows = []
    for i, q in enumerate(quarters):
        starting_pool = BP if i == 0 else prev_pool
        if i == 0:
            trig, scale = "N/A", None
        else:
            trig, scale = (prev_ret > 0), 1 + prev_ret / 100   # always compound (down if <=0)
        g = T[T["quarter"] == q]; n = len(g)
        _, cn = fpr.scaled(g, alloc)                     # NET scaled pnl
        ret_q = cn / pool * 100
        ending = pool + cn
        qnrows.append({
            "quarter": qlabel(q),
            "prior_quarter": "N/A — first quarter" if i == 0 else qlabel(prev_q),
            "prior_quarter_net_compounded_return_pct": "" if i == 0 else round(prev_ret, 4),
            "compounding_trigger_met": trig,
            "starting_pool_inr": round(starting_pool, 0),
            "scale_factor_applied": "" if scale is None else round(scale, 6),
            "pool_for_this_quarter_inr": round(pool, 0),
            "per_trade_allocation_used_inr": round(alloc, 0),
            "n_trades_this_quarter": n,
            "net_pnl_this_quarter_inr": round(cn, 0),
            "compounded_return_pct_this_quarter": round(ret_q, 4),
            "ending_pool_value_inr": round(ending, 0),
            "cumulative_growth_multiple": round(ending / BP, 4),
        })
        prev_pool, prev_ret, prev_q = pool, ret_q, q
        f = 1 + ret_q / 100; pool *= f; alloc *= f     # always compound ending pool (up or down)
    qdetail_net = pd.DataFrame(qnrows)

    # ══════════ YEARLY (full compounding; driven by prior year two half-year fixed-base returns) ══════════
    halfy = {k: fpr.pmetrics(g)["gross_total_return_fixedbase_pct"] for k, g in T.groupby("half_year")}
    years = sorted(T["year"].unique())
    pool, alloc = BP, BA
    prev_pool = prev_alloc = None
    yrows = []
    for i, y in enumerate(years):
        starting_pool = BP if i == 0 else prev_pool
        if i == 0:
            trig, scale, avg_half, h1, h2 = "N/A", None, None, None, None
        else:
            py = years[i - 1]
            h1, h2 = halfy.get(f"{py}H1"), halfy.get(f"{py}H2")
            hs = [h for h in [h1, h2] if h is not None]
            avg_half = float(np.mean(hs)) if hs else 0.0
            trig, scale = (avg_half > 0), 1 + avg_half / 100   # always compound (down if <=0)
            pool = prev_pool * scale; alloc = prev_alloc * scale
        g = T[T["year"] == y]; n = len(g)
        cg, _ = fpr.scaled(g, alloc)
        ret_y = cg / pool * 100
        ending = pool + cg
        yrows.append({
            "year": int(y),
            "prior_year": "N/A — first year" if i == 0 else str(int(years[i - 1])),
            "prior_year_H1_fixedbase_return_pct": "" if i == 0 or h1 is None else round(h1, 4),
            "prior_year_H2_fixedbase_return_pct": "" if i == 0 or h2 is None else round(h2, 4),
            "avg_prior_year_half_return_pct": "" if i == 0 else round(avg_half, 4),
            "compounding_trigger_met": trig,
            "starting_pool_inr": round(starting_pool, 0),
            "scale_factor_applied": "" if scale is None else round(scale, 6),
            "pool_for_this_year_inr": round(pool, 0),
            "per_trade_allocation_used_inr": round(alloc, 0),
            "n_trades_this_year": n,
            "gross_pnl_this_year_inr": round(cg, 0),
            "compounded_return_pct_this_year": round(ret_y, 4),
            "ending_pool_value_inr": round(ending, 0),
            "cumulative_growth_multiple": round(ending / BP, 4),
        })
        prev_pool, prev_alloc = pool, alloc
    ydetail = pd.DataFrame(yrows)

    # ══════════ YEARLY NET (full compounding; driven by prior year two NET half-year returns) ══════════
    halfy_net = {k: fpr.pmetrics(g)["net_total_return_fixedbase_pct"] for k, g in T.groupby("half_year")}
    pool, alloc = BP, BA
    prev_pool = prev_alloc = None
    ynrows = []
    for i, y in enumerate(years):
        starting_pool = BP if i == 0 else prev_pool
        if i == 0:
            trig, scale, avg_half, h1, h2 = "N/A", None, None, None, None
        else:
            py = years[i - 1]
            h1, h2 = halfy_net.get(f"{py}H1"), halfy_net.get(f"{py}H2")
            hs = [h for h in [h1, h2] if h is not None]
            avg_half = float(np.mean(hs)) if hs else 0.0
            trig, scale = (avg_half > 0), 1 + avg_half / 100   # always compound (down if <=0)
            pool = prev_pool * scale; alloc = prev_alloc * scale
        g = T[T["year"] == y]; n = len(g)
        _, cn = fpr.scaled(g, alloc)                     # NET scaled pnl
        ret_y = cn / pool * 100
        ending = pool + cn
        ynrows.append({
            "year": int(y),
            "prior_year": "N/A — first year" if i == 0 else str(int(years[i - 1])),
            "prior_year_H1_net_fixedbase_return_pct": "" if i == 0 or h1 is None else round(h1, 4),
            "prior_year_H2_net_fixedbase_return_pct": "" if i == 0 or h2 is None else round(h2, 4),
            "avg_prior_year_half_return_pct": "" if i == 0 else round(avg_half, 4),
            "compounding_trigger_met": trig,
            "starting_pool_inr": round(starting_pool, 0),
            "scale_factor_applied": "" if scale is None else round(scale, 6),
            "pool_for_this_year_inr": round(pool, 0),
            "per_trade_allocation_used_inr": round(alloc, 0),
            "n_trades_this_year": n,
            "net_pnl_this_year_inr": round(cn, 0),
            "compounded_return_pct_this_year": round(ret_y, 4),
            "ending_pool_value_inr": round(ending, 0),
            "cumulative_growth_multiple": round(ending / BP, 4),
        })
        prev_pool, prev_alloc = pool, alloc
    ydetail_net = pd.DataFrame(ynrows)

    # ══════════ narratives ══════════
    def qnarr(r):
        q = r["quarter"]; gp = r["gross_pnl_this_quarter_inr"]; pf = r["pool_for_this_quarter_inr"]
        if str(r["prior_quarter"]).startswith("N/A"):
            return (f"{q}: First quarter — no prior quarter to compound from, so starting pool = ₹{pf:,.0f} (base), "
                    f"per-trade allocation = ₹{r['per_trade_allocation_used_inr']:,.0f} (base). {r['n_trades_this_quarter']} "
                    f"trades made ₹{gp:,.0f}. Return = {gp:,.0f}/{pf:,.0f}×100 = {r['compounded_return_pct_this_quarter']:.2f}%. "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        if r["compounding_trigger_met"] is True:
            pr, sf = r["prior_quarter_compounded_return_pct"], r["scale_factor_applied"]
            return (f"{q}: Prior quarter ({r['prior_quarter']}) returned {pr:+.2f}% (positive) → compounding triggers. "
                    f"Pool = ₹{r['starting_pool_inr']:,.0f} × {sf:.4f} = ₹{pf:,.0f}; allocation = ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                    f"{r['n_trades_this_quarter']} trades made ₹{gp:,.0f}. Return = {r['compounded_return_pct_this_quarter']:.2f}%. "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        pr, sf = r["prior_quarter_compounded_return_pct"], r["scale_factor_applied"]
        return (f"{q}: Prior quarter ({r['prior_quarter']}) returned {pr:+.2f}% (negative) → the loss is baked in; the pool "
                f"carries the prior quarter's ENDING value. Pool = ₹{r['starting_pool_inr']:,.0f} × {sf:.4f} = ₹{pf:,.0f}; "
                f"allocation = ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                f"{r['n_trades_this_quarter']} trades made ₹{gp:,.0f}. Return = {r['compounded_return_pct_this_quarter']:.2f}%. "
                f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")

    def ynarr(r):
        y = r["year"]; gp = r["gross_pnl_this_year_inr"]; pf = r["pool_for_this_year_inr"]
        if str(r["prior_year"]).startswith("N/A"):
            return (f"{y}: First year — base pool ₹{pf:,.0f}, allocation ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                    f"{r['n_trades_this_year']} trades made ₹{gp:,.0f}. Return = {r['compounded_return_pct_this_year']:.2f}%. "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        if r["compounding_trigger_met"] is True:
            return (f"{y}: Prior year ({r['prior_year']}) half-year fixed-base returns were {r['prior_year_H1_fixedbase_return_pct']:.2f}% (H1) "
                    f"and {r['prior_year_H2_fixedbase_return_pct']:.2f}% (H2), averaging {r['avg_prior_year_half_return_pct']:.2f}% (positive) → "
                    f"compounding triggers. Pool = ₹{r['starting_pool_inr']:,.0f} × {r['scale_factor_applied']:.4f} = ₹{pf:,.0f}; "
                    f"allocation = ₹{r['per_trade_allocation_used_inr']:,.0f}. {r['n_trades_this_year']} trades made ₹{gp:,.0f}. "
                    f"Return = {r['compounded_return_pct_this_year']:.2f}%. Ending pool = ₹{r['ending_pool_value_inr']:,.0f} "
                    f"(×{r['cumulative_growth_multiple']:.2f}).")
        return (f"{y}: Prior year ({r['prior_year']}) half-average = {r['avg_prior_year_half_return_pct']:.2f}% (negative) → pool scales "
                f"DOWN: ₹{r['starting_pool_inr']:,.0f} × {r['scale_factor_applied']:.4f} = ₹{pf:,.0f} / allocation ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                f"{r['n_trades_this_year']} trades made ₹{gp:,.0f}. Return = {r['compounded_return_pct_this_year']:.2f}%. "
                f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")

    qnarrative = pd.DataFrame({"quarter": qdetail["quarter"], "explanation": qdetail.apply(qnarr, axis=1)})

    def qnarr_net(r):
        q = r["quarter"]; npl = r["net_pnl_this_quarter_inr"]; pf = r["pool_for_this_quarter_inr"]
        if str(r["prior_quarter"]).startswith("N/A"):
            return (f"{q}: First quarter — base pool ₹{pf:,.0f}, allocation ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                    f"{r['n_trades_this_quarter']} trades made ₹{npl:,.0f} NET. Return = {r['compounded_return_pct_this_quarter']:.2f}% (net). "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        if r["compounding_trigger_met"] is True:
            pr, sf = r["prior_quarter_net_compounded_return_pct"], r["scale_factor_applied"]
            return (f"{q}: Prior quarter ({r['prior_quarter']}) NET return {pr:+.2f}% (positive) → compounding triggers. "
                    f"Pool = ₹{r['starting_pool_inr']:,.0f} × {sf:.4f} = ₹{pf:,.0f}; allocation = ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                    f"{r['n_trades_this_quarter']} trades made ₹{npl:,.0f} NET. Return = {r['compounded_return_pct_this_quarter']:.2f}% (net). "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        pr, sf = r["prior_quarter_net_compounded_return_pct"], r["scale_factor_applied"]
        return (f"{q}: Prior quarter ({r['prior_quarter']}) NET return {pr:+.2f}% (negative) → the loss is baked in; the pool carries "
                f"the prior quarter's ENDING value. Pool = ₹{r['starting_pool_inr']:,.0f} × {sf:.4f} = ₹{pf:,.0f} / allocation "
                f"₹{r['per_trade_allocation_used_inr']:,.0f}. {r['n_trades_this_quarter']} trades made "
                f"₹{npl:,.0f} NET. Return = {r['compounded_return_pct_this_quarter']:.2f}% (net). "
                f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
    qnarrative_net = pd.DataFrame({"quarter": qdetail_net["quarter"], "explanation": qdetail_net.apply(qnarr_net, axis=1)})
    ynarrative = pd.DataFrame({"year": ydetail["year"], "explanation": ydetail.apply(ynarr, axis=1)})

    def ynarr_net(r):
        y = r["year"]; npl = r["net_pnl_this_year_inr"]; pf = r["pool_for_this_year_inr"]
        if str(r["prior_year"]).startswith("N/A"):
            return (f"{y}: First year — base pool ₹{pf:,.0f}, allocation ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                    f"{r['n_trades_this_year']} trades made ₹{npl:,.0f} NET. Return = {r['compounded_return_pct_this_year']:.2f}% (net). "
                    f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
        if r["compounding_trigger_met"] is True:
            return (f"{y}: Prior year ({r['prior_year']}) NET half-year returns were {r['prior_year_H1_net_fixedbase_return_pct']:.2f}% (H1) "
                    f"and {r['prior_year_H2_net_fixedbase_return_pct']:.2f}% (H2), averaging {r['avg_prior_year_half_return_pct']:.2f}% (positive) → "
                    f"compounding triggers. Pool = ₹{r['starting_pool_inr']:,.0f} × {r['scale_factor_applied']:.4f} = ₹{pf:,.0f}; "
                    f"allocation = ₹{r['per_trade_allocation_used_inr']:,.0f}. {r['n_trades_this_year']} trades made ₹{npl:,.0f} NET. "
                    f"Return = {r['compounded_return_pct_this_year']:.2f}% (net). Ending pool = ₹{r['ending_pool_value_inr']:,.0f} "
                    f"(×{r['cumulative_growth_multiple']:.2f}).")
        return (f"{y}: Prior year ({r['prior_year']}) NET half-average = {r['avg_prior_year_half_return_pct']:.2f}% (negative) → pool scales "
                f"DOWN: ₹{r['starting_pool_inr']:,.0f} × {r['scale_factor_applied']:.4f} = ₹{pf:,.0f} / allocation ₹{r['per_trade_allocation_used_inr']:,.0f}. "
                f"{r['n_trades_this_year']} trades made ₹{npl:,.0f} NET. Return = {r['compounded_return_pct_this_year']:.2f}% (net). "
                f"Ending pool = ₹{r['ending_pool_value_inr']:,.0f} (×{r['cumulative_growth_multiple']:.2f}).")
    ynarrative_net = pd.DataFrame({"year": ydetail_net["year"], "explanation": ydetail_net.apply(ynarr_net, axis=1)})

    # ══════════ combined summary ══════════
    def blk(prefix, det, trig_col, pool_col, alloc_col):
        n_trig = int((det["compounding_trigger_met"] == True).sum())
        n_flat = int((det["compounding_trigger_met"] == False).sum())
        hi = det.loc[det[alloc_col].idxmax()]; lo = det.loc[det[alloc_col].idxmin()]
        keyname = "quarter" if prefix == "quarterly" else "year"
        return [
            {"metric": f"{prefix}_total_periods", "value": len(det)},
            {"metric": f"{prefix}_n_compounding_triggered", "value": n_trig},
            {"metric": f"{prefix}_n_periods_scaled_down_on_loss", "value": n_flat},
            {"metric": f"{prefix}_final_ending_pool_value_inr", "value": round(det['ending_pool_value_inr'].iloc[-1], 0)},
            {"metric": f"{prefix}_final_cumulative_growth_multiple", "value": round(det['cumulative_growth_multiple'].iloc[-1], 4)},
            {"metric": f"{prefix}_highest_per_trade_allocation_inr", "value": round(hi[alloc_col], 0)},
            {"metric": f"{prefix}_highest_per_trade_allocation_period", "value": str(hi[keyname])},
            {"metric": f"{prefix}_lowest_per_trade_allocation_inr", "value": round(lo[alloc_col], 0)},
            {"metric": f"{prefix}_lowest_per_trade_allocation_period", "value": str(lo[keyname])},
        ]
    summ = pd.DataFrame(
        [{"metric": "compounding_rule", "value": "FULL COMPOUNDING — pool & allocation carry the prior period's ENDING value every period (scale up on gains, down on losses; no reset, no flat carry)"}]
        + blk("quarterly", qdetail, "compounding_trigger_met", "pool_for_this_quarter_inr", "per_trade_allocation_used_inr")
        + blk("yearly", ydetail, "compounding_trigger_met", "pool_for_this_year_inr", "per_trade_allocation_used_inr"))
    _ntg = int((ydetail_net["compounding_trigger_met"] == True).sum())
    _nfl = int((ydetail_net["compounding_trigger_met"] == False).sum())
    _hi = ydetail_net.loc[ydetail_net["per_trade_allocation_used_inr"].idxmax()]
    _lo = ydetail_net.loc[ydetail_net["per_trade_allocation_used_inr"].idxmin()]
    summ = pd.concat([summ, pd.DataFrame([
        {"metric": "yearly_n_compounding_triggered_net", "value": _ntg},
        {"metric": "yearly_n_periods_scaled_down_on_loss_net", "value": _nfl},
        {"metric": "yearly_final_ending_pool_value_inr_net", "value": round(ydetail_net['ending_pool_value_inr'].iloc[-1], 0)},
        {"metric": "yearly_final_cumulative_growth_multiple_net", "value": round(ydetail_net['cumulative_growth_multiple'].iloc[-1], 4)},
        {"metric": "yearly_highest_per_trade_allocation_inr_net", "value": round(_hi['per_trade_allocation_used_inr'], 0)},
        {"metric": "yearly_highest_per_trade_allocation_period_net", "value": str(int(_hi['year']))},
        {"metric": "yearly_lowest_per_trade_allocation_inr_net", "value": round(_lo['per_trade_allocation_used_inr'], 0)},
        {"metric": "yearly_lowest_per_trade_allocation_period_net", "value": str(int(_lo['year']))},
    ])], ignore_index=True)
    _qtg = int((qdetail_net["compounding_trigger_met"] == True).sum())
    _qfl = int((qdetail_net["compounding_trigger_met"] == False).sum())
    _qhi = qdetail_net.loc[qdetail_net["per_trade_allocation_used_inr"].idxmax()]
    _qlo = qdetail_net.loc[qdetail_net["per_trade_allocation_used_inr"].idxmin()]
    summ = pd.concat([summ, pd.DataFrame([
        {"metric": "quarterly_n_compounding_triggered_net", "value": _qtg},
        {"metric": "quarterly_n_periods_scaled_down_on_loss_net", "value": _qfl},
        {"metric": "quarterly_final_ending_pool_value_inr_net", "value": round(qdetail_net['ending_pool_value_inr'].iloc[-1], 0)},
        {"metric": "quarterly_final_cumulative_growth_multiple_net", "value": round(qdetail_net['cumulative_growth_multiple'].iloc[-1], 4)},
        {"metric": "quarterly_highest_per_trade_allocation_inr_net", "value": round(_qhi['per_trade_allocation_used_inr'], 0)},
        {"metric": "quarterly_highest_per_trade_allocation_period_net", "value": str(_qhi['quarter'])},
        {"metric": "quarterly_lowest_per_trade_allocation_inr_net", "value": round(_qlo['per_trade_allocation_used_inr'], 0)},
        {"metric": "quarterly_lowest_per_trade_allocation_period_net", "value": str(_qlo['quarter'])},
    ])], ignore_index=True)

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        qdetail.to_excel(w, sheet_name="quarterly_compounding_detail", index=False)
        qnarrative.to_excel(w, sheet_name="quarterly_narrative", index=False)
        qdetail_net.to_excel(w, sheet_name="quarterly_compounding_detail_net", index=False)
        qnarrative_net.to_excel(w, sheet_name="quarterly_narrative_net", index=False)
        ydetail.to_excel(w, sheet_name="yearly_compounding_detail", index=False)
        ynarrative.to_excel(w, sheet_name="yearly_narrative", index=False)
        ydetail_net.to_excel(w, sheet_name="yearly_compounding_detail_net", index=False)
        ynarrative_net.to_excel(w, sheet_name="yearly_narrative_net", index=False)
        summ.to_excel(w, sheet_name="summary", index=False)
        for name, sh in w.sheets.items():
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = (
                    110 if "narrative" in name and col[0].column_letter == "B" else min(width + 2, 32))

    print("QUARTERLY  final pool:", round(qdetail['ending_pool_value_inr'].iloc[-1]),
          "x", round(qdetail['cumulative_growth_multiple'].iloc[-1], 2),
          "| triggered:", int((qdetail['compounding_trigger_met'] == True).sum()),
          "| flat:", int((qdetail['compounding_trigger_met'] == False).sum()))
    print("YEARLY     final pool:", round(ydetail['ending_pool_value_inr'].iloc[-1]),
          "x", round(ydetail['cumulative_growth_multiple'].iloc[-1], 2),
          "| triggered:", int((ydetail['compounding_trigger_met'] == True).sum()),
          "| flat:", int((ydetail['compounding_trigger_met'] == False).sum()))
    print(ydetail[["year", "avg_prior_year_half_return_pct", "scale_factor_applied",
                   "pool_for_this_year_inr", "per_trade_allocation_used_inr",
                   "compounded_return_pct_this_year", "ending_pool_value_inr",
                   "cumulative_growth_multiple"]].to_string(index=False))
    print(f"\nSaved -> {XLSX}")


if __name__ == "__main__":
    main()
