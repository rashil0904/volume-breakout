# -*- coding: utf-8 -*-
"""make_pead_strategy_doc.py — builds results/PEAD_Strategy_Documentation.docx: a detailed plain-English
+ technical explanation of the full PEAD strategy (concept, data pipeline, signal & exit logic, the three
finalized books, parameter methodology, data-quality audit, results, and limitations).
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).resolve().parent.parent / "results" / "PEAD_Strategy_Documentation.docx"

C_DARK = RGBColor(0x1F, 0x49, 0x7D); C_MID = RGBColor(0x2E, 0x74, 0xB5)
C_GREEN = RGBColor(0x17, 0x5C, 0x2D); C_RED = RGBColor(0xC0, 0x00, 0x00)
C_ORANGE = RGBColor(0xC5, 0x5A, 0x11); C_GREY = RGBColor(0x59, 0x59, 0x59); C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FILL_HDR = "1F497D"; FILL_SUB = "D9E1F2"; FILL_CAVEAT = "FCE4D6"; FILL_KEY = "E2EFDA"; FILL_CODE = "F2F2F2"


def shade(cell, hexc):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), hexc); tcPr.append(shd)


def doc_setup(d):
    st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)


def H1(d, text, n=None):
    p = d.add_paragraph(); p.space_before = Pt(14)
    r = p.add_run((f"{n}.  " if n else "") + text); r.bold = True; r.font.size = Pt(16); r.font.color.rgb = C_DARK
    pб = p.paragraph_format; pб.space_before = Pt(16); pб.space_after = Pt(4)
    _bottom_border(p)
    return p


def H2(d, text):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = True; r.font.size = Pt(12.5); r.font.color.rgb = C_MID
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(2)
    return p


def _bottom_border(p):
    pPr = p._p.get_or_add_pPr(); pbdr = OxmlElement("w:pBdr"); bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6"); bottom.set(qn("w:space"), "2"); bottom.set(qn("w:color"), "1F497D")
    pbdr.append(bottom); pPr.append(pbdr)


def P(d, text, size=10.5, color=None, italic=False, bold=False):
    p = d.add_paragraph(); r = p.add_run(text); r.font.size = Pt(size); r.italic = italic; r.bold = bold
    if color:
        r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(6)
    return p


def RUNS(d, parts):
    """parts = list of (text, bold, color)"""
    p = d.add_paragraph()
    for text, bold, color in parts:
        r = p.add_run(text); r.bold = bold; r.font.size = Pt(10.5)
        if color:
            r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(6)
    return p


def BULLET(d, text, bold_lead=None):
    p = d.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True; r.font.size = Pt(10.5)
    r = p.add_run(text); r.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(2)
    return p


def FORMULA(d, text):
    t = d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.cell(0, 0); shade(c, FILL_CODE)
    p = c.paragraphs[0]; r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(10); r.font.color.rgb = C_ORANGE
    return t


def CALLOUT(d, title, text, fill=FILL_CAVEAT, tcolor=C_RED):
    t = d.add_table(rows=1, cols=1)
    c = t.cell(0, 0); shade(c, fill)
    p = c.paragraphs[0]; rr = p.add_run(title + "  "); rr.bold = True; rr.font.size = Pt(10.5); rr.font.color.rgb = tcolor
    r = p.add_run(text); r.font.size = Pt(10.5)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def TABLE(d, headers, rows, widths=None):
    t = d.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]; shade(c, FILL_HDR)
        p = c.paragraphs[0]; r = p.add_run(str(h)); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = C_WHITE
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            if ri % 2 == 1:
                shade(cells[j], "F2F6FC")
            p = cells[j].paragraphs[0]; r = p.add_run(str(v)); r.font.size = Pt(9.5)
            if j == 0:
                r.bold = True
    if widths:
        for j, wd in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(wd)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def build():
    d = Document(); doc_setup(d)

    # ---- Title ----
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Post-Earnings Announcement Drift (PEAD)"); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = C_DARK
    p2 = d.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p2.add_run("A Systematic Indian-Equity Earnings-Drift Strategy — Full Documentation"); r.font.size = Pt(13); r.font.color.rgb = C_MID
    p3 = d.add_paragraph(); p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p3.add_run("Three finalized books · F&O Long / F&O Short / Non-F&O Long · NSE cash & F&O · 2022–2026"); r.italic = True; r.font.size = Pt(10); r.font.color.rgb = C_GREY
    p4 = d.add_paragraph(); p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p4.add_run("Prepared 16 August 2026"); r.font.size = Pt(9); r.font.color.rgb = C_GREY
    d.add_paragraph()

    # ---- 1 Executive Summary ----
    H1(d, "Executive Summary", 1)
    P(d, "This strategy monetizes Post-Earnings Announcement Drift (PEAD): the well-documented tendency of a "
         "stock to keep drifting in the direction of its earnings-day price reaction for days to weeks after "
         "the result is announced. We measure each stock's reaction to its quarterly results (using BSE "
         "announcement timestamps and NSE prices), take a position the same or next session, and hold for a "
         "fixed window with a stop-loss and (for some books) a profit target.")
    P(d, "The strategy is deployed as three independent 'books', each with its own finalized parameters:")
    TABLE(d,
          ["Book", "Universe", "Signal", "Hold", "Stop", "Target"],
          [["F&O Long", "NSE F&O stocks", "reaction ≥ +4%", "T+15", "close-basis @ T0 low", "none"],
           ["F&O Short", "NSE F&O stocks", "reaction ≤ −4%", "T+3", "close-basis @ T0 high", "8%"],
           ["Non-F&O Long", "Nifty 500 ex-F&O", "reaction ≥ +7%", "T+10", "close-basis @ T0 low", "17%"]],
          widths=[1.1, 1.5, 1.2, 0.7, 1.5, 0.7])
    P(d, "Headline results (2022–2026, flat ₹50,000 per trade, net of 0.20% round-trip cost):")
    TABLE(d,
          ["Book", "Trades", "Total PnL", "Win %", "Avg / trade"],
          [["F&O Long", "470", "₹2,73,946", "47.9", "₹583"],
           ["F&O Short", "465", "₹1,05,810", "55.7", "₹228"],
           ["Non-F&O Long", "270", "₹2,63,373", "51.9", "₹975"],
           ["TOTAL", "1,205", "₹6,43,129", "51.8", "₹534"]],
          widths=[1.4, 1.0, 1.3, 0.9, 1.1])
    RUNS(d, [("Out-of-sample check: ", True, C_DARK),
             ("of the ₹6.43 lakh total, ₹5.51 lakh came from the in-sample period (result date in 2022–24) and "
              "₹0.92 lakh from out-of-sample (2025+). The combined book stays positive out-of-sample, and each "
              "book's parameters were chosen for out-of-sample robustness, not in-sample peak.", False, None)])

    # ---- 2 Concept ----
    H1(d, "The Concept — Why PEAD Exists", 2)
    P(d, "When a company reports earnings, the market often does not fully price the surprise on day one. "
         "Investors under-react to new information: analysts revise slowly, institutions accumulate/distribute "
         "over days, and attention is limited. The result is a predictable 'drift' — prices that jumped on "
         "good results tend to keep rising, and prices that fell on bad results tend to keep falling, for a "
         "period after the event. This is one of the most robust anomalies in the academic literature.")
    RUNS(d, [("Our operational definition of the 'surprise' is not the accounting number but the ", False, None),
             ("price reaction itself", True, C_DARK),
             (" — how far the stock moved on its results relative to where it was trading before. A large "
              "favourable reaction is our proxy for a positive surprise (go long); a large adverse reaction is "
              "our proxy for a negative surprise (go short, F&O only).", False, None)])
    CALLOUT(d, "Why price-reaction, not EPS surprise?",
            "Consensus-EPS data for the full NSE universe is expensive and patchy in India. The price reaction "
            "is directly observable, already blends all released information (revenue, margins, guidance), and "
            "is what we actually trade. The trade-off: some large reactions are driven by non-earnings news on "
            "the same day — handled by our data-quality guards (Section 9).", fill=FILL_SUB, tcolor=C_DARK)

    # ---- 3 Data ----
    H1(d, "Data & Pipeline", 3)
    H2(d, "3.1  Price data")
    BULLET(d, "Daily OHLCV for 1,609 NSE symbols, 3 Jan 2022 – 31 Jul 2026, sourced via the Upstox historical API.", "Source & coverage: ")
    BULLET(d, "This universe already spans all Nifty 500 constituents plus the F&O list, so no additional price pull was needed for the expansion.", "Breadth: ")
    H2(d, "3.2  VWAP-close (the reaction reference)")
    P(d, "The reaction is measured against the previous day's VWAP-close rather than the ordinary close, to "
         "match how the NSE official closing price is computed and to damp last-tick noise:")
    FORMULA(d, "VWAP-close(day) = Σ(typical_price × volume) / Σ(volume)  over the 15:00–15:29 one-minute candles,\n"
               "where typical_price = (High + Low + Close) / 3   [built from 1-minute intraday data]")
    H2(d, "3.3  Earnings announcements (BSE)")
    BULLET(d, "BSE Corporate Announcements API, category = 'Result'. The filing timestamp (DT_TM) gives the exact date and time the result hit the exchange.", "Source: ")
    BULLET(d, "NSE symbol → BSE numeric scrip code via the BSE scrip master (by symbol and by ISIN, ISIN sourced from the Upstox instrument master).", "Symbol mapping: ")
    BULLET(d, "Pure board-meeting notices, related-party-transaction filings and SEBI Reg-23(9) disclosures are excluded; genuine result declarations (result / financial / audited / unaudited / Reg-33 / UFR / 'quarter ended') are kept. The earliest genuine filing per (stock, date) is used.", "Declaration vs notice: ")
    BULLET(d, "Each result is classified by filing time — pre-market (<9:15), during-market (9:15–15:30), post-market (>15:30) — which drives when we can first act (Section 5).", "Session: ")
    H2(d, "3.4  Reconciliation of the F&O announcement set")
    P(d, "The F&O announcement dataset was independently re-pulled and reconciled against a fresh BSE download: "
         "3,611 of 3,624 events matched exactly (99.6%). Six events where the original build had latched onto "
         "the wrong filing (a related-party or 'meeting update' filing instead of the results) were corrected "
         "against source — POLYCAB, RVNL, YESBANK, INDIGO (wrong date/session) and GAIL, CONCOR (wrong "
         "filing, right day). The reconciled set is the baseline used everywhere below.")

    # ---- 4 Universe ----
    H1(d, "Universe Definition", 4)
    BULLET(d, "F&O universe — 208 NSE stocks currently in the equity-derivatives segment. These trade both long and short.", "")
    BULLET(d, "Nifty 500 — the current 500 constituents. The 292 that are NOT F&O-eligible form the non-F&O book and trade LONG ONLY.", "")
    RUNS(d, [("Why long-only for non-F&O: ", True, C_DARK),
             ("Indian cash equity cannot be shorted overnight, and these names have no single-stock futures to "
              "carry a multi-day short. Short signals are therefore never generated for the non-F&O universe — "
              "they are suppressed at signal creation, not filtered out afterwards.", False, None)])
    CALLOUT(d, "Point-in-time caveat.",
            "Membership and F&O-eligibility are a CURRENT snapshot. A long-only backtest on today's Nifty 500 "
            "carries survivorship bias — names that dropped out of the index (typically poor performers) are "
            "absent, which inflates long returns. Point-in-time membership is not reconstructed; treat the "
            "non-F&O absolute returns as optimistic.")

    # ---- 5 Signal ----
    H1(d, "Signal Construction", 5)
    H2(d, "5.1  Reaction return")
    FORMULA(d, "reaction_return %  =  (T0_close − prev_day_VWAP_close) / prev_day_VWAP_close × 100")
    P(d, "A long signal fires when reaction_return ≥ +K%; a short signal (F&O only) when reaction_return ≤ −K%. "
         "K is 4% for both F&O books and 7% for non-F&O (Section 8).")
    H2(d, "5.2  Session mapping → T0 (the reaction day we act on)")
    TABLE(d, ["Filing session", "T0 (reaction/entry day)", "Rationale"],
          [["Pre-market (<9:15)", "Same trading day", "Result is public before the open"],
           ["During-market (9:15–15:30)", "Next trading day", "Avoid acting mid-session on partial information"],
           ["Post-market (>15:30)", "Next trading day", "Result lands after the close"]],
          widths=[1.7, 1.9, 2.4])
    RUNS(d, [("Entry price = T0 close.", True, C_DARK),
             (" We measure the reaction using T0's close and enter at that same close (a marketable on-close "
              "proxy). The stop level is set from T0's own low (longs) or high (shorts).", False, None)])

    # ---- 6 Exits ----
    H1(d, "Exit Mechanics", 6)
    P(d, "Every open trade is evaluated day-by-day from T+1. Three exit types, checked in this order each day:")
    TABLE(d, ["Exit", "Trigger", "Fill"],
          [["Profit target", "Intraday high (long) / low (short) touches the target level", "At the target price (a gap through the level fills at the open)"],
           ["Stop-loss (close-basis)", "A later day's CLOSE crosses below T0 low (long) / above T0 high (short)", "At that breaching close"],
           ["Holding-period-end", "Neither of the above fires within T+n", "At the T+n close"]],
          widths=[1.5, 3.0, 2.6])
    RUNS(d, [("Same-day ordering: ", True, C_DARK),
             ("the target is an intraday touch and is therefore evaluated BEFORE the close on which the stop is "
              "judged even exists — so on a day that both hits the target intraday and closes through the stop, "
              "the target fires first. (The 'stop takes precedence' convention only applies when both legs are "
              "close-based, which is not the case here.)", False, None)])
    CALLOUT(d, "Why a close-basis stop?",
            "A close-basis stop (act only if the day CLOSES beyond T0's extreme) avoids the ~30 intraday "
            "'whipsaw' stops per book where price briefly pierces the level and recovers by the close. On F&O "
            "longs this measurably outperformed an intraday-touch stop.", fill=FILL_SUB, tcolor=C_DARK)

    # ---- 7 Methodology ----
    H1(d, "Parameter Methodology & Key Findings", 7)
    P(d, "Every parameter (threshold K, holding period, target) was chosen by systematic sweeps with two "
         "guardrails that run through the whole project:")
    BULLET(d, "Overfit caution — we select a STABLE NEIGHBOURING REGION of good combinations, never a single best cell. A parameter that only works in isolation is treated as noise.", "(1) ")
    BULLET(d, "In-sample / out-of-sample split — fit and compare on 2022–2024, then confirm the choice holds on 2025+. Parameters that look great in-sample but collapse out-of-sample are rejected.", "(2) ")
    H2(d, "7.1  F&O Long — holding × target grid")
    P(d, "Sweeping hold T+10…T+15 × target 5–20% showed the in-sample surface peaks at long holds and high "
         "targets, but that region's out-of-sample per-trade return collapses ~89% — a classic overfit. The "
         "robust region (best on the minimum of in- and out-of-sample) is T+11–T+13 with an 8–10% target.")
    RUNS(d, [("A key reversal: ", True, C_RED),
             ("an earlier study concluded 'F&O longs are best with no target'. On the corrected full dataset "
              "that is an in-sample artefact — with NO target, F&O longs have a NEGATIVE out-of-sample total in "
              "every holding period. In 2025+ an untargeted long gives its drift back. The finalized F&O-Long "
              "book still runs T+15 no-target for simplicity and maximum drift capture, but users should know a "
              "moderate target materially improves out-of-sample stability.", False, None)])
    H2(d, "7.2  MFE — is longer holding real drift or give-back?")
    P(d, "Max-Favourable-Excursion analysis (the peak unrealised gain over T+1…T+15) answered this directly for "
         "F&O longs: average MFE is +8.1%, and the day the peak lands is BIMODAL — a spike at T+1 (immediate "
         "continuation) and a second spike at T+15 (~40% of trades peak in T+11–15, i.e. the drift is still "
         "running at the window edge). So the drift is real and justifies long holds. BUT a no-target hold "
         "captures only 13–19% of that peak — the rest is given back — which is precisely why a target does the "
         "real work, especially out-of-sample. The bimodality hints that a trailing stop could beat a fixed "
         "target (a documented next step).")
    H2(d, "7.3  Non-F&O Long — 3-D sweep (K × hold × target)")
    P(d, "A full 4×15×16 = 960-combination sweep over the non-F&O universe produced an economically sensible, "
         "confidence-building pattern: weak reactions (K=4%) fade fast and are best captured with short holds "
         "(T+3–5); strong reactions (K=6–7%) drift longer and are best held T+9–T+12. The most out-of-sample-"
         "robust single point is K≥7%, T+10, 17% target — out-of-sample +1.65% per trade, holding at only ~36% "
         "degradation from in-sample (vs 89% for the overfit F&O-long corner).")

    # ---- 8 The three books ----
    H1(d, "The Three Finalized Books", 8)
    for name, cfg, why in [
        ("F&O LONG", "K = +4% · hold T+15 · close-basis SL @ T0 low · NO target",
         "Longs are the engine — trend-following behaviour with a fat right tail. T+15 maximises drift capture; "
         "a target caps the tail in-sample but improves out-of-sample stability (see 7.1)."),
        ("F&O SHORT", "K = −4% · hold T+3 · close-basis SL @ T0 high · 8% target",
         "Shorts are thinner and do not trend as cleanly, so the book takes profit quickly: a short T+3 hold and "
         "a fixed 8% target, which the sweep found optimal. Multi-day shorts are a single-stock-futures proxy."),
        ("NON-F&O LONG", "K = +7% · hold T+10 · close-basis SL @ T0 low · 17% target",
         "Mid/small-caps drift more strongly but need a stronger reaction filter (K=7%) to isolate clean "
         "surprises and a higher target (17%) to bank their larger moves. Long-only by construction."),
    ]:
        H2(d, name)
        FORMULA(d, cfg)
        P(d, why, size=10)

    # ---- 9 Data quality ----
    H1(d, "Data Quality & Audit", 9)
    CALLOUT(d, "Bug found and fixed — the date-parse defect.",
            "The original analysis scripts parsed the DD-MM-YYYY announcement timestamp WITHOUT a format, so "
            "pandas read it as MM-DD and silently dropped every event dated on the 13th–31st of a month — 63% "
            "of all earnings dates. The 'finalized' F&O study had unknowingly run on ~37% of the data. All "
            "current results use the corrected parse on the FULL event set (this is why F&O long trade counts "
            "roughly tripled versus the earlier numbers).")
    P(d, "A trade-genuineness audit then cross-checked every trade against the raw daily data. 1,564 of 1,567 "
         "trades were genuine (99.8%). Three were removed:")
    BULLET(d, "VEDL F&O short — a −64.7% 'reaction' that was actually the Vedanta demerger (unadjusted corporate action), not an earnings move.", "")
    BULLET(d, "FORCEMOT F&O long ×2 — a 112-day gap in the daily series caused two quarters to resolve to the same entry bar (a double-count) with a stale reference producing a fake +31% reaction.", "")
    P(d, "Three permanent guards now run inside every backtest build:")
    TABLE(d, ["Guard", "Removes"],
          [["Skip if prev-VWAP reference > 6 days stale", "Data-gap artefacts (e.g. FORCEMOT)"],
           ["De-duplicate by (symbol, T0 date)", "Double-counted events sharing an entry bar"],
           ["Skip if |reaction| > 50%", "Unadjusted splits / demergers (e.g. VEDL)"]],
          widths=[3.4, 3.4])
    P(d, "The non-F&O long book had zero artefacts — its conclusions were unaffected by the cleanup.", italic=True, color=C_GREY)

    # ---- 10 Sizing ----
    H1(d, "Position Sizing & Costs", 10)
    BULLET(d, "Flat ₹50,000 nominal per trade, sized independently (not a shared/capped pool). Trade PnL (₹) = net return % × ₹50,000 ÷ 100.", "Sizing: ")
    BULLET(d, "0.20% round-trip (≈ ₹100 per leg on ₹50k) is deducted from every trade — brokerage, STT, exchange and slippage proxy.", "Cost: ")
    BULLET(d, "Total PnL is an ADDITIVE sum of independent trades — it does not compound, and it assumes capital is always available for every concurrent signal.", "Accounting: ")

    # ---- 11 Results ----
    H1(d, "Results", 11)
    TABLE(d, ["Metric", "TOTAL", "F&O Long", "F&O Short", "Non-F&O Long"],
          [["Trades", "1,205", "470", "465", "270"],
           ["Win rate %", "51.8", "47.9", "55.7", "51.9"],
           ["Total PnL (₹)", "6,43,129", "2,73,946", "1,05,810", "2,63,373"],
           ["Avg / trade (₹)", "534", "583", "228", "975"],
           ["Avg hold (days)", "7.5", "≈13", "≈3", "≈8"],
           ["In-sample PnL (₹)", "5,50,862", "—", "—", "—"],
           ["Out-of-sample PnL (₹)", "92,267", "—", "—", "—"]],
          widths=[1.7, 1.2, 1.1, 1.1, 1.3])
    P(d, "The full trade lists, per-book summary metrics and an Indian-FY quarterly breakdown are in the "
         "companion workbook pead_final_report.xlsx. The quarterly sheet makes the F&O-Long give-back quarters "
         "visible (e.g. FY23 Q1) — the volatility the MFE analysis predicted for a no-target long book.")

    # ---- 12 Limitations ----
    H1(d, "Limitations & Caveats", 12)
    for t, x in [
        ("Survivorship bias (non-F&O)", "current-snapshot Nifty 500; dropped names absent; long returns optimistic."),
        ("Circuit-locked fills", "some +20% reactions are upper-circuit days; you cannot actually buy at a locked-up close."),
        ("Slippage under-stated", "the flat 0.20% cost is light for illiquid non-F&O names."),
        ("Cash-equity short proxy", "F&O shorts held multiple days assume a single-stock-futures leg."),
        ("Concurrency / capital", "16–22 positions can overlap in earnings season; ₹50k each implies ₹8–11 lakh peak deployment, and the total does not compound."),
        ("Same-source announcements", "only BSE 'Result' category is read; results filed solely under 'Board Meeting' for some names can be missed."),
        ("Point-in-time membership", "index and F&O eligibility are current, not reconstructed historically."),
    ]:
        RUNS(d, [("• " + t + " — ", True, C_RED), (x, False, None)])

    # ---- 13 Reproducibility ----
    H1(d, "Reproducibility — Scripts & Outputs", 13)
    TABLE(d, ["File", "Purpose"],
          [["build_nifty500_universe.py", "Nifty 500 constituents + fno_eligible tags"],
           ["fetch_bse_results_announcements.py / build_nifty500_announcements.py", "BSE result-declaration pulls (F&O + non-F&O)"],
           ["verify_bse_announcements.py / apply_6_corrections.py", "Reconciliation + the six corrections"],
           ["pead_long_hold_target_grid.py", "F&O-long hold × target sweep (guarded)"],
           ["pead_long_mfe_corrected.py", "F&O-long MFE / give-back analysis"],
           ["pead_nonfo_3d_grid.py", "Non-F&O 3-D K × hold × target sweep"],
           ["pead_trade_audit.py", "Trade-genuineness audit"],
           ["pead_final_report.py", "Consolidated Excel report (pead_final_report.xlsx)"]],
          widths=[3.7, 3.1])

    # ---- Glossary ----
    H1(d, "Glossary", 14)
    for term, defn in [
        ("PEAD", "Post-Earnings Announcement Drift — the tendency of prices to keep moving in the reaction direction after results."),
        ("T0", "The reaction/entry day (same day for pre-market results, next day for during/post-market)."),
        ("Reaction return", "(T0 close − previous-day VWAP-close) / previous-day VWAP-close × 100."),
        ("Close-basis SL", "A stop that fires only when a day CLOSES beyond T0's low/high, filled at that close."),
        ("MFE", "Max Favourable Excursion — the largest unrealised gain reached during the holding window."),
        ("IS / OOS", "In-sample (2022–24) used to choose parameters; out-of-sample (2025+) used to confirm them."),
        ("F&O", "Futures & Options — stocks in the NSE equity-derivatives segment (can be shorted; the rest are long-only)."),
    ]:
        RUNS(d, [(term + " — ", True, C_DARK), (defn, False, None)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.save(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    build()
