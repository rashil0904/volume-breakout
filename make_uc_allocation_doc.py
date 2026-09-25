#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_uc_allocation_doc.py
Creates results/uc_entry_and_allocation_guide.docx — a detailed, plain-English
explanation of (1) the Upper-Circuit (UC) entry criteria (Categories A / B / C) and
(2) the day-level capital-allocation sequencing used by the composite UC + double-down
strategy. Self-contained (does not import make_word_doc, which builds a different doc).
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(__file__).parent / "results" / "uc_entry_and_allocation_guide.docx"
OUT.parent.mkdir(exist_ok=True)

C_DARK = RGBColor(0x1F, 0x49, 0x7D); C_MID = RGBColor(0x2E, 0x74, 0xB5)
C_ORANGE = RGBColor(0xC5, 0x5A, 0x11); C_GREEN = RGBColor(0x17, 0x5C, 0x2D)
C_RED = RGBColor(0xC0, 0x00, 0x00); C_BLACK = RGBColor(0x10, 0x10, 0x10)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF); C_CODE = RGBColor(0x19, 0x3A, 0x64)
FONT = "Calibri"; MONO = "Courier New"


def _bg(cell, hexs):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexs)
    tcPr.append(shd)


def _bottom_border(p, color="2E74B5", sz="6"):
    pPr = p._p.get_or_add_pPr(); pBd = OxmlElement("w:pBdr"); bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), sz)
    bot.set(qn("w:space"), "1"); bot.set(qn("w:color"), color)
    pBd.append(bot); pPr.append(pBd)


def h1(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(16); p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(15); r.font.color.rgb = C_DARK; r.font.name = FONT
    _bottom_border(p); return p


def h2(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(12.5); r.font.color.rgb = C_MID; r.font.name = FONT
    return p


def h3(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(11); r.font.color.rgb = C_ORANGE; r.font.name = FONT
    return p


def body(text, bold=False, italic=False, size=10.5, color=C_BLACK, indent=0):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(3)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    r = p.add_run(text); r.bold = bold; r.italic = italic
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
    return p


def rich(parts, indent=0, space_after=3):
    """parts: list of (text, dict) where dict has bold/italic/color/mono keys."""
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    for text, st in parts:
        r = p.add_run(text)
        r.bold = st.get("bold", False); r.italic = st.get("italic", False)
        r.font.size = Pt(st.get("size", 10.5))
        r.font.name = MONO if st.get("mono") else FONT
        r.font.color.rgb = st.get("color", C_BLACK)
    return p


def formula(text, indent=1.2):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Cm(indent)
    r = p.add_run(text); r.font.name = MONO; r.font.size = Pt(9.5); r.font.color.rgb = C_CODE
    return p


def bullet(text, level=0, prefix=None, prefix_color=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.8 + level * 0.7)
    if prefix:
        r1 = p.add_run(prefix); r1.bold = True; r1.font.size = Pt(10.5); r1.font.name = FONT
        r1.font.color.rgb = prefix_color or C_BLACK
    r2 = p.add_run(text); r2.font.size = Pt(10.5); r2.font.name = FONT
    return p


def note(text, fill="FFF2CC", color=RGBColor(0x4A, 0x3B, 0x00), tag="NOTE"):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.4); p.paragraph_format.right_indent = Cm(0.4)
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"  {tag}:  " + text + "  "); r.font.size = Pt(9.5); r.font.name = FONT; r.font.color.rgb = color
    pPr = p._p.get_or_add_pPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    pPr.append(shd); return p


def table(headers, rows, widths=None, header_fill="2E74B5"):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers)); t.style = "Table Grid"
    for i, hd in enumerate(headers):
        c = t.rows[0].cells[i]; _bg(c, header_fill)
        pp = c.paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp.add_run(hd); rr.bold = True; rr.font.size = Pt(9); rr.font.color.rgb = C_WHITE; rr.font.name = FONT
    for ri, rd in enumerate(rows):
        fill = "EAF2FF" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(rd):
            c = t.rows[ri + 1].cells[ci]; _bg(c, fill)
            pp = c.paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            sval = str(val)
            rr = pp.add_run(sval); rr.font.size = Pt(9); rr.font.name = FONT
            if sval.upper() in ("NO ENTRY", "SKIP"):
                rr.bold = True; rr.font.color.rgb = C_RED
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t


# ════════════════════════════════════════════════════════════════════════════
doc = Document()
for s in doc.sections:
    s.top_margin = Cm(1.8); s.bottom_margin = Cm(1.8); s.left_margin = Cm(2.2); s.right_margin = Cm(2.2)
base = doc.styles["Normal"]; base.font.name = FONT; base.font.size = Pt(10.5)

# ── Title ──
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("UC Entry Criteria & Capital Allocation"); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = C_DARK; r.font.name = FONT
sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sp.add_run("Composite Volume-Breakout BTST Strategy  ·  Upper-Circuit Ladder + Double-Down Short")
r.italic = True; r.font.size = Pt(11); r.font.color.rgb = C_MID; r.font.name = FONT
dp = doc.add_paragraph(); dp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = dp.add_run("Detailed reference — how every qualifying stock-day is classified for entry, "
               "and how the ₹5,00,000 pool is allocated across trades each day.")
r.font.size = Pt(9.5); r.font.color.rgb = C_BLACK; r.font.name = FONT

# ── 1. Scope ──
h1("1.  What this document covers")
body("This note explains two things in detail, exactly as implemented in the backtest engine:")
bullet("how each qualifying stock-day is routed into one of three entry categories (A, B, C) based on its "
       "interaction with the day's upper circuit; and", prefix="Entry criteria — ", prefix_color=C_MID)
bullet("how the fixed ₹5,00,000 capital pool is split across the day's trades in a strict priority "
       "sequence (per-trade capital is NOT a flat ₹1,00,000 — it varies by category and by how crowded "
       "the day is).", prefix="Capital allocation — ", prefix_color=C_MID)
body("Everything else about the strategy — the qualifying filters, the next-day exits, the double-down "
     "intraday short, the 1-minute timing conventions and the cost model — is unchanged by these rules; "
     "they only decide WHICH entry method a signal uses and HOW MUCH capital it receives.")

# ── 2. Prerequisites ──
h1("2.  Prerequisites: the qualifying signal and the reference prices")
body("A stock-day only reaches the entry logic if it first passes the base filters (evaluated on 1-minute data):")
bullet("Market cap ₹1,500–₹5,000 Cr.", prefix="Size — ", prefix_color=C_MID)
bullet("Cumulative 09:15–14:59 volume today ≥ 6 × the trailing 36-day average full-day (09:15–15:29) "
       "volume.", prefix="Volume surge — ", prefix_color=C_MID)
bullet("The 15:00 price is ≥ +5% vs the previous day's VWAP-close reference.", prefix="Return — ", prefix_color=C_MID)
h3("The two reference prices every rule below is measured from")
rich([("Previous-day VWAP-close", {"bold": True}),
      (" = the volume-weighted average price of the LAST 30 one-minute candles of the prior day "
       "(15:00–15:29). This is the “prev_close” used for every percentage level below.", {})])
formula("prev_close = Σ(typical_price × volume) / Σ(volume),  over 15:00–15:29,  typical = (H+L+C)/3")
rich([("Upper Circuit (UC) level", {"bold": True}),
      (" = prev_close × 1.1995 (≈ +20%). “Hitting UC” means a 1-minute HIGH reaches this level "
       "(the stock is locked up at the circuit).", {})])
formula("UC = prev_close × 1.1995      (+19.95%, the upper circuit)\n"
        "+19% level = prev_close × 1.19   (first ladder rung)\n"
        "+17% level = prev_close × 1.17   (second ladder rung)")
note("All percentage levels (UC, +19%, +17%, +5%) are measured off the previous-day VWAP-close, for internal "
     "consistency with the +5% qualifying filter. Detection of UC hits and pullbacks is on 1-minute candles.")

# ── 3. Entry categories ──
h1("3.  Entry criteria — Categories A, B, C")
body("Each qualifying stock-day falls into EXACTLY ONE category, decided by the FIRST time it hits UC on the "
     "entry day. The three cases exist because buying a stock while it is pinned at +20% is adverse-selected "
     "(the ones still locked tend to be the ones that never come back down), so early-circuit stocks are "
     "entered more carefully via a pullback ladder.")
table(["First UC hit", "Category", "Entry method"],
      [["Before 15:00", "A", "Pullback ladder (+19% / +17%), see 3.1"],
       ["15:00 – 15:14 (before 15:15)", "B", "Full ₹1L at the UC price"],
       ["No UC hit, or first hit ≥ 15:15", "C", "Full ₹1L at the 15:15 candle open"]],
      widths=[2.1, 1.0, 3.4])

h2("3.1  Category A — hit UC before 3:00 pm (pullback ladder)")
body("The stock ran up and locked at the circuit early. Instead of chasing it at +20%, we place a two-rung "
     "limit ladder and only buy on the pullback:")
h3("Rung 1 — first 50% at +19%")
rich([("If, after the UC hit and before 15:00, a 1-minute LOW touches the +19% level "
       "(prev_close × 1.19), the first ", {}), ("₹50,000", {"bold": True}),
      (" fills at exactly the +19% level (limit fill).", {})])
h3("Rung 2 — second 50% at +17%")
rich([("If a 1-minute LOW then reaches the +17% level (prev_close × 1.17) before 15:00, the second ",
       {}), ("₹50,000", {"bold": True}),
      (" fills at exactly +17%, completing the position to ₹1,00,000 (a FULL fill).", {})])
h3("Rung 2 fallback — at 3:15 if +17% was not reached")
rich([("If rung 1 filled but the stock never dipped to +17% intraday, then at the 15:15 candle we fill the "
       "second ₹50,000 ", {}), ("only if", {"bold": True}),
      (" the 15:15 open is between +17% and +20% (i.e. it has NOT re-locked back into UC). The fill price is "
       "the 15:15 open.", {})])
h3("Half position — re-locked at 3:15")
rich([("If rung 1 filled but at 15:15 the stock is back at UC (re-locked, ≥ +20%), the second rung does "
       "NOT fill. The trade stays a ", {}), ("₹50,000 half position.", {"bold": True})])
h3("No entry — never pulled back")
rich([("If the stock hits UC but its low ", {}), ("never reaches +19%", {"bold": True, "color": C_RED}),
      (" before 15:00 and it stays locked through 15:15, there is ", {}),
      ("NO ENTRY", {"bold": True, "color": C_RED}), (" — the signal is skipped entirely.", {})])
body("Category A outcomes (decision table):", bold=True, size=10)
table(["Pulled to +19% (before 3pm)?", "Pulled to +17% (before 3pm)?", "State at 15:15", "Result"],
      [["Yes", "Yes", "—", "FULL ₹1L  (+19% & +17%)"],
       ["Yes", "No", "In [+17%, +20%)", "FULL ₹1L  (+19% & 15:15 open)"],
       ["Yes", "No", "Re-locked at UC", "HALF ₹50k  (+19% only)"],
       ["No", "—", "Still locked", "NO ENTRY"]],
      widths=[1.9, 1.9, 1.5, 2.0])
note("Cat A per-leg shares are floored independently; the position is the sum of the filled legs. "
     "The double-down short (Section 5) is sized on the ACTUAL filled quantity — a half position shorts half.")

h2("3.2  Category B — hit UC between 3:00 and 3:15")
rich([("The stock locks at the circuit inside the entry window itself, leaving no time for a pullback play. "
       "We simply enter the ", {}), ("full ₹1,00,000 at the UC price", {"bold": True}),
      (" (the circuit price = prev_close × 1.1995) at the moment it hits, in the 15:00–15:15 window.", {})])
note("Category B is the weakest entry in the backtest — its long leg is slightly negative on average "
     "(buying at the locked circuit price is adverse-selected), and it is only kept marginally positive by "
     "the double-down short. It is a small slice (36 trades).", tag="OBSERVED")

h2("3.3  Category C — everything else (the normal entry)")
rich([("If the stock never hit UC, or only hit it at/after 15:15, it is a normal signal: enter the ", {}),
      ("full ₹1,00,000 at the 15:15 one-minute candle open.", {"bold": True}),
      (" This is the plain-vanilla version of the strategy and the large majority of trades.", {})])

# ── 4. Capital allocation ──
h1("4.  Capital allocation — day-level priority sequencing")
body("Capital is NOT a flat ₹1,00,000 per trade. Each day the ₹5,00,000 pool is allocated in a STRICT "
     "priority order, so that the circuit plays (A and B) are funded first and Category C shares whatever "
     "remains. A hard ₹1,00,000-per-position cap always applies.")
formula("POOL = ₹5,00,000 per day        HARD CAP = ₹1,00,000 per position (always)")

h2("Phase 1 — intraday A & B commitments (chronological, first-come-first-served)")
bullet("Category A rung 1 commits ₹50,000 when the stock pulls back to +19% (after UC, before 3pm).", prefix="A leg 1: ", prefix_color=C_MID)
bullet("Category A rung 2 commits ₹50,000 when it pulls to +17% before 3pm.", prefix="A leg 2: ", prefix_color=C_MID)
bullet("Category B commits ₹1,00,000 at the UC price when it locks UC in the 3:00–3:15 window.", prefix="B: ", prefix_color=C_MID)
rich([("These commitments are applied ", {}), ("in the chronological order they trigger", {"bold": True}),
      (" through the day, each drawing from the pool. Running committed total = ", {}),
      ("X", {"bold": True, "mono": True}), (".", {})])
note("Pool exhaustion (flag a): if heavy UC activity means A/B commitments would exceed ₹5L before 15:15, "
     "it is first-come-first-served — the boundary leg takes whatever pool remains and later legs get ₹0. "
     "In 4.5 years this fully exhausted the pool on exactly ONE day.")

h2("Phase 2 — at 3:15, Priority 1: complete Category A's unfilled second legs")
rich([("For each Cat A trade holding only its first ₹50,000 (the +17% rung didn't fill intraday): if the "
       "stock is in [+17%, +20%) at 15:15, deploy the second ₹50,000 (fill at the 15:15 open). If it is "
       "re-locked at UC, it stays a half position. These top-ups ", {}),
      ("have priority over ALL of Category C", {"bold": True}), (" and add to X.", {})])

h2("Phase 3 — at 3:15, Priority 2: Category C splits the remainder")
formula("remaining_pool = 500000 − X          n_C = number of Category C signals that day\n"
        "per_C = min( 100000 ,  max(0, remaining_pool) / n_C )")
bullet("If remaining_pool ≥ n_C × ₹1L: every C trade gets the full ₹1,00,000 (any leftover stays idle).",
       prefix="Enough for all: ", prefix_color=C_MID)
bullet("If remaining_pool < n_C × ₹1L: the remainder is split EQUALLY — each C trade gets less than ₹1L.",
       prefix="Squeezed: ", prefix_color=C_MID)
bullet("Hard cap: no C trade ever exceeds ₹1,00,000, even if the equal-split math would; the excess stays idle.",
       prefix="Cap: ", prefix_color=C_MID)
bullet("On extreme heavy-UC days, if A/B exhaust the pool, C can get ₹0 (no C entries that day).",
       prefix="Zero: ", prefix_color=C_RED)
formula("shares = floor( allocated_capital / entry_price )      capital_deployed = shares × entry_price")

h2("Strict priority summary")
table(["Phase", "When", "Who", "Amount"],
      [["1", "Intraday (as they trigger)", "A legs, B", "₹50k / leg (A), ₹1L (B) — chronological FCFS"],
       ["2", "3:15", "A second-leg top-ups", "₹50k each (before any C)"],
       ["3", "3:15", "Category C", "min(₹1L, remaining / n_C) each"]],
      widths=[0.7, 1.6, 1.9, 3.0])
note("Leftover capital when C is fully funded at ₹1L each and money still remains stays IDLE — it is not "
     "redistributed to give anyone more than ₹1L (flag d).", tag="IDLE CAPITAL")

# ── 5. Sizing, short, costs ──
h1("5.  Per-trade sizing, the double-down short, and costs")
h3("Shares & blended entry")
body("Each leg's shares = floor(leg_allocation / leg_price). A Category A position's shares and capital are "
     "the sum of its filled legs; its blended entry = capital_deployed / total_shares.")
h3("Double-down intraday short (all categories)")
rich([("At each long exit (next trading day, at 09:45 / 12:00 / on the 14% target), a short is opened of size "
       "= the ", {}), ("filled long quantity", {"bold": True}),
      (" (a half position shorts half), and covered at that day's 15:00 candle open.", {})])
formula("short_pnl   = filled_shares × (long_exit_price − open_15:00)\n"
        "combined_pnl = long_pnl + short_pnl")
h3("Costs — proportional to actual filled value")
table(["Series", "Long cost", "Short cost"],
      [["Gross", "—", "—"],
       ["Net A", "0.23% × capital_deployed", "0.10% × short notional"],
       ["Net B", "0.38% × capital_deployed", "0.10% × short notional"]],
      widths=[1.2, 2.7, 2.7])
body("Because costs scale with the actual filled value, a Category A half position pays long cost on ~₹50k, "
     "not ₹1L.", italic=True, size=9.5)

# ── 6. What this looks like in practice ──
h1("6.  How it plays out in practice (backtest, 2022–2026)")
body("Across ~3,553 trades on the refreshed 1-minute data, the category mix and the real allocation behaviour "
     "were:")
table(["Metric", "Value"],
      [["Category A entered (of 292 A signals)", "276  (174 full-fill, 102 half-fill; 15 no-entry)"],
       ["Category B entered", "36"],
       ["Category C entered", "3,241"],
       ["Avg A+B capital committed per day", "₹26,337  (max ₹5,00,000 on one day)"],
       ["Days Category C got full ₹1L each", "809"],
       ["Days Category C was equal-split (< ₹1L)", "174"],
       ["Days Category C got ₹0 (pool exhausted by A/B)", "1"],
       ["Avg Category C allocation per trade", "₹94,367"],
       ["Avg capital deployed: A-full / A-half / B / C", "₹99,201 / ₹49,603 / ₹99,610 / ₹87,436"]],
      widths=[3.6, 3.4])
body("The pool constraint is real but light: A and B commit only ~₹26k/day on average, so Category C is "
     "funded at the full ₹1L on the large majority of days and only squeezed on heavy-UC days. Modelling the "
     "true ₹5L pool this way keeps the reported returns honest — the strategy never assumes more than "
     "₹5,00,000 of simultaneous capital.", size=10)

# ── worked examples ──
h1("7.  Worked examples")
h2("7.1  A light day  —  1 Category B + 4 Category C")
bullet("Phase 1: the B trade commits ₹1,00,000 at its UC hit.  →  X = ₹1,00,000.")
bullet("Phase 3: remaining = ₹5L − ₹1L = ₹4,00,000;  n_C = 4;  per_C = min(₹1L, ₹4L/4) = ₹1,00,000.")
bullet("Result: B gets ₹1L, each C gets the full ₹1L. Leftover = ₹0.", prefix_color=C_GREEN, prefix="")
h2("7.2  A heavy-UC day  —  3 Category A (full) + 6 Category C")
bullet("Phases 1–2: the 3 full A trades commit ₹1L each.  →  X = ₹3,00,000.")
bullet("Phase 3: remaining = ₹2,00,000;  n_C = 6;  per_C = min(₹1L, ₹2L/6) = ₹33,333 each.")
bullet("Result: A funded fully, Category C squeezed to ₹33,333 apiece (equal-split day).")
h2("7.3  An extreme day  —  5 Category A (full) + 2 Category C")
bullet("Phases 1–2: 5 full A trades commit ₹1L each.  →  X = ₹5,00,000 (pool exhausted).")
bullet("Phase 3: remaining = ₹0;  per_C = ₹0.  Category C gets nothing that day.", prefix_color=C_RED, prefix="")
body("This last case occurred on exactly one trading day in 4.5 years.", italic=True, size=9.5)

# ── footer ──
fp = doc.add_paragraph(); fp.paragraph_format.space_before = Pt(14)
_bottom_border(fp, color="BBBBBB", sz="4")
fp2 = doc.add_paragraph()
r = fp2.add_run("Reference for the composite UC-ladder + double-down BTST strategy on 1-minute NSE data. "
                "All percentage levels are measured from the previous-day VWAP-close; UC = prev_close × 1.1995. "
                "Figures from the backtest over 2022-02 to 2026-07.")
r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x70, 0x70, 0x70); r.font.name = FONT

doc.save(OUT)
print(f"Saved -> {OUT}")
