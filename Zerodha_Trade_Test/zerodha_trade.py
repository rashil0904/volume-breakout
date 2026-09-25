#!/usr/bin/env python3
"""
zerodha_trade.py — Standalone interactive single-trade testing
==============================================================
Standalone: does NOT read trade-list CSVs or call prepare_data.py.
Use this to confirm the full auth → funds → holdings → order flow
works correctly before connecting it to the real strategy.

Usage:
    python3 zerodha_trade.py

Flow:
    1. Authenticate (cached token today, or full login if first run)
    2. Print available funds
    3. Print current holdings
    4. Prompt: action / symbol / quantity / dry-run
    5. Dry-run → print order preview only
       Live    → type YES to confirm, then place order
"""

from kiteconnect import KiteConnect
from zerodha_auth import authenticate

# ── CONFIG — fill these in before running ─────────────────────────────────────
API_KEY = "ca08i1k9is9pwjgf"
API_SECRET = "tjnrffsfd2achmc0rtvkf9w2nesd0jqo"

# Fixed order defaults
EXCHANGE   = "NSE"
ORDER_TYPE = KiteConnect.ORDER_TYPE_MARKET
PRODUCT    = KiteConnect.PRODUCT_CNC
VALIDITY   = KiteConnect.VALIDITY_DAY


# ── Display helpers ───────────────────────────────────────────────────────────

def print_funds(kite: KiteConnect) -> None:
    print("\n─── Available Funds ─────────────────────────────────────")
    try:
        margins = kite.margins(segment="equity")
        avail   = margins.get("net", margins.get("available", {}).get("cash", "N/A"))
        used    = margins.get("utilised", {}).get("debits", "N/A")
        print(f"  Available cash / net margin : ₹{avail:,.2f}" if isinstance(avail, float)
              else f"  Available cash / net margin : {avail}")
        if isinstance(used, float):
            print(f"  Used margin                 : ₹{used:,.2f}")
    except Exception as e:
        print(f"  Could not fetch funds: {e}")
    print("─────────────────────────────────────────────────────────")


def print_holdings(kite: KiteConnect) -> dict:
    """Print holdings table and return a dict {symbol: quantity} for sell checks."""
    print("\n─── Current Holdings ────────────────────────────────────")
    held = {}
    try:
        holdings = kite.holdings()
        if not holdings:
            print("  (no holdings)")
        else:
            fmt = "  {:<16} {:>6}  {:>10}  {:>10}  {:>10}"
            print(fmt.format("Symbol", "Qty", "Avg Price", "LTP", "P&L"))
            print("  " + "-" * 58)
            for h in holdings:
                sym  = h["tradingsymbol"]
                qty  = h["quantity"]
                avg  = h.get("average_price", 0)
                ltp  = h.get("last_price", 0)
                pnl  = h.get("pnl", (ltp - avg) * qty)
                sign = "+" if pnl >= 0 else ""
                print(fmt.format(sym, qty, f"₹{avg:,.2f}", f"₹{ltp:,.2f}",
                                 f"{sign}₹{pnl:,.2f}"))
                held[sym.upper()] = qty
    except Exception as e:
        print(f"  Could not fetch holdings: {e}")
    print("─────────────────────────────────────────────────────────")
    return held


# ── Interactive prompts ───────────────────────────────────────────────────────

def prompt_action() -> str:
    while True:
        raw = input("\nAction (BUY / SELL): ").strip().upper()
        if raw in ("BUY", "SELL"):
            return raw
        print("  Invalid — please type BUY or SELL.")


def prompt_symbol() -> str:
    while True:
        raw = input("Trading symbol (e.g. SBIN): ").strip().upper()
        if raw:
            return raw
        print("  Symbol cannot be empty.")


def prompt_quantity() -> int:
    while True:
        raw = input("Quantity (whole shares): ").strip()
        try:
            qty = int(raw)
            if qty > 0:
                return qty
        except ValueError:
            pass
        print("  Please enter a positive whole number.")


def prompt_dry_run() -> bool:
    raw = input("Dry run? [Y/n]: ").strip().lower()
    return raw not in ("n", "no")


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(kite: KiteConnect, action: str, symbol: str, qty: int) -> None:
    tx = (KiteConnect.TRANSACTION_TYPE_BUY if action == "BUY"
          else KiteConnect.TRANSACTION_TYPE_SELL)
    order_id = kite.place_order(
        variety=KiteConnect.VARIETY_REGULAR,
        tradingsymbol=symbol,
        exchange=EXCHANGE,
        transaction_type=tx,
        quantity=qty,
        order_type=ORDER_TYPE,
        product=PRODUCT,
        validity=VALIDITY,
    )
    print(f"\n  Order placed. Order ID: {order_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    kite = authenticate(API_KEY, API_SECRET)

    print_funds(kite)
    held = print_holdings(kite)

    # Gather trade parameters
    action  = prompt_action()
    symbol  = prompt_symbol()
    qty     = prompt_quantity()
    dry_run = prompt_dry_run()

    # Sell-specific holdings check
    if action == "SELL":
        held_qty = held.get(symbol, 0)
        if held_qty == 0:
            print(f"\n  WARNING: {symbol} not found in your current holdings.")
        elif held_qty < qty:
            print(f"\n  WARNING: You hold only {held_qty} shares of {symbol} "
                  f"but are trying to sell {qty}.")
        else:
            print(f"\n  Holdings check OK: you hold {held_qty} shares of {symbol}.")

    # Order summary
    print("\n─── Order Summary ───────────────────────────────────────")
    print(f"  Action    : {action}")
    print(f"  Symbol    : {symbol}")
    print(f"  Quantity  : {qty}")
    print(f"  Exchange  : {EXCHANGE}")
    print(f"  Type      : {ORDER_TYPE} ({PRODUCT}, {VALIDITY})")
    print(f"  Mode      : {'DRY RUN — no order will be placed' if dry_run else 'LIVE'}")
    print("─────────────────────────────────────────────────────────")

    if dry_run:
        print("\n[DRY RUN] Order NOT placed. Set dry run = n to place a real order.")
        return

    # Live order — require explicit YES
    confirm = input("\nType YES to place this order (anything else cancels): ").strip()
    if confirm != "YES":
        print("Cancelled — order not placed.")
        return

    try:
        place_order(kite, action, symbol, qty)
    except Exception as e:
        print(f"\n  Order FAILED: {e}")

    # CDSL note for delivery sells
    if action == "SELL":
        print(
            "\n  NOTE — Delivery sell (CNC): Zerodha requires CDSL TPIN + OTP "
            "authorization\n"
            "  unless you have DDPI / POA set up with them. If the order doesn't\n"
            "  go through as expected, check the Kite app, your registered email,\n"
            "  or SMS for a pending e-DIS authorization request."
        )


if __name__ == "__main__":
    main()
