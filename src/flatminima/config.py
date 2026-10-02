"""Project-wide configuration: asset universe, sample period and seeds."""

from __future__ import annotations

START_DATE = "2005-01-03"
SEED = 20260930

# Liquid US-listed ETFs with Yahoo history from before 2005. Broad asset-class coverage
# (equity regions/styles, sectors, rates, credit, real estate, gold).
# Caveat: a universe of ETFs that still exist today carries survivorship bias; this is
# documented in the README and discussed in the limitations section.
UNIVERSE: dict[str, str] = {
    "SPY": "US large cap",
    "QQQ": "US Nasdaq-100",
    "IWM": "US small cap",
    "MDY": "US mid cap",
    "IWD": "US large value",
    "IWF": "US large growth",
    "EFA": "Developed ex-US",
    "EEM": "Emerging markets",
    "EWJ": "Japan",
    "EWZ": "Brazil",
    "FXI": "China large cap",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLV": "Health care",
    "XLP": "Consumer staples",
    "XLY": "Consumer discretionary",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "VNQ": "US REITs",
    "TLT": "US Treasuries 20y+",
    "IEF": "US Treasuries 7-10y",
    "SHY": "US Treasuries 1-3y",
    "TIP": "US TIPS",
    "LQD": "US investment grade credit",
    "AGG": "US aggregate bonds",
    "GLD": "Gold",
}
