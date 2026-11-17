"""Series catalog: every FRED / BLS identifier the pipeline pulls lives here.

Adding an industry or changing a series is a one-line edit in this file; nothing
else in the pipeline hard-codes an ID.
"""
from __future__ import annotations

from dataclasses import dataclass

EARLIEST = "2006-01-01"  # CES "all employees" hourly earnings begin March 2006


@dataclass(frozen=True)
class Unit:
    """One cross-sectional unit (an industry or a state)."""

    key: str
    label: str
    emp_fred: str
    wage_fred: str
    emp_bls: str
    wage_bls: str
    seasonally_adjusted: bool = True
    aggregate: bool = False  # True for benchmark rows excluded from the pooled panel

    def ids(self, source: str) -> tuple[str, str]:
        return (self.emp_fred, self.wage_fred) if source == "fred" else (self.emp_bls, self.wage_bls)


# --- National macro series -------------------------------------------------
# role -> (FRED id, BLS id or None when the series is not a BLS product)
MACRO = {
    "unrate": ("UNRATE", "LNS14000000"),      # unemployment rate, SA, %
    "cpi": ("CPIAUCSL", "CUSR0000SA0"),       # CPI-U all items, SA, 1982-84=100
    "payems": ("PAYEMS", "CES0000000001"),    # total nonfarm payrolls, SA, thousands
    "ahe": ("CES0500000003", "CES0500000003"),  # avg hourly earnings, total private, SA
    "pce": ("PCEC96", None),                  # real personal consumption (BEA), monthly
    "indpro": ("INDPRO", None),               # industrial production index (Fed Board)
}
MACRO_LABELS = {
    "unrate": "Unemployment rate",
    "cpi": "Consumer price index (CPI-U)",
    "payems": "Total nonfarm payrolls",
    "ahe": "Average hourly earnings, total private",
    "pce": "Real personal consumption",
    "indpro": "Industrial production",
}
DEMAND_PROXIES = {"pce": "Real consumer spending (PCE)", "indpro": "Industrial production"}
DEMAND_NAMES = {"pce": "real consumer spending", "indpro": "industrial production"}  # mid-sentence form


# --- Industries: CES supersectors -------------------------------------------
# (key, label, CES supersector code, FRED employment mnemonic, aggregate?)
_INDUSTRIES = [
    ("total_private", "Total private", "05", "USPRIV", True),
    ("mining", "Mining and logging", "10", "USMINE", False),
    ("construction", "Construction", "20", "USCONS", False),
    ("manufacturing", "Manufacturing", "30", "MANEMP", False),
    ("trade_transport", "Trade, transport and utilities", "40", "USTPU", False),
    ("information", "Information", "50", "USINFO", False),
    ("finance", "Financial activities", "55", "USFIRE", False),
    ("prof_business", "Professional and business services", "60", "USPBS", False),
    ("educ_health", "Education and health services", "65", "USEHS", False),
    ("leisure", "Leisure and hospitality", "70", "USLAH", False),
    ("other_services", "Other services", "80", "USSERV", False),
]

INDUSTRIES = [
    Unit(
        key=key,
        label=label,
        emp_fred=fred_emp,
        wage_fred=f"CES{code}00000003",   # avg hourly earnings, all employees, SA
        emp_bls=f"CES{code}00000001",     # all employees, thousands, SA
        wage_bls=f"CES{code}00000003",
        seasonally_adjusted=True,
        aggregate=agg,
    )
    for key, label, code, fred_emp, agg in _INDUSTRIES
]


# --- States: CES State & Area, total private ---------------------------------
# State hourly earnings are only published not seasonally adjusted, so the
# employment series is taken NSA as well and seasonality is handled in the model.
STATE_FIPS = {
    "AL": ("Alabama", "01"), "AK": ("Alaska", "02"), "AZ": ("Arizona", "04"),
    "AR": ("Arkansas", "05"), "CA": ("California", "06"), "CO": ("Colorado", "08"),
    "CT": ("Connecticut", "09"), "DE": ("Delaware", "10"), "DC": ("District of Columbia", "11"),
    "FL": ("Florida", "12"), "GA": ("Georgia", "13"), "HI": ("Hawaii", "15"),
    "ID": ("Idaho", "16"), "IL": ("Illinois", "17"), "IN": ("Indiana", "18"),
    "IA": ("Iowa", "19"), "KS": ("Kansas", "20"), "KY": ("Kentucky", "21"),
    "LA": ("Louisiana", "22"), "ME": ("Maine", "23"), "MD": ("Maryland", "24"),
    "MA": ("Massachusetts", "25"), "MI": ("Michigan", "26"), "MN": ("Minnesota", "27"),
    "MS": ("Mississippi", "28"), "MO": ("Missouri", "29"), "MT": ("Montana", "30"),
    "NE": ("Nebraska", "31"), "NV": ("Nevada", "32"), "NH": ("New Hampshire", "33"),
    "NJ": ("New Jersey", "34"), "NM": ("New Mexico", "35"), "NY": ("New York", "36"),
    "NC": ("North Carolina", "37"), "ND": ("North Dakota", "38"), "OH": ("Ohio", "39"),
    "OK": ("Oklahoma", "40"), "OR": ("Oregon", "41"), "PA": ("Pennsylvania", "42"),
    "RI": ("Rhode Island", "44"), "SC": ("South Carolina", "45"), "SD": ("South Dakota", "46"),
    "TN": ("Tennessee", "47"), "TX": ("Texas", "48"), "UT": ("Utah", "49"),
    "VT": ("Vermont", "50"), "VA": ("Virginia", "51"), "WA": ("Washington", "53"),
    "WV": ("West Virginia", "54"), "WI": ("Wisconsin", "55"), "WY": ("Wyoming", "56"),
}
DEFAULT_STATES = ["IL", "IN", "WI", "MI", "OH", "MN", "CA", "TX", "NY", "FL"]


def _state_unit(abbr: str) -> Unit:
    name, fips = STATE_FIPS[abbr]
    # SMU + state FIPS + area 00000 (statewide) + industry 05000000 (total private) + data type
    base = f"SMU{fips}0000005000000"
    return Unit(
        key=abbr,
        label=name,
        emp_fred=f"{base}01", wage_fred=f"{base}03",
        emp_bls=f"{base}01", wage_bls=f"{base}03",
        seasonally_adjusted=False,
    )


STATES = [_state_unit(a) for a in STATE_FIPS]


def catalog(dimension: str) -> list[Unit]:
    if dimension == "industries":
        return INDUSTRIES
    if dimension == "states":
        return STATES
    raise ValueError(f"unknown dimension: {dimension!r}")


# --- Pandemic window ---------------------------------------------------------
# Average hourly earnings jumped in spring 2020 because low-wage workers were laid
# off first (a composition effect), which mechanically produces a large negative
# wage-employment correlation. Observations whose differencing window touches this
# period can be dropped.
PANDEMIC_START = "2020-03-01"
PANDEMIC_END = "2021-06-01"
