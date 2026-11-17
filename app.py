"""Labor Market Elasticity Dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import os

import pandas as pd
import streamlit as st

from labordash import charts, elasticity as el
from labordash.config import DEFAULT_STATES, DEMAND_NAMES, DEMAND_PROXIES, STATE_FIPS
from labordash.pipeline import build_model_frame, load_data, macro_indicators

st.set_page_config(page_title="Labor Market Elasticity Dashboard", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _secret(name: str) -> str:
    """API key from the environment or Streamlit secrets; empty string if neither is set."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # no secrets.toml present
        return ""


def _theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def stretch(fn, *args, **kwargs):
    """Full-width element across Streamlit versions.

    Newer releases take ``width="stretch"`` and warn on ``use_container_width``;
    older ones only know the latter. Each element switched in a different
    release, so the function's own signature decides.
    """
    try:
        param = inspect.signature(fn).parameters.get("width")
        modern = param is not None and isinstance(param.default, str)
    except (TypeError, ValueError):
        modern = False
    if modern:
        return fn(*args, width="stretch", **kwargs)
    return fn(*args, use_container_width=True, **kwargs)


def show(fig):
    stretch(st.plotly_chart, fig, config={"displayModeBar": False})


@st.cache_data(ttl=3 * 3600, show_spinner="Pulling the latest data from FRED and BLS…")
def get_data(dimension: str, unit_keys: tuple, source: str, keyed: tuple, _fred_key: str, _bls_key: str, _force: bool):
    b = load_data(dimension, list(unit_keys) if unit_keys else None, source,
                  fred_key=_fred_key or None, bls_key=_bls_key or None, force=_force)
    return b.macro, b.panel, b.meta, {u.key: u.label for u in b.units}


def month(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y") if pd.notna(ts) else "n/a"


def last_valid(s: pd.Series):
    s = s.dropna()
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (pd.NaT, float("nan"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    dimension = st.radio("Compare across", ["Industries", "States"], horizontal=True).lower()
    unit_keys: tuple = ()
    if dimension == "states":
        picked = st.multiselect("States", list(STATE_FIPS), default=DEFAULT_STATES, format_func=lambda a: STATE_FIPS[a][0])
        unit_keys = tuple(sorted(picked))

    st.subheader("Model")
    horizon = st.radio(
        "Change measured over", [12, 3, 1], horizontal=True,
        format_func=lambda h: {12: "12 months", 3: "3 months", 1: "1 month"}[h],
        help="Elasticities are estimated on log changes over this many months. Longer changes capture slower adjustment.",
    )
    demand = st.selectbox("Aggregate demand control", list(DEMAND_PROXIES), format_func=DEMAND_PROXIES.get)
    exclude_pandemic = st.checkbox(
        "Exclude the pandemic shock", value=True,
        help="Drops changes that overlap March 2020 to June 2021, when layoffs of low-wage workers pushed average "
             "hourly earnings up mechanically.",
    )

    st.subheader("Data")
    source = st.radio("Labor series from", ["FRED", "BLS"], horizontal=True,
                      help="Employment, earnings, CPI and unemployment. The demand control always comes from FRED.").lower()
    with st.expander("API keys (optional)"):
        st.caption("Without keys the app uses FRED's public CSV endpoint and the BLS v1 API (25 requests per day). "
                   "Free keys raise those limits.")
        fred_key = st.text_input("FRED API key", value=_secret("FRED_API_KEY"), type="password")
        bls_key = st.text_input("BLS API key", value=_secret("BLS_API_KEY"), type="password")
    if st.button("Refresh data now"):
        get_data.clear()
        st.session_state["force"] = True

if dimension == "states" and len(unit_keys) < 1:
    st.info("Pick at least one state in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Extract + transform
# --------------------------------------------------------------------------- #
force = bool(st.session_state.pop("force", False))
macro, panel, meta, labels = get_data(dimension, unit_keys, source, (bool(fred_key), bool(bls_key)),
                                      fred_key, bls_key, force)
mode = _theme_mode()

st.title("Labor Market Elasticity Dashboard")

problems = meta[meta["status"].isin(["stale", "failed"])]
missing_core = [c for c in ("cpi", demand, "unrate", "payems", "ahe") if c not in macro]
if missing_core or panel.empty:
    st.error("Core series could not be loaded, so nothing can be estimated. Details below.")
    stretch(st.dataframe, problems[["series", "series_id", "source", "status", "error"]], hide_index=True)
    st.stop()

frame = build_model_frame(macro, panel, horizon, demand)
seasonal = el.needs_month_dummies(dimension == "industries", horizon)
opts = dict(exclude_pandemic=exclude_pandemic, month_dummies=seasonal)
results = el.estimate_all(frame, horizon, **opts)
pooled_frame = frame[frame["unit"] != "total_private"]
pooled = el.pooled_twfe(pooled_frame, horizon, **opts)

fetched = pd.to_datetime(meta["fetched_at"], utc=True).max()
st.caption(
    f"Live data: employment and earnings through **{month(panel.dropna(subset=['emp'])['date'].max())}**, "
    f"CPI through **{month(last_valid(macro['cpi'])[0])}**, {DEMAND_NAMES[demand]} through "
    f"**{month(last_valid(macro[demand])[0])}**. Estimation sample ends **{month(frame['date'].max()) if len(frame) else 'n/a'}**. "
    f"Last pulled {fetched.strftime('%d %b %Y %H:%M UTC') if pd.notna(fetched) else 'n/a'}."
)
if len(problems):
    n_stale, n_failed = (problems["status"] == "stale").sum(), (problems["status"] == "failed").sum()
    st.warning(f"{n_failed} series failed to load and {n_stale} are served from an older cached copy. "
               "See the Data and methods tab for details.")

tab_overview, tab_elast, tab_roll, tab_data = st.tabs(["Overview", "Elasticities", "Over time", "Data and methods"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    ind = macro_indicators(macro)
    tiles = [
        ("unrate", "Unemployment rate", "inverse", "Share of the labor force without a job"),
        ("inflation", "Inflation", "inverse", "CPI, change from a year earlier"),
        ("real_wage_growth", "Real wage growth", "normal", "Average hourly earnings less inflation, from a year earlier"),
        ("payroll_growth", "Payroll growth", "normal", "Total nonfarm jobs, change from a year earlier"),
    ]
    for col, (key, label, delta_color, helptext) in zip(st.columns(4), tiles):
        date, value = last_valid(ind[key])
        prior = ind[key].get(date - pd.DateOffset(months=12)) if pd.notna(date) else None
        delta = f"{value - prior:+.1f} pts vs a year ago" if prior is not None and pd.notna(prior) else None
        col.metric(f"{label} · {month(date)}", f"{value:.1f}%", delta, delta_color=delta_color, help=helptext)

    recent = ind.loc[ind.index >= ind.index.max() - pd.DateOffset(years=10)]
    left, right = st.columns(2)
    with left:
        show(charts.line_chart(recent["unrate"], "Unemployment rate", mode))
        show(charts.line_chart(recent["real_wage_growth"], "Real hourly earnings, change from a year earlier", mode, zero_line=True))
    with right:
        show(charts.line_chart(recent["inflation"], "CPI inflation, change from a year earlier", mode))
        show(charts.line_chart(recent["payroll_growth"], "Nonfarm payrolls, change from a year earlier", mode, zero_line=True))

# --------------------------------------------------------------------------- #
# Elasticities
# --------------------------------------------------------------------------- #
with tab_elast:
    if results.empty:
        st.info("Not enough observations to estimate with these settings. Try a longer sample or a shorter horizon.")
    else:
        units_only = results[results["unit"] != "total_private"]
        sig = ((units_only["wage_lo"] > 0) | (units_only["wage_hi"] < 0)).sum()
        c1, c2, c3 = st.columns(3)
        if pooled:
            c1.metric("Pooled wage elasticity", f"{pooled['wage_elast']:+.2f}",
                      help="Two-way fixed effects across all units: unit effects plus month effects, clustered by unit.")
            c1.caption(f"95% interval {pooled['wage_lo']:+.2f} to {pooled['wage_hi']:+.2f} · "
                       f"{pooled['units']} units × {pooled['months']} months")
        else:
            c1.metric("Pooled wage elasticity", "n/a", help="Needs at least three units.")
        c2.metric("Median unit wage elasticity", f"{units_only['wage_elast'].median():+.2f}")
        c2.caption(f"Range {units_only['wage_elast'].min():+.2f} to {units_only['wage_elast'].max():+.2f}")
        c3.metric("Median demand elasticity", f"{units_only['demand_elast'].median():+.2f}")
        c3.caption(f"{sig} of {len(units_only)} wage elasticities differ from zero at 95%")

        which = st.radio("Show", ["wage", "demand"], horizontal=True,
                         format_func={"wage": "Elasticity to the real wage", "demand": "Elasticity to aggregate demand"}.get)
        if which == "wage":
            st.markdown(
                "**Percent change in employment associated with a 1% rise in the real hourly wage**, holding aggregate "
                "demand fixed. A negative value means employment falls as labor gets more expensive (movement along "
                "labor demand). A positive value means jobs and pay rise together, the mark of demand-driven swings."
            )
            show(charts.coef_plot(results, "wage", mode, xtitle="Employment elasticity to the real wage",
                                  reference=pooled["wage_elast"] if pooled else None,
                                  reference_label=f"Pooled {pooled['wage_elast']:+.2f}" if pooled else ""))
        else:
            st.markdown(
                f"**Percent change in employment associated with a 1% rise in {DEMAND_NAMES[demand]}**, "
                "holding the real wage fixed. Larger values mark the most cyclical units."
            )
            show(charts.coef_plot(results, "demand", mode, xtitle="Employment elasticity to aggregate demand"))
        st.caption("Whiskers are 95% Newey-West intervals. Filled dots exclude zero, hollow dots do not.")

        table = results[["label", "wage_elast", "wage_se", "demand_elast", "demand_se", "nobs", "r2", "start", "end"]].rename(
            columns={"label": "Unit", "wage_elast": "Wage elasticity", "wage_se": "SE (wage)",
                     "demand_elast": "Demand elasticity", "demand_se": "SE (demand)", "nobs": "Months",
                     "r2": "R²", "start": "From", "end": "To"})
        for c in ("From", "To"):
            table[c] = pd.to_datetime(table[c]).dt.strftime("%b %Y")
        stretch(st.dataframe, table.round(3), hide_index=True)

        st.subheader("What drives one estimate")
        pick = st.selectbox("Unit", list(results["unit"]), format_func=labels.get, key="pr_unit")
        row = results.set_index("unit").loc[pick]
        pr = el.partial_regression(frame[frame["unit"] == pick], horizon, **opts)
        show(charts.partial_scatter(pr, float(row["wage_elast"]), mode))
        st.caption("Each dot is one month. Both axes are net of the demand control, so the fitted slope is exactly "
                   f"the wage elasticity for {labels[pick]} ({row['wage_elast']:+.2f}).")

# --------------------------------------------------------------------------- #
# Rolling
# --------------------------------------------------------------------------- #
with tab_roll:
    if results.empty:
        st.info("Not enough observations to estimate with these settings.")
    else:
        sel_col, win_col = st.columns(2)
        pick = sel_col.selectbox("Unit", list(results["unit"]), format_func=labels.get, key="roll_unit")
        window = win_col.slider("Rolling window (months)", 36, 120, 60, step=12,
                                help="How many months each re-estimate uses. Longer windows are smoother and slower to react.")
        row = results.set_index("unit").loc[pick]
        roll = el.rolling_unit(frame[frame["unit"] == pick], horizon, window, **opts)
        if roll.empty:
            st.info("No window has enough observations. Shorten the rolling window.")
        else:
            st.markdown(
                f"Each point is plotted at the date its window **ends** and uses the **{window} months** up to then. "
                f"The line therefore starts {window} months after the data begin, and the right-hand edge always "
                f"reflects the newest data (window ending {month(roll['date'].max())})."
            )
            show(charts.rolling_chart(roll, "wage", f"{labels[pick]}: employment elasticity to the real wage", mode,
                                      full_sample=float(row["wage_elast"]), shade_pandemic=not exclude_pandemic))
            show(charts.rolling_chart(roll, "demand", f"{labels[pick]}: employment elasticity to aggregate demand", mode,
                                      full_sample=float(row["demand_elast"]), shade_pandemic=not exclude_pandemic))
            note = "Shaded band is the 95% Newey-West interval; the dashed line is the full-sample estimate."
            if exclude_pandemic:
                note += " Gaps are windows left with too few observations once pandemic months are dropped."
            st.caption(note)

# --------------------------------------------------------------------------- #
# Data and methods
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Data freshness")
    fresh = meta[["series", "series_id", "source", "status", "last_observation", "observations", "fetched_at", "error"]].copy()
    fresh["last_observation"] = pd.to_datetime(fresh["last_observation"]).dt.strftime("%b %Y")
    fresh["fetched_at"] = pd.to_datetime(fresh["fetched_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    stretch(st.dataframe, fresh, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download elasticities (CSV)", results.to_csv(index=False).encode(), "elasticities.csv", "text/csv")
    d2.download_button("Download estimation panel (CSV)", frame.to_csv(index=False).encode(), "estimation_panel.csv", "text/csv")

    st.subheader("Method")
    hac_lags = int(results["hac_lags"].median()) if len(results) else el.nw_lags(200, horizon)
    st.markdown(
        f"""
For each unit *i* the app estimates, on log changes over *h* = {horizon} month{'s' if horizon > 1 else ''}:

`Δlog(Employment) = α + η · Δlog(Hourly earnings ÷ CPI) + γ · Δlog({DEMAND_NAMES[demand].capitalize()}) + ε`

- **η, wage elasticity.** Percent change in employment per 1% change in the real hourly wage.
- **γ, demand elasticity.** Percent change in employment per 1% change in aggregate demand.
- **Inference.** Newey-West standard errors with {hac_lags} lags (the overlap created by {horizon}-month changes plus
  the usual rule of thumb).
- **Pooled estimate.** Unit and month fixed effects, standard errors clustered by unit. Month effects absorb every
  economy-wide shock, so η comes only from units whose real wages moved differently in the same month.
- **Seasonality.** {"State series are not seasonally adjusted; month dummies are added at horizons under 12 months." if dimension == "states" else "Industry series are seasonally adjusted at source."}
- **Pandemic.** {"Changes overlapping March 2020 to June 2021 are dropped." if exclude_pandemic else "Pandemic months are included, so the 2020 composition effect is in the estimates."}

**How to read η.** These are reduced-form elasticities, not structural ones. Wages and employment are determined
together, so a single equation cannot separate the labor demand curve from the labor supply curve. A negative η says
wage-side shocks dominated in that unit and period; a positive η says demand-side shocks did. Average hourly earnings
also shift when the mix of workers changes, even if nobody's pay does. Separating supply from demand needs an
instrument or a policy experiment, which is the natural next module for this project.
"""
    )
