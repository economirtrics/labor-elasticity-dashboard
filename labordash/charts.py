"""Plotly figures. One hue for data, neutral ink for everything else."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import PANDEMIC_END, PANDEMIC_START

# Colors validated for color-vision-deficiency separation and contrast on each surface.
PALETTE = {
    "light": {"series": "#2a78d6", "accent": "#eb6834", "ink": "#52514e", "grid": "#e7e6e2",
              "surface": "#ffffff", "band": "rgba(42,120,214,0.16)", "shade": "rgba(82,81,78,0.08)"},
    "dark": {"series": "#3987e5", "accent": "#d95926", "ink": "#c3c2b7", "grid": "#33332f",
             "surface": "#0e1117", "band": "rgba(57,135,229,0.22)", "shade": "rgba(195,194,183,0.10)"},
}


def _layout(fig: go.Figure, p: dict, *, height: int, title: str | None = None, hovermode: str = "closest") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=56, t=44 if title else 12, b=8),
        hovermode=hovermode,
        showlegend=False,
        font=dict(size=13),
    )
    if title:  # an explicit empty title renders as the word "undefined" under Streamlit's theme
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=p["grid"], ticks="outside", tickcolor=p["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=p["grid"], zeroline=False)
    return fig


def _hline(fig: go.Figure, y: float, color: str, dash: str = "solid", width: float = 1) -> None:
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
                  line=dict(color=color, width=width, dash=dash), layer="below")


def line_chart(s: pd.Series, title: str, mode: str = "light", *, decimals: int = 1, suffix: str = "%",
               zero_line: bool = False, height: int = 250) -> go.Figure:
    """Single-series time line with the latest value labelled at the end."""
    p = PALETTE[mode]
    s = s.dropna(how="all")
    fig = go.Figure(
        go.Scatter(
            x=s.index, y=s.to_numpy(), mode="lines", connectgaps=False,
            line=dict(color=p["series"], width=2),
            hovertemplate=f"%{{x|%b %Y}}: %{{y:.{decimals}f}}{suffix}<extra></extra>",
        )
    )
    valid = s.dropna()
    if len(valid):
        fig.add_annotation(x=valid.index[-1], y=float(valid.iloc[-1]), text=f"{valid.iloc[-1]:.{decimals}f}{suffix}",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=12, color=p["ink"]))
    if zero_line:
        _hline(fig, 0, p["ink"])
    return _layout(fig, p, height=height, title=title, hovermode="x")


def coef_plot(res: pd.DataFrame, prefix: str, mode: str = "light", *, xtitle: str = "",
              reference: float | None = None, reference_label: str = "") -> go.Figure:
    """Dot-and-whisker plot: one row per unit, sorted by the point estimate.

    Filled dots have a 95% interval that excludes zero; hollow dots do not.
    """
    p = PALETTE[mode]
    df = res.sort_values(f"{prefix}_elast").reset_index(drop=True)
    est, lo, hi = (df[f"{prefix}_{k}"].to_numpy(float) for k in ("elast", "lo", "hi"))
    significant = (lo > 0) | (hi < 0)

    fig = go.Figure()
    for flag, name, fill in ((True, "95% interval excludes zero", p["series"]), (False, "95% interval includes zero", p["surface"])):
        m = significant == flag
        if not m.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=est[m], y=df["label"].to_numpy()[m], mode="markers", name=name,
                marker=dict(size=11, color=fill, line=dict(color=p["series"], width=2)),
                error_x=dict(type="data", symmetric=False, array=(hi - est)[m], arrayminus=(est - lo)[m],
                             color=p["series"], thickness=2, width=0),
                customdata=np.column_stack([lo[m], hi[m], df["nobs"].to_numpy()[m]]),
                hovertemplate="<b>%{y}</b><br>Elasticity %{x:.2f}<br>95% interval %{customdata[0]:.2f} to "
                              "%{customdata[1]:.2f}<br>%{customdata[2]:.0f} monthly observations<extra></extra>",
            )
        )
    fig.add_shape(type="line", xref="x", x0=0, x1=0, yref="paper", y0=0, y1=1, line=dict(color=p["ink"], width=1), layer="below")
    if reference is not None:
        fig.add_shape(type="line", xref="x", x0=reference, x1=reference, yref="paper", y0=0, y1=1,
                      line=dict(color=p["accent"], width=2, dash="dash"), layer="below")
        fig.add_annotation(x=reference, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=reference_label, font=dict(size=12, color=p["ink"]))
    _layout(fig, p, height=max(220, 90 + 36 * len(df)))
    fig.update_layout(margin=dict(t=28, r=16))  # filled vs hollow is explained in the caption under the chart
    fig.update_yaxes(categoryorder="array", categoryarray=list(df["label"]), showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=p["grid"], title_text=xtitle)
    return fig


def _segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a monthly frame into runs of consecutive months."""
    if df.empty:
        return []
    months = df["date"].dt.year * 12 + df["date"].dt.month
    return [g for _, g in df.groupby((months.diff() != 1).cumsum())]


def rolling_chart(roll: pd.DataFrame, prefix: str, title: str, mode: str = "light", *,
                  full_sample: float | None = None, shade_pandemic: bool = True, height: int = 320) -> go.Figure:
    """Rolling-window estimate with its 95% band; gaps are left as gaps."""
    p = PALETTE[mode]
    fig = go.Figure()
    for seg in _segments(roll):
        fig.add_trace(go.Scatter(x=seg["date"], y=seg[f"{prefix}_hi"], mode="lines", line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=seg["date"], y=seg[f"{prefix}_lo"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=p["band"], hoverinfo="skip"))
    full = roll.set_index("date").asfreq("MS") if len(roll) else roll
    if len(roll):
        fig.add_trace(
            go.Scatter(
                x=full.index, y=full[f"{prefix}_elast"], mode="lines", connectgaps=False,
                line=dict(color=p["series"], width=2),
                customdata=np.column_stack([full[f"{prefix}_lo"], full[f"{prefix}_hi"]]),
                hovertemplate="Window ending %{x|%b %Y}<br>Elasticity %{y:.2f}<br>95% interval %{customdata[0]:.2f} "
                              "to %{customdata[1]:.2f}<extra></extra>",
            )
        )
        last = roll.iloc[-1]
        fig.add_annotation(x=last["date"], y=float(last[f"{prefix}_elast"]), text=f"{last[f'{prefix}_elast']:.2f}",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=12, color=p["ink"]))
    _hline(fig, 0, p["ink"])
    if full_sample is not None:
        _hline(fig, full_sample, p["accent"], dash="dash", width=2)
        fig.add_annotation(xref="paper", x=0, xanchor="left", y=full_sample, yanchor="bottom", showarrow=False,
                           text=f"Full sample {full_sample:.2f}", font=dict(size=12, color=p["ink"]))
    if shade_pandemic and len(roll) and roll["date"].min() < pd.Timestamp(PANDEMIC_END):
        fig.add_shape(type="rect", xref="x", x0=PANDEMIC_START, x1=PANDEMIC_END, yref="paper", y0=0, y1=1,
                      fillcolor=p["shade"], line=dict(width=0), layer="below")
        fig.add_annotation(x=PANDEMIC_START, y=1, yref="paper", yanchor="top", xanchor="left", showarrow=False,
                           text="Pandemic", font=dict(size=11, color=p["ink"]))
    return _layout(fig, p, height=height, title=title)


def partial_scatter(pr: pd.DataFrame, slope: float, mode: str = "light", height: int = 360) -> go.Figure:
    """Added-variable plot behind one wage elasticity."""
    p = PALETTE[mode]
    fig = go.Figure(
        go.Scatter(
            x=pr["wage_resid"] * 100, y=pr["emp_resid"] * 100, mode="markers",
            marker=dict(size=8, color=p["series"], opacity=0.65, line=dict(color=p["surface"], width=1)),
            customdata=pr["date"].dt.strftime("%b %Y"),
            hovertemplate="%{customdata}<br>Real wage %{x:+.2f} pts<br>Employment %{y:+.2f} pts<extra></extra>",
        )
    )
    if len(pr):
        xs = np.array([pr["wage_resid"].min(), pr["wage_resid"].max()]) * 100
        fig.add_trace(go.Scatter(x=xs, y=slope * xs, mode="lines", line=dict(color=p["accent"], width=2), hoverinfo="skip"))
        fig.add_annotation(x=xs[1], y=slope * xs[1], text=f"slope {slope:.2f}", showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=12, color=p["ink"]))
    _layout(fig, p, height=height)
    fig.update_xaxes(title_text="Real wage change, net of controls (log points × 100)", showgrid=True, gridcolor=p["grid"])
    fig.update_yaxes(title_text="Employment change, net of controls")
    return fig
