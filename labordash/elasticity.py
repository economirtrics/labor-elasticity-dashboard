"""Elasticity estimation.

Model, for unit i (an industry or a state) and differencing horizon h months:

    dlog(E_it) = a_i + eta_i * dlog(W_it / P_t) + gamma_i * dlog(Y_t) + e_it

    eta_i    employment elasticity with respect to the real hourly wage
    gamma_i  employment elasticity with respect to aggregate demand

Both are estimated by OLS with Newey-West (HAC) standard errors, per unit, over
the full sample and over rolling windows. A pooled two-way fixed-effects version
identifies a common eta from within-unit, within-month variation.

The linear algebra is written out in NumPy so the rolling estimator stays fast
and every step is inspectable; ``tests/test_crosscheck_statsmodels.py`` checks the
standard errors against statsmodels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import PANDEMIC_END, PANDEMIC_START

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Core linear algebra
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    params: np.ndarray
    se: np.ndarray
    resid: np.ndarray
    nobs: int
    r2: float
    lags: int


def nw_lags(nobs: int, horizon: int) -> int:
    """Newey-West truncation lag: overlap from h-month differences plus the usual rule."""
    return int((horizon - 1) + np.floor(4 * (nobs / 100.0) ** (2.0 / 9.0)))


def hac_cov(X: np.ndarray, resid: np.ndarray, maxlags: int) -> np.ndarray:
    """Newey-West covariance with a Bartlett kernel (no small-sample correction)."""
    xu = X * resid[:, None]
    S = xu.T @ xu
    for lag in range(1, min(maxlags, len(resid) - 1) + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        G = xu[lag:].T @ xu[:-lag]
        S += w * (G + G.T)
    bread = np.linalg.pinv(X.T @ X)
    return bread @ S @ bread


def cluster_cov(X: np.ndarray, resid: np.ndarray, groups: np.ndarray, k_params: int | None = None) -> np.ndarray:
    """One-way cluster-robust covariance with the G/(G-1) * (N-1)/(N-K) correction."""
    n, k = X.shape
    k_params = k if k_params is None else k_params
    xu = X * resid[:, None]
    codes, _ = pd.factorize(groups)
    g = codes.max() + 1
    sums = np.zeros((g, k))
    np.add.at(sums, codes, xu)
    bread = np.linalg.pinv(X.T @ X)
    correction = g / (g - 1.0) * (n - 1.0) / (n - k_params)
    return correction * bread @ (sums.T @ sums) @ bread


def ols_hac(y: np.ndarray, X: np.ndarray, maxlags: int) -> Fit:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = hac_cov(X, resid, maxlags)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan
    return Fit(beta, np.sqrt(np.clip(np.diag(cov), 0, None)), resid, len(y), r2, maxlags)


# --------------------------------------------------------------------------- #
# Sample selection and design matrix
# --------------------------------------------------------------------------- #
def pandemic_mask(dates: pd.Series | pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """True for observations whose h-month differencing window touches the pandemic."""
    d = pd.DatetimeIndex(dates)
    start = pd.Timestamp(PANDEMIC_START)
    end = pd.Timestamp(PANDEMIC_END) + pd.DateOffset(months=horizon)
    return np.asarray((d >= start) & (d < end))


def _month_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    m = np.asarray(dates.month)
    return (m[:, None] == np.arange(2, 13)[None, :]).astype(float)


def _design(df: pd.DataFrame, month_dummies: bool) -> tuple[np.ndarray, np.ndarray]:
    """y and X = [1, dlog real wage, dlog demand, (11 month dummies)]."""
    y = df["dlog_emp"].to_numpy(float)
    cols = [np.ones(len(df)), df["dlog_rwage"].to_numpy(float), df["dlog_demand"].to_numpy(float)]
    X = np.column_stack(cols)
    if month_dummies:
        X = np.column_stack([X, _month_dummies(pd.DatetimeIndex(df["date"]))])
    return y, X


def prepare(df: pd.DataFrame, horizon: int, exclude_pandemic: bool) -> pd.DataFrame:
    df = df.sort_values("date")
    if exclude_pandemic:
        df = df[~pandemic_mask(df["date"], horizon)]
    return df.reset_index(drop=True)


def needs_month_dummies(seasonally_adjusted: bool, horizon: int) -> bool:
    """NSA data differenced over less than a year still carries seasonality."""
    return (not seasonally_adjusted) and horizon < 12


# --------------------------------------------------------------------------- #
# Per-unit estimates
# --------------------------------------------------------------------------- #
def _row(fit: Fit) -> dict:
    eta, gamma = fit.params[1], fit.params[2]
    se_eta, se_gamma = fit.se[1], fit.se[2]
    return {
        "wage_elast": eta, "wage_se": se_eta,
        "wage_lo": eta - Z95 * se_eta, "wage_hi": eta + Z95 * se_eta,
        "demand_elast": gamma, "demand_se": se_gamma,
        "demand_lo": gamma - Z95 * se_gamma, "demand_hi": gamma + Z95 * se_gamma,
        "nobs": fit.nobs, "r2": fit.r2, "hac_lags": fit.lags,
    }


def estimate_unit(df: pd.DataFrame, horizon: int, *, exclude_pandemic: bool = True, month_dummies: bool = False) -> dict | None:
    """Full-sample elasticities for one unit's rows of the model frame."""
    df = prepare(df, horizon, exclude_pandemic)
    y, X = _design(df, month_dummies)
    if len(y) < max(24, 3 * X.shape[1]):
        return None
    fit = ols_hac(y, X, nw_lags(len(y), horizon))
    return {**_row(fit), "start": df["date"].iloc[0], "end": df["date"].iloc[-1]}


def estimate_all(frame: pd.DataFrame, horizon: int, *, exclude_pandemic: bool = True, month_dummies: bool = False) -> pd.DataFrame:
    """One row of elasticities per unit."""
    rows = []
    for (unit, label), g in frame.groupby(["unit", "label"], sort=False):
        res = estimate_unit(g, horizon, exclude_pandemic=exclude_pandemic, month_dummies=month_dummies)
        if res:
            rows.append({"unit": unit, "label": label, **res})
    return pd.DataFrame(rows)


def rolling_unit(
    df: pd.DataFrame,
    horizon: int,
    window: int = 60,
    *,
    exclude_pandemic: bool = True,
    month_dummies: bool = False,
) -> pd.DataFrame:
    """Re-estimate the unit regression on every trailing ``window``-month sample.

    Windows are defined in calendar months and each estimate is dated at the
    window's last month. The first estimate waits for a full window of history;
    after that, a window is skipped when sample exclusions leave it with fewer
    than 60% of its months.
    """
    df = prepare(df, horizon, exclude_pandemic)
    y, X = _design(df, month_dummies)
    dates = pd.DatetimeIndex(df["date"])
    min_obs = max(24, 3 * X.shape[1], int(np.ceil(0.6 * window)))
    all_ends = pd.date_range(dates.min(), dates.max(), freq="MS") if len(dates) else []
    present = set(dates)

    rows = []
    for end in all_ends:
        if end not in present:
            continue
        lo = end - pd.DateOffset(months=window - 1)
        if lo < dates.min():  # not yet a full window of history
            continue
        sel = np.asarray((dates >= lo) & (dates <= end))
        n = int(sel.sum())
        if n < min_obs:
            continue
        fit = ols_hac(y[sel], X[sel], nw_lags(n, horizon))
        rows.append({"date": end, **_row(fit)})
    return pd.DataFrame(rows)


def partial_regression(df: pd.DataFrame, horizon: int, *, exclude_pandemic: bool = True, month_dummies: bool = False) -> pd.DataFrame:
    """Frisch-Waugh residuals: employment and wage changes net of the other regressors.

    The slope of ``emp_resid`` on ``wage_resid`` equals the wage elasticity, so
    plotting them shows exactly which observations drive the estimate.
    """
    df = prepare(df, horizon, exclude_pandemic)
    y, X = _design(df, month_dummies)
    Z = np.delete(X, 1, axis=1)
    x = X[:, 1]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return pd.DataFrame({"date": df["date"].to_numpy(), "wage_resid": rx, "emp_resid": ry})


# --------------------------------------------------------------------------- #
# Pooled two-way fixed effects
# --------------------------------------------------------------------------- #
def pooled_twfe(frame: pd.DataFrame, horizon: int, *, exclude_pandemic: bool = True, month_dummies: bool = False) -> dict | None:
    """Common wage elasticity with unit and month fixed effects.

        dlog(E_it) = a_i + l_t + eta * dlog(w_it) + e_it

    Month effects absorb every aggregate shock (including the demand proxy), so
    eta is identified from units whose real wages moved differently in the same
    month. Standard errors are clustered by unit; with few units they are
    optimistic, so the interval uses a t distribution with G-1 degrees of freedom.
    """
    df = frame.sort_values(["unit", "date"])
    if exclude_pandemic:
        df = df[~pandemic_mask(df["date"], horizon)]
    units = df["unit"].to_numpy()
    n_units = len(pd.unique(units))
    if n_units < 3 or len(df) < 60:
        return None

    y = df["dlog_emp"].to_numpy(float)
    x = df["dlog_rwage"].to_numpy(float)
    unit_d = pd.get_dummies(df["unit"]).to_numpy(float)
    time_d = pd.get_dummies(df["date"]).to_numpy(float)
    D = np.column_stack([unit_d, time_d])
    if month_dummies:  # unit-specific seasonality for NSA data at short horizons
        um = pd.get_dummies(df["unit"].astype(str) + "_" + pd.DatetimeIndex(df["date"]).month.astype(str))
        D = np.column_stack([D, um.to_numpy(float)])

    rank_d = np.linalg.matrix_rank(D)
    ry = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    sxx = float(rx @ rx)
    if sxx <= 0:
        return None
    eta = float(rx @ ry) / sxx
    resid = ry - eta * rx
    cov = cluster_cov(rx[:, None], resid, units, k_params=rank_d + 1)
    se = float(np.sqrt(cov[0, 0]))
    tcrit = float(stats.t.ppf(0.975, n_units - 1))
    return {
        "wage_elast": eta, "wage_se": se,
        "wage_lo": eta - tcrit * se, "wage_hi": eta + tcrit * se,
        "nobs": int(len(y)), "units": int(n_units),
        "months": int(df["date"].nunique()),
        "within_r2": 1.0 - float(resid @ resid) / float(ry @ ry) if float(ry @ ry) > 0 else np.nan,
    }
