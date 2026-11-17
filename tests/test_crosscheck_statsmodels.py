"""Check the hand-written covariance estimators against statsmodels.

Skipped automatically when statsmodels is not installed.
"""
import numpy as np
import pytest

from labordash import elasticity as el

sm = pytest.importorskip("statsmodels.api")


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    u = np.convolve(rng.standard_normal(n + 5), np.ones(6) / 6, mode="valid")  # serially correlated
    y = X @ np.array([0.1, -0.4, 0.8]) + u
    return y, X


@pytest.mark.parametrize("lags", [0, 4, 15])
def test_hac_matches_statsmodels(lags):
    y, X = _data()
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    fit = el.ols_hac(y, X, lags)
    assert np.allclose(fit.params, ref.params)
    assert np.allclose(fit.se, ref.bse, rtol=1e-8)


def test_cluster_matches_statsmodels():
    y, X = _data(seed=1)
    groups = np.repeat(np.arange(15), 20)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    cov = el.cluster_cov(X, y - X @ beta, groups)
    assert np.allclose(np.sqrt(np.diag(cov)), ref.bse, rtol=1e-8)
