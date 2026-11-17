# Labor Market Elasticity Dashboard

An interactive Streamlit app that pulls the latest U.S. labor-market data from the FRED and BLS APIs every time it loads, then estimates how employment responds to wages and to aggregate demand, by industry and by state, over the full sample and over rolling windows.

Nothing is bundled or hard-coded: when BLS publishes a new jobs report, the next page load picks it up and every elasticity is re-estimated.

## What it shows

| Tab | Contents |
|---|---|
| **Overview** | Unemployment, CPI inflation, real wage growth and payroll growth, each stamped with the month it refers to |
| **Elasticities** | Wage and demand elasticities for every industry or state with 95% intervals, a pooled panel estimate, and an added-variable plot showing which months drive any one estimate |
| **Over time** | The same elasticities re-estimated on a rolling window, so the latest window always ends at the newest data |
| **Data and methods** | Per-series freshness (source, last observation, fetch time, status), CSV downloads, and the model written out |

## Quick start

```bash
git clone <your-repo-url> && cd labor-elasticity-dashboard
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app works without API keys. Free keys raise the rate limits and are worth adding before sharing a link:

| Source | Without a key | With a key | Get one |
|---|---|---|---|
| FRED | Public CSV endpoint | JSON API, 120 requests per minute | <https://fred.stlouisfed.org/docs/api/api_key.html> |
| BLS | API v1: 25 requests per day, 10 years each | API v2: 500 per day, 20 years each | <https://data.bls.gov/registrationEngine/> |

Set them as environment variables (`FRED_API_KEY`, `BLS_API_KEY`), in `.streamlit/secrets.toml`, or paste them into the sidebar.

## The model

For each unit *i* (an industry or a state) and a differencing horizon of *h* months (1, 3 or 12):

```
Δh log(E_it) = α_i + η_i · Δh log(W_it / P_t) + γ_i · Δh log(Y_t) + ε_it
```

- `E` employment, `W` average hourly earnings, `P` the CPI, `Y` an aggregate demand proxy (real consumer spending or industrial production).
- **η, wage elasticity:** percent change in employment per 1% change in the real hourly wage.
- **γ, demand elasticity:** percent change in employment per 1% change in aggregate demand.
- **Inference:** Newey-West (HAC) standard errors. The truncation lag is `(h − 1) + ⌊4(T/100)^(2/9)⌋`, which covers the serial correlation that overlapping *h*-month changes create by construction.
- **Rolling estimates:** the regression is re-run on every trailing window (36 to 120 months). Windows that lose more than 40% of their months to sample exclusions are skipped rather than estimated on scraps.
- **Pooled estimate:** `Δh log(E_it) = α_i + λ_t + η · Δh log(w_it) + ε_it` with unit and month fixed effects and unit-clustered standard errors. Month effects absorb every economy-wide shock, so η is identified only from units whose real wages moved differently in the same month.

### Data handling choices

- **Pandemic composition effect.** In spring 2020 low-wage workers were laid off first, so average hourly earnings jumped while employment collapsed. Left in, that episode alone produces a large spurious negative elasticity. By default the app drops every observation whose differencing window overlaps March 2020 to June 2021; a sidebar toggle puts them back.
- **Seasonality.** Industry series are seasonally adjusted at source. State hourly earnings are published only unadjusted, so state models add month dummies at horizons under 12 months (12-month changes difference seasonality out).
- **Missing months.** Every series is placed on a gap-free monthly index before differencing, so a missing month removes the observations that touch it instead of silently turning a 12-month change into a 13-month one. Short gaps in the CPI (for example October 2025, which was never collected) are bridged log-linearly so the deflator stays usable.
- **Resilience.** Each series is cached on disk for six hours. If a source is down, the last good copy is used and the series is flagged as stale in the app.

### How to read the numbers

These are **reduced-form** elasticities. Wages and employment are determined jointly, so one equation cannot separate the labor demand curve from the labor supply curve. A negative η means wage-side shocks dominated in that unit and period (movement along labor demand); a positive η means demand-side shocks dominated (jobs and pay rising together). Average hourly earnings also move when the mix of workers changes, not only when pay does.

Two inference caveats, both visible in the test suite's Monte Carlo: Newey-West intervals under-cover with 12-month overlapping changes (roughly 85% coverage for a nominal 95% interval in simulation), and unit-clustered errors are unreliable with only ten industries, so the pooled interval uses a *t* distribution with G − 1 degrees of freedom.

## Project layout

```
app.py                     Streamlit front end
labordash/
  config.py                Series catalog: every FRED and BLS identifier in one place
  sources.py               FRED and BLS API clients, retries, disk cache with stale fallback
  pipeline.py              Extract and transform: tidy panel, real wages, log changes
  elasticity.py            OLS + Newey-West, rolling windows, pooled two-way fixed effects
  charts.py                Plotly figures
  refresh.py               Command-line refresh that writes dated CSV snapshots
tests/                     pytest suite (see below)
```

Adding an industry, or swapping a series, is a one-line change in `labordash/config.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The app only ever displays live data. The test suite is the one place simulated data appears: it generates series with a *known* elasticity and checks that

- the estimator recovers the planted value at every horizon, and is unbiased across repeated draws;
- rolling windows track a planted structural break;
- month dummies remove planted seasonality that otherwise flips the sign of the estimate;
- the pooled fixed-effects estimator recovers a common elasticity;
- the hand-written Newey-West and cluster-robust covariances are cross-checked against `statsmodels`;
- the FRED JSON, FRED CSV and BLS clients parse their wire formats, respect API limits, handle missing-value markers, and fall back to cache during an outage.

## Scheduled snapshots

```bash
python -m labordash.refresh --dimension industries --horizon 12
```

pulls fresh data and writes `data/snapshots/elasticities_<dimension>_h<horizon>_<YYYY-MM>.csv` plus the rolling estimates, ready to be run from cron or a scheduled GitHub Action to build an archive of how the estimates evolve.

## Deploying

The app runs as-is on Streamlit Community Cloud: point it at this repository with `app.py` as the entry file and add `FRED_API_KEY` and `BLS_API_KEY` under the app's secrets.

## Roadmap

1. **Policy module.** Difference-in-differences and synthetic control for state minimum-wage changes, using the state panel already loaded here.
2. **Identification.** Shift-share (Bartik) instruments for local labor demand to separate demand and supply elasticities.
3. **Finer cuts.** Metro areas and three-digit industries from the same CES series families.

## Data sources

U.S. Bureau of Labor Statistics (Current Employment Statistics, State and Area Employment, Current Population Survey, Consumer Price Index); U.S. Bureau of Economic Analysis (real personal consumption expenditures); Board of Governors of the Federal Reserve System (industrial production); retrieved through FRED, Federal Reserve Bank of St. Louis, and the BLS Public Data API. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
