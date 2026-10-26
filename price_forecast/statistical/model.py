"""Inflation-anchored AR(1), with explicit soft forecast guidance.

Run from any directory. No monthly forecast deliverable is required.
Historical loss and guidance loss are averages, in separately declared units.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
D = pd.read_csv(ROOT.parent / 'inputs/monthly_data.csv', index_col=0, parse_dates=True)
P = D['index'].dropna()
C = D['cpi_index_dec2020']
R = (100 * (np.log(P).diff() - np.log(C).diff())).reindex(P.index)
Z = pd.concat([R.rename('real'), R.shift().rename('lag')], axis=1).dropna()
YEARS = np.arange(2026, 2031)
PI = np.array([100 * (C.loc[f'{y}-12-01'] / C.loc[f'{y-1}-12-01'] - 1) for y in YEARS])
TARGET = np.r_[10., PI[1:] + [1.5, 2., 1.8, 1.8]]
WEIGHTS = [0., .03, .1, .3, 1., 3., 10.]
GUIDANCE_SCALE_PP = 1.
MAX_TRAIN_RMSE_INCREASE = .10


def predict_real(z, lag):
    mu, phi = z
    return (1 - phi) * mu + phi * lag


def path(z, origin, dates):
    real = float(R.loc[origin])
    level = float(P.loc[origin])
    result = []
    for date in dates:
        real = predict_real(z, real)
        previous = date - pd.offsets.MonthBegin(1)
        inflation = 100 * np.log(C.loc[date] / C.loc[previous])
        level *= np.exp((inflation + real) / 100)
        result.append(level)
    return pd.Series(result, index=dates, dtype=float)


def annual(z):
    dates = pd.date_range(P.index[-1] + pd.offsets.MonthBegin(1), '2030-12-01', freq='MS')
    full = pd.concat([P, path(z, P.index[-1], dates)])
    return np.array([100 * (full.loc[f'{y}-12-01'] / full.loc[f'{y-1}-12-01'] - 1) for y in YEARS])


def fit(sample, weight=0., scale=1., mask=None):
    if mask is None:
        mask = np.ones(len(YEARS), dtype=bool)

    def residual(z):
        history = (predict_real(z, sample.lag.to_numpy()) - sample.real.to_numpy()) / (scale * np.sqrt(len(sample)))
        if weight == 0:
            return history
        guide = (annual(z)[mask] - TARGET[mask]) * np.sqrt(weight / mask.sum()) / GUIDANCE_SCALE_PP
        return np.r_[history, guide]

    # Stationarity only; negative phi permits a reversal after an unusually strong month.
    starts = [[.15, -.4], [.15, .4], [.6, .8]]
    fits = [least_squares(residual, s, bounds=([-5., -.95], [5., .95]),
                          max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11) for s in starts]
    valid = [f for f in fits if f.success]
    if not valid:
        raise RuntimeError('Optimizer failed')
    return min(valid, key=lambda f: np.sum(f.fun ** 2)).x


def rmse(z, sample=Z):
    return float(np.sqrt(np.mean((predict_real(z, sample.lag) - sample.real) ** 2)))


def main():
    baseline = fit(Z)
    scale = rmse(baseline)
    rows, params = [], {}
    for weight in WEIGHTS:
        z = fit(Z, weight, scale)
        params[weight] = z
        growth = annual(z)
        rows.append(dict(weight=weight, mu_log_pp_month=z[0], phi=z[1],
                         historical_rmse_log_pp_month=rmse(z),
                         historical_rmse_change_pct=100 * (rmse(z) / scale - 1),
                         guidance_rmse_pp=float(np.sqrt(np.mean((growth - TARGET) ** 2)))))
    sensitivity = pd.DataFrame(rows)
    eligible = sensitivity[sensitivity.historical_rmse_change_pct <= 100 * MAX_TRAIN_RMSE_INCREASE + 1e-9]
    chosen = float(eligible.sort_values(['guidance_rmse_pp', 'weight']).iloc[0].weight)
    selected = params[chosen]
    sensitivity['selected'] = sensitivity.weight == chosen
    sensitivity.to_csv(OUT / 'weight_sensitivity.csv', index=False)
    comparison = pd.DataFrame(dict(year=YEARS, inflation_pct=PI, guidance_pct=TARGET,
                                   unconstrained_pct=annual(baseline), guided_pct=annual(selected)))
    comparison['guided_premium_pp'] = comparison.guided_pct - comparison.inflation_pct
    comparison['deviation_from_guidance_pp'] = comparison.guided_pct - comparison.guidance_pct
    comparison.to_csv(OUT / 'annual_comparison.csv', index=False)
    Z.assign(fitted_unconstrained=predict_real(baseline, Z.lag),
             fitted_guided=predict_real(selected, Z.lag)).to_csv(OUT / 'historical_fit.csv', index_label='month')
    omission = []
    for i, omitted in enumerate(YEARS):
        mask = YEARS != omitted
        z = fit(Z, chosen, scale, mask)
        for year, growth in zip(YEARS, annual(z)):
            omission.append(dict(omitted_guidance_year=int(omitted), year=int(year), growth_pct=growth))
    pd.DataFrame(omission).to_csv(OUT / 'leave_one_guidance_out.csv', index=False)

    # Only baseline can be honestly backtested: no historical expert-guidance vintages exist.
    # Future actual CPI is supplied identically to all benchmarks (conditional test).
    folds = []
    for origin in Z.index:
        train = Z.loc[:origin]
        end = origin + pd.DateOffset(months=12)
        if len(train) < 24 or end > P.index[-1]:
            continue
        z = fit(train)
        dates = pd.date_range(origin + pd.offsets.MonthBegin(1), end, freq='MS')
        actual = 100 * (P.loc[end] / P.loc[origin] - 1)
        forecasts = dict(ar1=100 * (path(z, origin, dates).iloc[-1] / P.loc[origin] - 1),
                         inflation_only=100 * (C.loc[end] / C.loc[origin] - 1),
                         constant_real_growth=100 * ((C.loc[end] / C.loc[origin]) * np.exp(12 * train.real.mean() / 100) - 1))
        for model, value in forecasts.items():
            folds.append(dict(origin=str(origin.date()), target=str(end.date()), model=model,
                              actual_growth_pct=actual, forecast_growth_pct=value, error_pp=value - actual))
    folds = pd.DataFrame(folds)
    folds.to_csv(OUT / 'baseline_backtest_folds.csv', index=False)
    metrics = folds.groupby('model').error_pp.agg(n='size', rmse=lambda x: np.sqrt(np.mean(x*x)), mae=lambda x: np.mean(abs(x)))
    metrics.to_csv(OUT / 'baseline_backtest_metrics.csv')
    result = dict(n_history=len(Z), history_start=str(Z.index[0].date()), history_end=str(Z.index[-1].date()),
                  selected_weight=chosen, selection_rule='minimum guidance RMSE among weights with training RMSE deterioration <=10%; not cross-validated',
                  monthly_historical_error_scale=scale, guidance_error_scale_pp=GUIDANCE_SCALE_PP,
                  baseline_parameters=baseline.tolist(), guided_parameters=selected.tolist(),
                  baseline_rmse=scale, guided_rmse=rmse(selected),
                  backtest_note='Unconstrained model only; 12-month horizons overlap; realized CPI is supplied. Guided forecast has no honest historical backtest.',
                  leave_one_out_note='Fixed selected weight, refitted parameters; diagnostic of guidance influence, not validation.')
    assert np.isfinite(comparison.select_dtypes('number')).all().all()
    assert abs(selected[1]) < 1
    assert rmse(selected) <= scale * 1.10 + 1e-8
    # Check recursion and closed form agree for the real-growth component.
    real = float(R.iloc[-1])
    for h in range(1, 53):
        real = predict_real(selected, real)
        assert np.isclose(real, selected[0] + selected[1]**h * (R.iloc[-1] - selected[0]))
    result['numerical_checks_passed'] = True
    (OUT / 'diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label, style in [('guidance_pct', 'Expert guidance', 'k--'),
                                  ('unconstrained_pct', 'Historical fit only', 'o-'),
                                  ('guided_pct', 'Historical fit + guidance', 's-')]:
        ax.plot(comparison.year, comparison[column], style, label=label)
    ax.set(xticks=YEARS, ylabel='Price growth, Dec/Dec (%)', title='Inflation-anchored AR(1): effect of forecast guidance')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / 'comparison.svg'); plt.close(fig)
    print(comparison.round(3).to_string(index=False))
    print(sensitivity.round(4).to_string(index=False))
    print(metrics.round(3).to_string())


if __name__ == '__main__':
    main()
