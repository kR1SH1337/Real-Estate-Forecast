"""Diagnose observed monthly inputs; never changes the forecasting model.

Install optional requirements-diagnostics.txt, then run from any directory:
python scripts/stationarity_diagnostics.py
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews, coint
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cash_model import read_sources, fit_model


def continuous(s):
    if s.isna().any() or not s.index.equals(pd.date_range(s.index[0], s.index[-1], freq='MS')):
        raise ValueError('Tests require a complete regular monthly segment')
    return s.to_numpy(float)


def tests(s):
    y = continuous(s)
    out = {'n': len(y), 'start': str(s.index[0].date()), 'end': str(s.index[-1].date()),
           'adf': [], 'kpss': [], 'zivot_andrews': []}
    for regression in ['c', 'ct']:
        for selection in ['AIC', 'BIC', None]:
            for lag in ([4] if selection else range(7)):
                v = adfuller(y, maxlag=lag, regression=regression, autolag=selection)
                stored = adfuller(y, maxlag=int(v[2]), regression=regression, autolag=None, store=True)[-1]
                lb = acorr_ljungbox(stored.resols.resid, lags=[6], return_df=True)
                out['adf'].append({'deterministic': regression, 'selection': selection or 'fixed',
                    'maxlag': lag, 'lag': int(v[2]), 'stat': float(v[0]), 'p': float(v[1]),
                    'nobs': int(v[3]), 'critical': v[4],
                    'auxiliary_ljung_box_6_p_descriptive': float(lb.lb_pvalue.iloc[0])})
        for lag in ['auto', 0, 1, 2, 3, 4, 6]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                v = kpss(y, regression=regression, nlags=lag)
            out['kpss'].append({'deterministic': regression, 'selection': str(lag), 'lag': int(v[2]),
                'stat': float(v[0]), 'p': float(v[1]), 'critical': v[3],
                'warnings': [str(w.message) for w in caught]})
        for selection in ['AIC', 'BIC']:
            v = zivot_andrews(y, trim=.15, maxlag=4, regression=regression, autolag=selection)
            out['zivot_andrews'].append({'deterministic': regression, 'selection': selection,
                'stat': float(v[0]), 'p': float(v[1]), 'critical': v[2], 'lag': int(v[3]),
                'break_date': str(s.index[v[4]].date())})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reference_results/stationarity.json')
    args = parser.parse_args()
    data, rates = read_sources(ROOT/'data/cash_sales.csv', ROOT/'data/macro_scenario.xlsx',
                               ROOT/'data/price_index.xlsx')
    # Latest contiguous observed sales segment, without interpolation or time compression.
    missing = np.flatnonzero(data.real.isna())
    sample = data.iloc[missing[-1]+1:] if len(missing) else data
    out = {'statsmodels_version': statsmodels.__version__, 'primary_sample': 'latest continuous observed segment',
        'missing_sales_dates': [str(t.date()) for t in data.index[data.real.isna()]],
        'future_scenario_excluded': True, 'tests': {}, 'sources_sha256': {}}
    # The September observation only requires rates through August. The supplied
    # macro workbook contains a future scenario: do not treat September raw KS as fact.
    last_rate = data.index[-1] - pd.offsets.MonthBegin(1)
    out['last_raw_rate_tested'] = str(last_rate.date())
    series = {'real_sales':sample.real, 'ks_avg3':sample.ks_avg3,
        'ks_raw':rates.reindex(sample.index[sample.index <= last_rate]), 'nominal_sales':sample.nominal,
        'price_factor':sample.price_factor, 'ks_avg3_full':data.ks_avg3,
        'ks_raw_full':rates.reindex(data.index[data.index <= last_rate])}
    for name, s in series.items():
        out['tests'][name] = tests(s)
        if name in ['real_sales','ks_avg3','ks_raw']:
            out['tests'][name+'_diff'] = tests(s.diff().iloc[1:])
    # Missing-February sensitivity; interpolation is diagnostic only, never used for training.
    out['tests']['real_sales_interpolated_sensitivity'] = tests(data.real.interpolate(method='linear',limit_area='inside'))
    # Ordinary ADF critical values for estimated regression residuals are not valid
    # cointegration critical values. Report MacKinnon Engle-Granger OLS results instead.
    out['engle_granger_exploratory'] = []
    for trend in ['c','ct']:
        for select in ['AIC','BIC']:
            v = coint(sample.real, sample.ks_avg3, trend=trend, maxlag=4, autolag=select)
            out['engle_granger_exploratory'].append({'trend':trend,'selection':select,
                'stat':float(v[0]),'p':float(v[1]),'critical_1_5_10':v[2].tolist()})
    # Descriptive residual stationarity: KPSS only, with generated-residual caveat.
    fitted = fit_model(data.real, data.ks_avg3)
    resid = sample.real - fitted['coef'][0] - fitted['coef'][1]*sample.ks_avg3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');v=kpss(resid, regression='c', nlags='auto')
    out['residual_kpss_descriptive'] = {'stat':float(v[0]),'p':float(v[1]),'lags':int(v[2]),
        'warning':'Estimated full-sample Huber residuals; ordinary KPSS inference is descriptive, not a calibrated cointegration test.',
        'table_warnings':[str(w.message) for w in caught]}
    # July 2024: descriptive split, not an extra ordinary-ADF test of adjusted residuals.
    out['july_2024_summary'] = {}
    for name in ['real_sales','ks_avg3']:
        s=series[name];out['july_2024_summary'][name]={}
        for label,g in [('before',s.loc[:'2024-06-01']),('after',s.loc['2024-07-01':])]:
            out['july_2024_summary'][name][label]={'n':len(g),'mean':float(g.mean()),'std':float(g.std())}
            details = out['july_2024_summary'][name][label]
            details['adf_c_maxlag2'] = {}
            for selection in ['AIC','BIC']:
                v = adfuller(continuous(g),maxlag=2,regression='c',autolag=selection)
                details['adf_c_maxlag2'][selection]={'stat':float(v[0]),'p':float(v[1]),'lag':int(v[2])}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');v=kpss(continuous(g),regression='c',nlags='auto')
            details['kpss_c_auto']={'stat':float(v[0]),'p':float(v[1]),'lag':int(v[2]),
                                  'warnings':[str(w.message) for w in caught]}
    for filename in ['cash_sales.csv','macro_scenario.xlsx','price_index.xlsx']:
        out['sources_sha256'][filename]=hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest()
    out['notes'] = ['ADF H0: unit root; KPSS H0: level/trend stationarity.',
        'c: constant; ct: constant plus deterministic linear trend.',
        'ADF autolag max=4; fixed lags 0..6 are sensitivity, not opportunities to pick a preferred result.',
        'KPSS p=.10 means p>=.10 and p=.01 means p<=.01 when table warning is present.',
        'Zivot-Andrews allows one endogenously selected break; July 2024 is not imposed.',
        'The smoothed KS factor is mechanically persistent; raw KS is reported separately.',
        'Only about 3.5 annual cycles: seasonal roots and regime stability are not resolved.',
        'Engle-Granger is exploratory and only interpretable as cointegration if both inputs are I(1).',
        'No forecasts or model coefficients were changed by this script.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,r in out['tests'].items():
        a=next(v for v in r['adf'] if v['selection']=='BIC' and v['deterministic']=='c')
        k=next(v for v in r['kpss'] if v['selection']=='auto' and v['deterministic']=='c')
        print(f'{name:38} n={r["n"]:2} ADF(BIC) lag={a["lag"]} p={a["p"]:.4f}; KPSS lag={k["lag"]} stat={k["stat"]:.4f} p={k["p"]:.4f}')
    print(args.output)


if __name__ == '__main__':
    main()
