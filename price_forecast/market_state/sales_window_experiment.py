"""Compare trailing 3/6-month sales averages in real-index level equations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import real_index_experiment as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sales_window_results'
old=core.old


def main(start=None):
    out=OUT if start is None else ROOT/'sales_window_from2022_results'
    out.mkdir(exist_ok=True)
    # This is average monthly sales over three completed months, not a quarterly sum.
    old.X['mos_avg3_lag1']=np.log(old.S.shift(1)/old.Q.rolling(3).mean().shift(1))
    available=pd.concat([core.REAL.rename('target'),core.REAL.shift().rename('lag'),old.X[['mos_lag1','mos_avg3_lag1']]],axis=1).dropna().index
    sample=old.COMMON if start is None else available[available>=pd.Timestamp(start)]
    years=[2023,2024,2025] if start is None else list(range(pd.Timestamp(start).year,2026))
    models={};fits=[];folds=[];sensitivity=[];annual=[]
    for kind in ['raw','log']:
        for window,feature in [(6,'mos_lag1'),(3,'mos_avg3_lag1')]:
            name=f'{kind}_avg{window}'
            m=core.fit_level(sample,feature,kind);models[name]=m
            target=core.REAL if kind=='raw' else core.LOGREAL
            shift=core.SHIFTS.mul(pd.Series(m['beta'])).sum(axis=1)
            fitted_y=m['c']+m['rho']*(target.shift()-shift.shift())+shift+m['gamma']*(old.X[feature]-m['x_mean'])
            real_fit=fitted_y if kind=='raw' else 100*np.exp(fitted_y/100)
            observed=core.REAL.loc[sample];pred=real_fit.loc[sample]
            residual=pred-observed
            clean=target.loc[sample]-shift.loc[sample]
            clean_fit=fitted_y.loc[sample]-shift.loc[sample]
            fits.append(dict(model=name,window_months=window,rmse_index_points=float(np.sqrt(np.mean(residual**2))),mae_index_points=float(np.mean(abs(residual))),
                r2_level=float(1-np.sum(residual**2)/np.sum((observed-observed.mean())**2)),r2_clean_level=float(1-np.sum((clean_fit-clean)**2)/np.sum((clean-clean.mean())**2)),
                rho=m['rho'],gamma=m['gamma']))
            nominal=old.P.iloc[0]*real_fit/100*old.C.reindex(old.P.index)/old.C.loc[old.P.index[0]]
            monthly_fit=100*np.log(nominal/old.P.shift())
            for year in years:
                annual.append(dict(model=name,year=year,actual_growth_pct=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1)),
                    fitted_accumulated_growth_pct=float(100*np.expm1(monthly_fit.loc[str(year)].sum()/100))))
            for n in range(30,len(sample)):
                train=sample[:n];origin=train[-1];dates=sample[n:n+12]
                fm=core.fit_level(train,feature,kind)
                forecast=core.predict_level(fm,origin,dates)
                for h in [1,6,12]:
                    if len(dates)<h:continue
                    folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(forecast.iloc[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin]),
                        future_event_months=int(old.F.loc[dates[:h]].any(axis=1).sum()),gamma=fm['gamma'],rho=fm['rho']))
            for substart in [sample[0],pd.Timestamp('2023-01-01')]:
                subsample=sample[sample>=substart];sm=core.fit_level(subsample,feature,kind)
                sensitivity.append(dict(model=name,start=str(substart.date()),n=len(subsample),gamma=sm['gamma'],rho=sm['rho']))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x)),bias='mean').reset_index()
    calm=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x))).reset_index()
    if start is None:
        previous=pd.read_csv(ROOT/'real_index_results/backtest_metrics.csv')
        for name,ref in [('raw_avg6','real_index_market'),('log_avg6','log_index_market')]:
            assert np.allclose(metrics[metrics.model==name].sort_values('horizon').rmse,previous[previous.model==ref].sort_values('horizon').rmse)
    assert old.X.loc[sample,['mos_lag1','mos_avg3_lag1']].notna().all().all()
    pd.DataFrame(fits).to_csv(out/'historical_fit_metrics.csv',index=False)
    metrics.to_csv(out/'backtest_metrics.csv',index=False);f.to_csv(out/'backtest_folds.csv',index=False)
    calm.to_csv(out/'non_event_monthly_metrics.csv',index=False)

    if start is not None:
        previous_folds=pd.read_csv(ROOT/'sales_window_results/backtest_folds.csv')
        joined=previous_folds.merge(f,on=['model','origin','horizon'],suffixes=('_old','_new'))
        comparison=joined.groupby(['model','horizon']).agg(n=('origin','size'),old_rmse=('error_pp_old',lambda x:np.sqrt(np.mean(x*x))),new_rmse=('error_pp_new',lambda x:np.sqrt(np.mean(x*x)))).reset_index()
        comparison.to_csv(out/'matching_origins_comparison.csv',index=False)

    pd.DataFrame(annual).to_csv(out/'historical_annual_comparison.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'sample_sensitivity.csv',index=False)
    (out/'parameters.json').write_text(json.dumps(models,indent=2))
    (out/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(sample),start=str(sample[0].date()),end=str(sample[-1].date()),
        change='Trailing average monthly sales over 3 rather than 6 completed months. Same one-month lag in stock, same units, events and matching sample.',
        validation='Expanding-window forecasts from 30 training months; actual future CPI, first available market indicator held fixed, no future events supplied; overlapping windows and retrospective event calendar.',
        quarterly_meaning='Rolling three-month average, not fixed calendar quarters or three-month sum.',production_model_changed=False),indent=2))
    print('FIT');print(pd.DataFrame(fits).round(5).to_string(index=False))
    print('BACKTEST');print(metrics.round(4).to_string(index=False))
    print('CALM');print(calm.round(4).to_string(index=False))
    print('ANNUAL');print(pd.DataFrame(annual).round(4).to_string(index=False))
    print('SENSITIVITY');print(pd.DataFrame(sensitivity).round(5).to_string(index=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--start',choices=['2022-01-01'],default=None)
    main(parser.parse_args().start)
