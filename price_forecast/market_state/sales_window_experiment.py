"""Compare trailing 3/6-month sales averages in real-index level equations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import real_index_experiment as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sales_window_results'
old=core.old


def main():
    OUT.mkdir(exist_ok=True)
    # This is average monthly sales over three completed months, not a quarterly sum.
    old.X['mos_avg3_lag1']=np.log(old.S.shift(1)/old.Q.rolling(3).mean().shift(1))
    models={};fits=[];folds=[];sensitivity=[];annual=[]
    for kind in ['raw','log']:
        for window,feature in [(6,'mos_lag1'),(3,'mos_avg3_lag1')]:
            name=f'{kind}_avg{window}'
            m=core.fit_level(old.COMMON,feature,kind);models[name]=m
            target=core.REAL if kind=='raw' else core.LOGREAL
            shift=core.SHIFTS.mul(pd.Series(m['beta'])).sum(axis=1)
            fitted_y=m['c']+m['rho']*(target.shift()-shift.shift())+shift+m['gamma']*(old.X[feature]-m['x_mean'])
            real_fit=fitted_y if kind=='raw' else 100*np.exp(fitted_y/100)
            observed=core.REAL.loc[old.COMMON];pred=real_fit.loc[old.COMMON]
            residual=pred-observed
            clean=target.loc[old.COMMON]-shift.loc[old.COMMON]
            clean_fit=fitted_y.loc[old.COMMON]-shift.loc[old.COMMON]
            fits.append(dict(model=name,window_months=window,rmse_index_points=float(np.sqrt(np.mean(residual**2))),mae_index_points=float(np.mean(abs(residual))),
                r2_level=float(1-np.sum(residual**2)/np.sum((observed-observed.mean())**2)),r2_clean_level=float(1-np.sum((clean_fit-clean)**2)/np.sum((clean-clean.mean())**2)),
                rho=m['rho'],gamma=m['gamma']))
            nominal=old.P.iloc[0]*real_fit/100*old.C.reindex(old.P.index)/old.C.loc[old.P.index[0]]
            monthly_fit=100*np.log(nominal/old.P.shift())
            for year in [2023,2024,2025]:
                annual.append(dict(model=name,year=year,actual_growth_pct=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1)),
                    fitted_accumulated_growth_pct=float(100*np.expm1(monthly_fit.loc[str(year)].sum()/100))))
            for n in range(30,len(old.COMMON)):
                train=old.COMMON[:n];origin=train[-1];dates=old.COMMON[n:n+12]
                fm=core.fit_level(train,feature,kind)
                forecast=core.predict_level(fm,origin,dates)
                for h in [1,6,12]:
                    if len(dates)<h:continue
                    folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(forecast.iloc[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin]),
                        future_event_months=int(old.F.loc[dates[:h]].any(axis=1).sum()),gamma=fm['gamma'],rho=fm['rho']))
            for start in [old.COMMON[0],pd.Timestamp('2023-01-01')]:
                sample=old.COMMON[old.COMMON>=start];sm=core.fit_level(sample,feature,kind)
                sensitivity.append(dict(model=name,start=str(start.date()),n=len(sample),gamma=sm['gamma'],rho=sm['rho']))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x)),bias='mean').reset_index()
    calm=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x))).reset_index()
    previous=pd.read_csv(ROOT/'real_index_results/backtest_metrics.csv')
    for name,ref in [('raw_avg6','real_index_market'),('log_avg6','log_index_market')]:
        assert np.allclose(metrics[metrics.model==name].sort_values('horizon').rmse,previous[previous.model==ref].sort_values('horizon').rmse)
    assert old.X.loc[old.COMMON,['mos_lag1','mos_avg3_lag1']].notna().all().all()
    pd.DataFrame(fits).to_csv(OUT/'historical_fit_metrics.csv',index=False)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False);f.to_csv(OUT/'backtest_folds.csv',index=False)
    calm.to_csv(OUT/'non_event_monthly_metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'historical_annual_comparison.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'sample_sensitivity.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(models,indent=2))
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(old.COMMON),start=str(old.COMMON[0].date()),end=str(old.COMMON[-1].date()),
        change='Trailing average monthly sales over 3 rather than 6 completed months. Same one-month lag in stock, same units, events and matching sample.',
        validation='24/19/13 expanding-window forecasts; actual future CPI, first available market indicator held fixed, no future events supplied; overlapping windows and retrospective event calendar.',
        quarterly_meaning='Rolling three-month average, not fixed calendar quarters or three-month sum.',production_model_changed=False),indent=2))
    print('FIT');print(pd.DataFrame(fits).round(5).to_string(index=False))
    print('BACKTEST');print(metrics.round(4).to_string(index=False))
    print('CALM');print(calm.round(4).to_string(index=False))
    print('ANNUAL');print(pd.DataFrame(annual).round(4).to_string(index=False))
    print('SENSITIVITY');print(pd.DataFrame(sensitivity).round(5).to_string(index=False))


if __name__=='__main__':main()
