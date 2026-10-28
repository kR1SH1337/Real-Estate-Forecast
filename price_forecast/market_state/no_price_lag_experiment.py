"""Remove only lagged real price growth; keep sales, stock and event definitions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import direct_sales_growth_experiment as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'no_price_lag_results'
old=core.old
CONFIGS={'all_events_ar':('sales_stock_all_events',True),'all_events_noar':('sales_stock_all_events',False),
         'data_only_ar':('sales_stock_data_only',True),'data_only_noar':('sales_stock_data_only',False)}


def fit_noar(train,config):
    events=[k for k in config['events'] if old.F.loc[train,k].nunique()>1]
    cols=config['controls'];means=core.X.loc[train,cols].mean()
    design=np.c_[np.ones(len(train)),old.F.loc[train,events].to_numpy(),(core.X.loc[train,cols]-means).to_numpy()]
    y=old.R.loc[train].to_numpy();coef,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
    assert rank==design.shape[1]
    k=len(events)
    return dict(c=float(coef[0]),phi=0.,beta=dict(zip(events,map(float,coef[1:1+k]))),coefficients=dict(zip(cols,map(float,coef[1+k:]))),
                means=means.to_dict(),n=len(train),rmse_log_pp=float(np.sqrt(np.mean((y-design@coef)**2))))


def main():
    OUT.mkdir(exist_ok=True)
    parameters={};summary=[];annual=[];history=[];folds=[];sensitivity=[]
    for name,(reference,lag) in CONFIGS.items():
        config=core.CONFIGS[reference]
        fitfunc=core.fit if lag else fit_noar
        m=fitfunc(core.SAMPLE,config);parameters[name]=m
        ef=old.F.mul(pd.Series(m['beta'])).sum(axis=1)
        pred=m['c']+m['phi']*(old.R.shift()-ef.shift())+ef
        for col,coef in m['coefficients'].items():pred+=coef*(core.X[col]-m['means'][col])
        residual=pred.loc[core.SAMPLE]-old.R.loc[core.SAMPLE]
        nullconfig=dict(events=config['events'],controls=[])
        restricted=fitfunc(core.SAMPLE,nullconfig)
        sse=float(np.sum(residual**2));sse_null=restricted['n']*restricted['rmse_log_pp']**2
        tss=float(np.sum((old.R.loc[core.SAMPLE]-old.R.loc[core.SAMPLE].mean())**2))
        summary.append(dict(model=name,n=m['n'],c=m['c'],phi=m['phi'],sales=m['coefficients']['sales_now'],stock=m['coefficients']['stock_lag1'],rmse_log_pp=float(np.sqrt(np.mean(residual**2))),r2_growth=1-sse/tss,market_incremental_r2=(sse_null-sse)/tss,market_partial_r2=1-sse/sse_null,restricted_rmse=restricted['rmse_log_pp']))
        for year in range(2022,2026):
            inflation=100*np.log(old.C/old.C.shift()).loc[str(year)].reindex(pred.loc[str(year)].index)
            annual.append(dict(model=name,year=year,actual_growth_pct=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1)),fitted_growth_pct=float(100*np.expm1((pred.loc[str(year)]+inflation).sum()/100))))
        for date in core.SAMPLE:history.append(dict(model=name,month=str(date.date()),actual_real_log_pp=float(old.R.loc[date]),fitted_real_log_pp=float(pred.loc[date])))
        for n in range(30,len(core.SAMPLE)):
            train=core.SAMPLE[:n];origin=train[-1];dates=core.SAMPLE[n:n+12]
            fm=fitfunc(train,config);forecast=core.predict(fm,origin,dates);oracle=core.predict(fm,origin,dates,True)
            for h in [1,6,12]:
                if len(dates)<h:continue
                actual=old.P.loc[dates[h-1]]
                folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(forecast.iloc[h-1]-actual)/old.P.loc[origin]),realized_sales_diagnostic_error_pp=float(100*(oracle.iloc[h-1]-actual)/old.P.loc[origin]),sales=fm['coefficients']['sales_now'],stock=fm['coefficients']['stock_lag1'],future_event_months=int(old.F.loc[dates[:h]].any(axis=1).sum())))
        for start in ['2022-01-01','2023-01-01']:
            sm=fitfunc(core.SAMPLE[core.SAMPLE>=start],config)
            sensitivity.append(dict(model=name,start=start,n=sm['n'],c=sm['c'],phi=sm['phi'],sales=sm['coefficients']['sales_now'],stock=sm['coefficients']['stock_lag1']))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v)),mae=lambda v:np.mean(abs(v)),bias='mean').reset_index()
    oracle=f.groupby(['model','horizon']).realized_sales_diagnostic_error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
    calm=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
    previous=pd.read_csv(core.OUT/'backtest_metrics.csv')
    for name,reference in [('all_events_ar','sales_stock_all_events'),('data_only_ar','sales_stock_data_only')]:
        assert np.allclose(metrics[metrics.model==name].sort_values('horizon').rmse,previous[previous.model==reference].sort_values('horizon').rmse)
    # Removing the AR lag cannot improve the nested full-sample SSE.
    s=pd.DataFrame(summary).set_index('model')
    for prefix in ['all_events','data_only']:
        assert s.loc[prefix+'_noar','rmse_log_pp']>=s.loc[prefix+'_ar','rmse_log_pp']-1e-8
    assert np.isfinite(metrics[['rmse','mae']]).all().all()

    origin=core.SAMPLE[29];dates=core.SAMPLE[30:42]
    noar=fit_noar(core.SAMPLE[:30],core.CONFIGS['sales_stock_data_only'])
    before=core.predict(noar,origin,dates);saved_growth=float(old.R.loc[origin]);old.R.loc[origin]=saved_growth+10.
    after=core.predict(noar,origin,dates);old.R.loc[origin]=saved_growth
    assert np.allclose(before,after)

    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    pd.DataFrame(summary).to_csv(OUT/'historical_fit_metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'historical_annual_comparison.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'historical_fit.csv',index=False)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False);f.to_csv(OUT/'backtest_folds.csv',index=False)
    oracle.to_csv(OUT/'realized_sales_diagnostic_metrics.csv',index=False)
    calm.to_csv(OUT/'non_event_monthly_metrics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'sample_sensitivity.csv',index=False)
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(core.SAMPLE),change='Only lagged real price growth removed (phi=0); lagged inventory retained.',noar_fit='Full-rank ordinary least squares; AR references retain original nonlinear latent-AR estimation.',market_partial_r2='1-SSE(full sales+stock model)/SSE(same event and AR specification without sales and stock); descriptive incremental fit, not causal attribution.',forecast='Same as direct-sales experiment: unknown future sales held at six-month mean known at origin; stock held at origin; actual future CPI; future policy events not supplied.',limits='Current sales endogenous; short revised history and retrospective events; overlapping horizons, all annual windows include events.',production_model_changed=False),indent=2))
    print('FIT');print(pd.DataFrame(summary).round(5).to_string(index=False))
    print('ANNUAL');print(pd.DataFrame(annual).pivot(index='model',columns='year',values='fitted_growth_pct').round(3).to_string())
    print('FORECAST');print(metrics.round(4).to_string(index=False))
    print('SENSITIVITY');print(pd.DataFrame(sensitivity).round(4).to_string(index=False))


if __name__=='__main__':main()
