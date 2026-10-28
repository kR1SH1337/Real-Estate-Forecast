"""Monthly real price growth on sales and inventory separately.
Contemporaneous sales are observed explanatory data, not known future inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import model as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'direct_sales_growth_results'
X=pd.DataFrame({'sales_now':np.log(old.Q),'sales_lag1':np.log(old.Q.shift(1)),
                'stock_lag1':np.log(old.S.shift(1)),'ratio6':old.X.mos_lag1})
SAMPLE=pd.concat([old.R.rename('r'),old.R.shift().rename('lag'),X],axis=1).dropna().loc['2022-01-01':].index
ALL_EVENTS=['early','recent','data']
CONFIGS={
    'ar_all_events':dict(events=ALL_EVENTS,controls=[]),
    'ratio6_all_events':dict(events=ALL_EVENTS,controls=['ratio6']),
    'sales_stock_all_events':dict(events=ALL_EVENTS,controls=['sales_now','stock_lag1']),
    'sales_stock_data_only':dict(events=['data'],controls=['sales_now','stock_lag1']),
    'sales_only_data_only':dict(events=['data'],controls=['sales_now']),
    'lagged_sales_stock_data_only':dict(events=['data'],controls=['sales_lag1','stock_lag1'])}


def fit(train,config):
    events=[k for k in config['events'] if old.F.loc[train,k].nunique()>1]
    cols=config['controls'];means=X.loc[train,cols].mean()
    xx=(X.loc[train,cols]-means).to_numpy()
    y=old.R.loc[train].to_numpy();lag=old.R.shift().loc[train].to_numpy()
    f=old.F.loc[train,events].to_numpy();fl=old.F.shift().loc[train,events].to_numpy();k=len(events)
    def residual(p):return y-f@p[2:2+k]-p[0]-p[1]*(lag-fl@p[2:2+k])-xx@p[2+k:]
    opts=[]
    for phi in [.1,.6]:
        start=np.r_[.3,phi,np.ones(k),np.zeros(len(cols))]
        lo=np.full(len(start),-np.inf);hi=np.full(len(start),np.inf);lo[1]=-.95;hi[1]=.95
        opt=least_squares(residual,start,bounds=(lo,hi),max_nfev=2000)
        if opt.success:opts.append(opt)
    opt=min(opts,key=lambda x:x.fun@x.fun)
    return dict(c=float(opt.x[0]),phi=float(opt.x[1]),beta=dict(zip(events,map(float,opt.x[2:2+k]))),
                coefficients=dict(zip(cols,map(float,opt.x[2+k:]))),means=means.to_dict(),n=len(train),rmse_log_pp=float(np.sqrt(np.mean(opt.fun**2))))


def predict(m,origin,dates,realized_sales=False):
    state=float(old.R.loc[origin]-sum(v*old.F.loc[origin,k] for k,v in m['beta'].items()))
    level=float(old.P.loc[origin]);out=[]
    stock=float(old.S.loc[origin]);qbar=float(old.Q.loc[:origin].tail(6).mean())
    assert np.isfinite(qbar) and qbar>0
    for h,date in enumerate(dates):
        control=0.
        for col,coef in m['coefficients'].items():
            if col=='sales_now':value=float(X.loc[date,col]) if realized_sales else np.log(qbar)
            elif col=='sales_lag1':value=float(X.loc[date,col]) if realized_sales or h==0 else np.log(qbar)
            elif col=='stock_lag1':value=np.log(stock)
            elif col=='ratio6':value=np.log(stock/qbar)
            else:raise ValueError(col)
            control+=coef*(value-m['means'][col])
        state=m['c']+m['phi']*state+control
        level*=np.exp(state/100)*old.C.loc[date]/old.C.loc[date-pd.offsets.MonthBegin(1)]
        assert np.isfinite(level) and level>0
        out.append(level)
    return pd.Series(out,index=dates)


def main():
    OUT.mkdir(exist_ok=True)
    parameters={};summary=[];annual=[];hist=[];folds=[];sensitivity=[]
    for name,config in CONFIGS.items():
        m=fit(SAMPLE,config);parameters[name]=m
        effect=old.F.mul(pd.Series(m['beta'])).sum(axis=1)
        predicted=m['c']+m['phi']*(old.R.shift()-effect.shift())+effect
        for col,coef in m['coefficients'].items():predicted+=coef*(X[col]-m['means'][col])
        residual=predicted.loc[SAMPLE]-old.R.loc[SAMPLE]
        clean=old.R.loc[SAMPLE]-effect.loc[SAMPLE]
        summary.append(dict(model=name,n=len(SAMPLE),rmse_log_pp=float(np.sqrt(np.mean(residual**2))),mae_log_pp=float(np.mean(abs(residual))),
            r2_actual_growth=float(1-np.sum(residual**2)/np.sum((old.R.loc[SAMPLE]-old.R.loc[SAMPLE].mean())**2)),
            r2_clean_growth=float(1-np.sum(residual**2)/np.sum((clean-clean.mean())**2)),phi=m['phi'],
            sales=m['coefficients'].get('sales_now',m['coefficients'].get('sales_lag1')),stock=m['coefficients'].get('stock_lag1'),ratio=m['coefficients'].get('ratio6'),
            early_beta=m['beta'].get('early'),recent_beta=m['beta'].get('recent'),data_beta=m['beta'].get('data')))
        inflation=100*np.log(old.C/old.C.shift()).reindex(old.P.index)
        for year in range(2022,2026):
            real=predicted.loc[str(year)];assert len(real)==12 and real.notna().all()
            annual.append(dict(model=name,year=year,actual_nominal_growth_pct=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1)),
                fitted_nominal_growth_pct=float(100*np.expm1((real+inflation.loc[real.index]).sum()/100))))
        for date in SAMPLE:hist.append(dict(model=name,month=str(date.date()),actual_real_log_pp=float(old.R.loc[date]),fitted_real_log_pp=float(predicted.loc[date]),sales_m2=float(old.Q.loc[date]),stock_lag1_m2=float(old.S.shift().loc[date])))
        for n in range(30,len(SAMPLE)):
            train=SAMPLE[:n];origin=train[-1];dates=SAMPLE[n:n+12];fm=fit(train,config)
            forecast=predict(fm,origin,dates);diagnostic=predict(fm,origin,dates,True)
            for h in [1,6,12]:
                if len(dates)<h:continue
                actual=old.P.loc[dates[h-1]]
                folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(forecast.iloc[h-1]-actual)/old.P.loc[origin]),realized_sales_diagnostic_error_pp=float(100*(diagnostic.iloc[h-1]-actual)/old.P.loc[origin]),future_event_months=int(old.F.loc[dates[:h]].any(axis=1).sum()),sales_coefficient=fm['coefficients'].get('sales_now',fm['coefficients'].get('sales_lag1')),stock_coefficient=fm['coefficients'].get('stock_lag1')))
        for start in ['2022-01-01','2023-01-01']:
            sm=fit(SAMPLE[SAMPLE>=start],config)
            sensitivity.append(dict(model=name,start=start,n=sm['n'],c=sm['c'],phi=sm['phi'],sales=sm['coefficients'].get('sales_now',sm['coefficients'].get('sales_lag1')),stock=sm['coefficients'].get('stock_lag1'),ratio=sm['coefficients'].get('ratio6')))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x)),bias='mean').reset_index()
    diagnostic=f.groupby(['model','horizon']).realized_sales_diagnostic_error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x))).reset_index()
    calm=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x))).reset_index()
    # Reproduce the same-sample original one-regressor fits.
    for name,feature in [('ar_all_events',None),('ratio6_all_events','mos_lag1')]:
        ref=old.fit(SAMPLE,feature);m=parameters[name]
        assert np.allclose([m['c'],m['phi'],m['rmse_log_pp']],[ref['c'],ref['phi'],ref['rmse']])
    train=SAMPLE[:30];origin=train[-1];dates=SAMPLE[30:42];m=fit(train,CONFIGS['sales_stock_data_only'])
    before=predict(m,origin,dates);saved=X.copy();savedq=old.Q.copy();saveds=old.S.copy()
    X.loc[X.index>origin,'sales_now']=999.;old.Q.loc[old.Q.index>origin]=1e12;old.S.loc[old.S.index>origin]=1e12
    after=predict(m,origin,dates);X[:]=saved;old.Q[:]=savedq;old.S[:]=saveds
    assert np.allclose(before,after) and len(SAMPLE)==57
    assert np.isfinite(metrics[['rmse','mae']]).all().all()
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    pd.DataFrame(summary).to_csv(OUT/'historical_fit_metrics.csv',index=False)
    pd.DataFrame(hist).to_csv(OUT/'historical_fit.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'historical_annual_comparison.csv',index=False)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False);f.to_csv(OUT/'backtest_folds.csv',index=False)
    diagnostic.to_csv(OUT/'realized_sales_diagnostic_metrics.csv',index=False)
    calm.to_csv(OUT/'non_event_monthly_metrics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'sample_sensitivity.csv',index=False)
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(SAMPLE),start=str(SAMPLE[0].date()),end=str(SAMPLE[-1].date()),
        target='100*delta log(P/CPI), monthly real growth, not real-index level',
        specification='Centered log contemporaneous sales and centered log prior-month opening unsold stock, independent unrestricted coefficients; latent AR(1) plus optional event dummies.',
        forecast='Future sales held at mean of six months known at origin. Stock held at origin opening stock. Lagged sales known at first step, then held at origin mean. Actual future CPI, no future events.',
        realized_sales_diagnostic='Future actual sales supplied only in diagnostic, never in standard score. Stock remains held at origin.',
        limits='Current sales and prices jointly determined; no causal identification. No extra seasonality. Short revised sample, retrospective events, overlapping horizons, all annual windows include events.',
        events='data_only removes policy groups but retains February/April 2025 data-anomaly group; group remains a hypothesis.',production_model_changed=False),indent=2))
    print('FIT');print(pd.DataFrame(summary).round(4).to_string(index=False))
    print('ANNUAL');print(pd.DataFrame(annual).pivot(index='model',columns='year',values='fitted_nominal_growth_pct').round(3).to_string())
    print('FORECAST');print(metrics.round(4).to_string(index=False))
    print('REALIZED SALES');print(diagnostic.round(4).to_string(index=False))
    print('SENSITIVITY');print(pd.DataFrame(sensitivity).round(4).to_string(index=False))


if __name__=='__main__':main()
