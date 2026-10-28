"""Can monthly sales pulses replace policy-event indicators in real-index levels?
Contemporaneous realized sales are diagnostic inputs, never a usable forecast.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import real_index_experiment as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sales_pulse_results'
old=core.old
PULSE=np.log(old.Q/old.Q.rolling(6).mean().shift(1))
SHIFTS=core.SHIFTS.copy()
SHIFTS['sales_pulse']=PULSE.reindex(old.P.index).fillna(0.).cumsum()
X=pd.DataFrame({'market':old.X.mos_lag1,'pulse_now':PULSE,'pulse_lag1':PULSE.shift(1)})
SAMPLE=pd.concat([core.REAL.rename('target'),core.REAL.shift().rename('lag'),X],axis=1).dropna().loc['2022-01-01':].index
CONFIGS={
    'baseline':dict(events=['early','recent','data'],controls=['market']),
    'pulse_now_all_events':dict(events=['early','recent','data'],controls=['market','pulse_now']),
    'pulse_now_data_only':dict(events=['data'],controls=['market','pulse_now']),
    'pulse_lag_all_events':dict(events=['early','recent','data'],controls=['market','pulse_lag1']),
    'pulse_lag_data_only':dict(events=['data'],controls=['market','pulse_lag1']),
    'pulse_now_no_market':dict(events=['data'],controls=['pulse_now']),
    'cumulative_pulse_all_events':dict(events=['early','recent','data','sales_pulse'],controls=['market']),
    'cumulative_pulse_data_only':dict(events=['data','sales_pulse'],controls=['market']),
    'no_events_no_pulse':dict(events=[],controls=['market']),
    'pulse_now_no_events':dict(events=[],controls=['market','pulse_now']),
    'cumulative_pulse_no_events':dict(events=['sales_pulse'],controls=['market'])}


def fit(train,config):
    groups=[k for k in config['events'] if SHIFTS.loc[train,k].nunique()>1]
    cols=config['controls'];means=X.loc[train,cols].mean()
    controls=(X.loc[train,cols]-means).to_numpy()
    y=core.REAL.loc[train].to_numpy();lag=core.REAL.shift().loc[train].to_numpy()
    d=SHIFTS.loc[train,groups].to_numpy();dl=SHIFTS.shift().loc[train,groups].to_numpy();k=len(groups)
    def residual(p):return y-d@p[2:2+k]-p[0]-p[1]*(lag-dl@p[2:2+k])-controls@p[2+k:]
    fits=[]
    for rho in [.5,.9,1.,1.05]:
        start=np.r_[np.mean(y-rho*lag),rho,np.ones(k),np.zeros(len(cols))]
        opt=least_squares(residual,start,max_nfev=4000,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if opt.success:fits.append(opt)
    opt=min(fits,key=lambda o:o.fun@o.fun)
    return dict(c=float(opt.x[0]),rho=float(opt.x[1]),beta=dict(zip(groups,map(float,opt.x[2:2+k]))),
                coefficients=dict(zip(cols,map(float,opt.x[2+k:]))),means=means.to_dict(),n=len(train),rmse_index=float(np.sqrt(np.mean(opt.fun**2))))


def predict(m,origin,dates,realized_pulse=False):
    shift=sum(v*SHIFTS.loc[origin,k] for k,v in m['beta'].items())
    state=float(core.REAL.loc[origin]-shift)
    out=[]
    for h,date in enumerate(dates):
        control=0.
        for col,coef in m['coefficients'].items():
            if col=='market':x=float(X.loc[dates[0],col])
            elif realized_pulse:x=float(X.loc[date,col])
            elif col=='pulse_lag1' and h==0:x=float(X.loc[date,col])
            else:x=0.  # No unanticipated future pulse is assumed.
            control+=coef*(x-m['means'][col])
        state=m['c']+m['rho']*state+control
        adjusted_shift=shift
        if realized_pulse and 'sales_pulse' in m['beta']:
            adjusted_shift+=m['beta']['sales_pulse']*(SHIFTS.loc[date,'sales_pulse']-SHIFTS.loc[origin,'sales_pulse'])
        real=state+adjusted_shift
        assert np.isfinite(real) and real>0
        out.append(old.P.iloc[0]*real/100*old.C.loc[date]/old.C.loc[old.P.index[0]])
    return pd.Series(out,index=dates)


def main():
    OUT.mkdir(exist_ok=True)
    pars={};history=[];annual=[];summary=[];folds=[];sens=[]
    for name,config in CONFIGS.items():
        m=fit(SAMPLE,config);pars[name]=m
        shift=SHIFTS.mul(pd.Series(m['beta'],dtype=float)).sum(axis=1)
        pred=m['c']+m['rho']*(core.REAL.shift()-shift.shift())+shift
        for col,coef in m['coefficients'].items():pred+=coef*(X[col]-m['means'][col])
        err=pred.loc[SAMPLE]-core.REAL.loc[SAMPLE]
        summary.append(dict(model=name,rmse_index=float(np.sqrt(np.mean(err**2))),r2_level=float(1-np.sum(err**2)/np.sum((core.REAL.loc[SAMPLE]-core.REAL.loc[SAMPLE].mean())**2)),rho=m['rho'],market=m['coefficients'].get('market'),pulse=m['coefficients'].get('pulse_now',m['coefficients'].get('pulse_lag1',m['beta'].get('sales_pulse'))),early_beta=m['beta'].get('early'),recent_beta=m['beta'].get('recent'),data_beta=m['beta'].get('data')))
        nominal=old.P.iloc[0]*pred/100*old.C.reindex(pred.index)/old.C.loc[old.P.index[0]]
        changes=100*np.log(nominal/old.P.shift())
        for year in range(2022,2026):
            annual.append(dict(model=name,year=year,actual_growth_pct=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1)),fitted_growth_pct=float(100*np.expm1(changes.loc[str(year)].sum()/100))))
        for date in SAMPLE:history.append(dict(model=name,month=str(date.date()),actual_real_index=float(core.REAL.loc[date]),fitted_real_index=float(pred.loc[date]),pulse=float(PULSE.loc[date])))
        for n in range(30,len(SAMPLE)):
            train=SAMPLE[:n];origin=train[-1];dates=SAMPLE[n:n+12];fm=fit(train,config)
            forecast=predict(fm,origin,dates);oracle=predict(fm,origin,dates,True)
            for h in [1,6,12]:
                if len(dates)<h:continue
                actual=old.P.loc[dates[h-1]]
                folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(forecast.iloc[h-1]-actual)/old.P.loc[origin]),realized_sales_diagnostic_error_pp=float(100*(oracle.iloc[h-1]-actual)/old.P.loc[origin]),future_event_months=int(old.F.loc[dates[:h]].any(axis=1).sum()),pulse_coefficient=fm['coefficients'].get('pulse_now',fm['coefficients'].get('pulse_lag1',fm['beta'].get('sales_pulse')))))
        for start in ['2022-01-01','2023-01-01']:
            sm=fit(SAMPLE[SAMPLE>=start],config)
            sens.append(dict(model=name,start=start,n=sm['n'],rho=sm['rho'],market=sm['coefficients'].get('market'),pulse=sm['coefficients'].get('pulse_now',sm['coefficients'].get('pulse_lag1',sm['beta'].get('sales_pulse')))))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v)),mae=lambda v:np.mean(abs(v)),bias='mean').reset_index()
    oracle_metrics=f.groupby(['model','horizon']).realized_sales_diagnostic_error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
    calm=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
    ref=pd.read_csv(ROOT/'sales_window_from2022_results/backtest_metrics.csv')
    assert np.allclose(metrics[metrics.model=='baseline'].sort_values('horizon').rmse,ref[ref.model=='raw_avg6'].sort_values('horizon').rmse)
    assert len(SAMPLE)==57 and SAMPLE[0]==pd.Timestamp('2022-01-01')
    assert np.isfinite(metrics[['rmse','mae']]).all().all()
    train=SAMPLE[:30];origin=train[-1];dates=SAMPLE[30:42];m=fit(train,CONFIGS['pulse_now_data_only']);before=predict(m,origin,dates)
    saved=X.copy();X.loc[X.index>origin,'pulse_now']=999.;X.loc[X.index>origin+pd.offsets.MonthBegin(1),'pulse_lag1']=999.
    after=predict(m,origin,dates);X[:]=saved
    assert np.allclose(before,after)
    cm=fit(train,CONFIGS['cumulative_pulse_data_only']);cb=predict(cm,origin,dates)
    saved_cumulative=SHIFTS.sales_pulse.copy();SHIFTS.loc[core.SHIFTS.index>origin,'sales_pulse']=999.
    ca=predict(cm,origin,dates);SHIFTS['sales_pulse']=saved_cumulative
    assert np.allclose(cb,ca)
    (OUT/'parameters.json').write_text(json.dumps(pars,indent=2))
    pd.DataFrame(summary).to_csv(OUT/'historical_fit_metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'historical_annual_comparison.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'historical_fit.csv',index=False)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False);oracle_metrics.to_csv(OUT/'realized_sales_diagnostic_metrics.csv',index=False)
    calm.to_csv(OUT/'non_event_monthly_metrics.csv',index=False);f.to_csv(OUT/'backtest_folds.csv',index=False)
    pd.DataFrame(sens).to_csv(OUT/'sample_sensitivity.csv',index=False)
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(SAMPLE),pulse='log(current month sales / average of preceding six completed months)',contemporaneous_endogeneity='Current sales and price can be jointly determined; coefficient is not a causal effect.',forecast='Market held at first lagged value; expected future pulse zero. Lagged-pulse model uses known prior-month pulse at first step. Actual future CPI. No future event increments.',realized_sales='Diagnostic only: future observed sales pulse supplied; not an operational forecast.',events='data_only/no_market retain data-anomaly events; no_events variants remove all three dummy groups. Cumulative pulse uses the same persistent-level-shift treatment as cumulative event dummies.',validation='Matching expanding windows; revised data and retrospective event calendar; overlapping annual horizons, no clean annual event-free windows.',production_model_changed=False),indent=2))
    print('FIT');print(pd.DataFrame(summary).round(4).to_string(index=False))
    print('ANNUAL');print(pd.DataFrame(annual).pivot(index='model',columns='year',values='fitted_growth_pct').round(3).to_string())
    print('FORECAST');print(metrics.round(4).to_string(index=False))
    print('REALIZED SALES DIAGNOSTIC');print(oracle_metrics[oracle_metrics.horizon==1].round(4).to_string(index=False))
    print('SENSITIVITY');print(pd.DataFrame(sens).round(4).to_string(index=False))


if __name__=='__main__':main()
