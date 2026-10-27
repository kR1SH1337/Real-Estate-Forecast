"""Nominal-index level forecasts without CPI, on matching historical folds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import model as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'nominal_index_results'
TARGETS={'log':100*np.log(old.P/old.P.iloc[0]),'raw':old.P.copy()}
SHIFTS=old.F.cumsum()


def fit(train,feature=None,transform='log'):
    target=TARGETS[transform]
    active=[k for k in SHIFTS if SHIFTS.loc[train,k].nunique()>1]
    y=target.loc[train].to_numpy();lag=target.shift().loc[train].to_numpy()
    d=SHIFTS.loc[train,active].to_numpy();dl=SHIFTS.shift().loc[train,active].to_numpy()
    mean=float(old.X.loc[train,feature].mean()) if feature else 0.
    x=old.X.loc[train,feature].to_numpy()-mean if feature else np.zeros(len(train))
    k=len(active)
    def residual(p):
        return y-d@p[2:2+k]-p[0]-p[1]*(lag-dl@p[2:2+k])-(p[-1]*x if feature else 0.)
    fits=[]
    for rho in [.5,.9,1.,1.05]:
        start=np.r_[np.mean(y-rho*lag),rho,np.ones(k),np.zeros(1 if feature else 0)]
        opt=least_squares(residual,start,max_nfev=4000,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if opt.success:fits.append(opt)
    if not fits:raise RuntimeError('No converged fit')
    opt=min(fits,key=lambda f:f.fun@f.fun)
    return dict(c=float(opt.x[0]),rho=float(opt.x[1]),beta=dict(zip(active,map(float,opt.x[2:2+k]))),
                gamma=float(opt.x[-1]) if feature else 0.,x_mean=mean,feature=feature,
                transform=transform,n=len(train),rmse_target=float(np.sqrt(np.mean(opt.fun**2))))


def predict(m,origin,dates):
    shift=sum(v*SHIFTS.loc[origin,k] for k,v in m['beta'].items())
    state=float(TARGETS[m['transform']].loc[origin]-shift)
    x0=(float(old.X.loc[dates[0],m['feature']]) if dates[0] in old.X.index else old.live_x(m['feature'])) if m['feature'] else 0.
    rows=[]
    for date in dates:
        state=m['c']+m['rho']*state+m['gamma']*(x0-m['x_mean'])
        y=state+shift
        nominal=old.P.iloc[0]*np.exp(y/100) if m['transform']=='log' else y
        if not np.isfinite(nominal) or nominal<=0:raise ValueError('Nonpositive/nonfinite predicted nominal index')
        rows.append(nominal)
    return pd.Series(rows,index=dates)


def main():
    OUT.mkdir(exist_ok=True)
    configs=[('nominal_log_no_market','log',None),('nominal_log_market','log','mos_lag1'),
             ('nominal_index_no_market','raw',None),('nominal_index_market','raw','mos_lag1')]
    models={};folds=[];history=[];rolling=[];annual=[];sensitivity=[]
    for name,kind,feature in configs:
        m=fit(old.COMMON,feature,kind);models[name]=m
        for n in range(30,len(old.COMMON)):
            train=old.COMMON[:n];origin=train[-1];dates=old.COMMON[n:n+12]
            fm=fit(train,feature,kind);pred=predict(fm,origin,dates)
            rolling.append(dict(model=name,origin=str(origin.date()),**{k:v for k,v in fm.items() if k!='beta'}))
            for h in [1,6,12]:
                if len(dates)>=h:folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(pred.iloc[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin])))
        target=TARGETS[kind]
        shift=sum(v*SHIFTS[k] for k,v in m['beta'].items())
        fitted_y=m['c']+m['rho']*(target.shift()-shift.shift())+shift
        if feature:fitted_y+=m['gamma']*(old.X[feature]-m['x_mean'])
        fitted=old.P.iloc[0]*np.exp(fitted_y/100) if kind=='log' else fitted_y
        for date in old.COMMON:
            history.append(dict(model=name,month=str(date.date()),actual_nominal_index=float(old.P.loc[date]),fitted_nominal_index=float(fitted.loc[date]),
                                actual_nominal_log_growth=float(100*np.log(old.P.loc[date]/old.P.shift().loc[date])),
                                fitted_nominal_log_growth=float(100*np.log(fitted.loc[date]/old.P.shift().loc[date]))))
        dates=pd.date_range('2026-10-01','2030-12-01',freq='MS')
        future=predict(m,old.P.index[-1],dates);full=pd.concat([old.P,future])
        for year in range(2026,2031):
            annual.append(dict(model=name,year=year,nominal_growth_pct=float(100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1)),
                               year_end_index=float(full.loc[f'{year}-12-01']),case='Current market held; not a coupled equilibrium'))
        for start in [old.COMMON[0],pd.Timestamp('2023-01-01')]:
            sm=fit(old.COMMON[old.COMMON>=start],feature,kind)
            sensitivity.append(dict(model=name,start=str(start.date()),n=sm['n'],c=sm['c'],rho=sm['rho'],gamma=sm['gamma']))
    # Simple nominal references use no CPI and no future events.
    nominal_growth=100*np.log(old.P/old.P.shift())
    for n in range(30,len(old.COMMON)):
        train=old.COMMON[:n];origin=train[-1];dates=old.COMMON[n:n+12]
        for name,drift in [('nominal_random_walk',0.),('nominal_drift',float(nominal_growth.loc[train].mean()))]:
            pred=old.P.loc[origin]*np.exp(np.arange(1,len(dates)+1)*drift/100)
            for h in [1,6,12]:
                if len(dates)>=h:folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(pred[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin])))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x))).reset_index()
    reference=pd.read_csv(ROOT/'real_index_results/backtest_metrics.csv')
    comparison=pd.concat([reference,metrics],ignore_index=True)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False);comparison.to_csv(OUT/'comparison_metrics.csv',index=False)
    f.to_csv(OUT/'backtest_folds.csv',index=False);pd.DataFrame(history).to_csv(OUT/'historical_fit.csv',index=False)
    pd.DataFrame(rolling).to_csv(OUT/'rolling_parameters.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'conditional_annual_forecasts.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'sample_sensitivity.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(models,indent=2))
    # Check invariance to information not available at the training origin.
    train=old.COMMON[:30];origin=train[-1];dates=old.COMMON[30:42]
    before=fit(train,'mos_lag1');pred_before=predict(before,origin,dates)
    saved=TARGETS['log'].copy();saved_c=old.C.copy();saved_p=old.P.copy()
    TARGETS['log'].loc[TARGETS['log'].index>origin]=999.
    old.C.loc[old.C.index>origin]=999.;old.P.loc[old.P.index>origin]=999.
    after=fit(train,'mos_lag1');pred_after=predict(after,origin,dates)
    TARGETS['log'][:]=saved;old.C[:]=saved_c;old.P[:]=saved_p
    assert np.allclose([before['c'],before['rho'],before['gamma']],[after['c'],after['rho'],after['gamma']])
    assert np.allclose(pred_before,pred_after)
    assert np.isfinite(metrics[['rmse','mae']]).all().all()
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,n=len(old.COMMON),start=str(old.COMMON[0].date()),end=str(old.COMMON[-1].date()),
        validation='Matching expanding windows; 24/19/13 forecasts at 1/6/12 months. Revised data and retrospective event dates; overlapping horizons.',
        cpi_used_for_nominal_predictions=False,market='First available lagged value held fixed; future events not supplied.',
        reference_information='Real-model benchmarks use actual future CPI and thus have a different information set.',
        shock_semantics='Cumulative indicators retain level shifts; all coefficients re-estimated per fold.',
        persistence='Unconstrained; report rho and sensitivity, not a stationarity claim.',
        conditional_future='Frozen current market, no new shocks; not a four-block equilibrium.',
        production_model_changed=False),indent=2))
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame(sensitivity).round(5).to_string(index=False))
    print(pd.DataFrame(annual).round(3).to_string(index=False))


if __name__=='__main__':main()
