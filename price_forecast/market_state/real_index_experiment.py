"""Compare real-index levels with real monthly growth on matching backtest folds.

An additive shock in log growth becomes a cumulative level shift in log levels.
No future policy shock or future market regressor is supplied to forecasts.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import model as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'real_index_results'
REAL=100*(old.P/old.C.reindex(old.P.index))/(old.P.iloc[0]/old.C.loc[old.P.index[0]])
LOGREAL=100*np.log(REAL/100)
SHIFTS=old.F.cumsum()


def fit_level(train,feature=None,transform='log'):
    target=LOGREAL if transform=='log' else REAL
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
    if not fits:raise RuntimeError('No converged level fit')
    opt=min(fits,key=lambda f:f.fun@f.fun)
    return dict(c=float(opt.x[0]),rho=float(opt.x[1]),beta=dict(zip(active,map(float,opt.x[2:2+k]))),
                gamma=float(opt.x[-1]) if feature else 0.,x_mean=mean,feature=feature,
                transform=transform,n=len(train),rmse_target=float(np.sqrt(np.mean(opt.fun**2))))


def predict_level(m,origin,dates,realized=False):
    target=LOGREAL if m['transform']=='log' else REAL
    shift=sum(v*SHIFTS.loc[origin,k] for k,v in m['beta'].items())
    state=float(target.loc[origin]-shift)
    x0=float(old.X.loc[dates[0],m['feature']]) if m['feature'] else 0.
    rows=[]
    for date in dates:
        x=float(old.X.loc[date,m['feature']]) if realized and m['feature'] else x0
        state=m['c']+m['rho']*state+m['gamma']*(x-m['x_mean'])
        level=state+shift  # no unknown future increments to cumulative shock indicators
        real=100*np.exp(level/100) if m['transform']=='log' else level
        if not np.isfinite(real) or real<=0:raise ValueError('Invalid predicted real index')
        nominal=old.P.iloc[0]*(real/100)*(old.C.loc[date]/old.C.loc[old.P.index[0]])
        rows.append(nominal)
    return pd.Series(rows,index=dates)


def main():
    OUT.mkdir(exist_ok=True)
    configs=[('growth_no_market','growth',None),('growth_market','growth','mos_lag1'),
             ('log_index_no_market','log',None),('log_index_market','log','mos_lag1'),
             ('real_index_no_market','raw',None),('real_index_market','raw','mos_lag1')]
    models={};folds=[];history=[];stability=[]
    for name,kind,feature in configs:
        m=old.fit(old.COMMON,feature) if kind=='growth' else fit_level(old.COMMON,feature,kind)
        models[name]=m
        for n in range(30,len(old.COMMON)):
            train=old.COMMON[:n];origin=train[-1];dates=old.COMMON[n:n+12]
            fm=old.fit(train,feature) if kind=='growth' else fit_level(train,feature,kind)
            pred=old.predict(fm,origin,dates)[0] if kind=='growth' else predict_level(fm,origin,dates)
            stability.append(dict(model=name,origin=str(origin.date()),**fm))
            for h in [1,6,12]:
                if len(dates)<h:continue
                err=100*(pred.iloc[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin]
                folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(err)))
        for date in old.COMMON:
            origin=date-pd.offsets.MonthBegin(1)
            if kind=='growth':
                # In-sample fitted value includes the contemporaneous estimated event effect.
                state=old.R.loc[origin]-old.effect(m,origin)
                rr=m['c']+m['phi']*state+m['gamma']*((old.X.loc[date,feature]-m['x_mean']) if feature else 0.)+old.effect(m,date)
                nominal=old.P.loc[origin]*old.C.loc[date]/old.C.loc[origin]*np.exp(rr/100)
            else:
                target=LOGREAL if kind=='log' else REAL
                prevshift=sum(v*SHIFTS.loc[origin,k] for k,v in m['beta'].items())
                shift=sum(v*SHIFTS.loc[date,k] for k,v in m['beta'].items())
                y=m['c']+m['rho']*(target.loc[origin]-prevshift)+m['gamma']*((old.X.loc[date,feature]-m['x_mean']) if feature else 0.)+shift
                real=100*np.exp(y/100) if kind=='log' else y
                nominal=old.P.iloc[0]*(real/100)*old.C.loc[date]/old.C.loc[old.P.index[0]]
            history.append(dict(model=name,month=str(date.date()),actual_nominal_index=float(old.P.loc[date]),fitted_nominal_index=float(nominal),
                                actual_real_index=float(REAL.loc[date]),fitted_real_index=float(100*nominal/old.P.iloc[0]*old.C.loc[old.P.index[0]]/old.C.loc[date]),
                                actual_real_log_growth=float(old.R.loc[date]),fitted_real_log_growth=float(100*np.log(nominal/old.P.loc[origin]*old.C.loc[origin]/old.C.loc[date]))))
    # Same folds and actual CPI; simple references without policy correction.
    for n in range(30,len(old.COMMON)):
        train=old.COMMON[:n];origin=train[-1];dates=old.COMMON[n:n+12]
        for name,drift in [('real_random_walk',0.),('real_drift',float(old.R.loc[train].mean()))]:
            pred=old.P.loc[origin]*old.C.loc[dates]/old.C.loc[origin]*np.exp(np.arange(1,len(dates)+1)*drift/100)
            for h in [1,6,12]:
                if len(dates)>=h:folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=float(100*(pred.iloc[h-1]-old.P.loc[dates[h-1]])/old.P.loc[origin])))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x))).reset_index()
    f.to_csv(OUT/'backtest_folds.csv',index=False);metrics.to_csv(OUT/'backtest_metrics.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'historical_fit.csv',index=False)
    pd.DataFrame(stability).drop(columns=['beta','season_beta'],errors='ignore').to_csv(OUT/'rolling_parameters.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(models,indent=2))
    sensitivity=[]
    for start in [old.COMMON[0],pd.Timestamp('2023-01-01')]:
        train=old.COMMON[old.COMMON>=start]
        for name,kind,feature in configs:
            m=old.fit(train,feature) if kind=='growth' else fit_level(train,feature,kind)
            sensitivity.append(dict(model=name,start=str(start.date()),n=len(train),c=m['c'],persistence=m.get('rho',m.get('phi')),gamma=m['gamma']))
    pd.DataFrame(sensitivity).to_csv(OUT/'sample_sensitivity.csv',index=False)
    # Matching growth calculations must reproduce the existing saved backtest.
    oldmetrics=pd.read_csv(old.OUT/'backtest_metrics.csv')
    for name,reference in [('growth_no_market','baseline'),('growth_market','mos_lag1')]:
        assert np.allclose(metrics[metrics.model==name].sort_values('horizon').rmse,oldmetrics[oldmetrics.model==reference].sort_values('horizon').rmse)
    # Data after an origin must not affect that origin's level parameters.
    train=old.COMMON[:30];before=fit_level(train,'mos_lag1');saved=LOGREAL.copy()
    LOGREAL.loc[LOGREAL.index>train[-1]]=999.
    after=fit_level(train,'mos_lag1');LOGREAL[:]=saved
    assert np.allclose([before['c'],before['rho'],before['gamma']],[after['c'],after['rho'],after['gamma']])
    assert np.isfinite(metrics[['rmse','mae']]).all().all()
    diagnostics=dict(start=str(old.COMMON[0].date()),end=str(old.COMMON[-1].date()),n=54,checks_passed=True,
        real_index='100*(P/CPI)/(P0/CPI0); not subtraction of two index levels',
        shock_semantics='Cumulative event indicators in level equations preserve permanent level effect of monthly growth events; coefficients re-estimated within every training fold.',
        persistence='Unconstrained level persistence; a value near or above one is reported, not forced to imply mean reversion.',
        validation='Same expanding windows and forecast origins as growth model. Actual future CPI; first available lagged market value held constant; no future events supplied. Revised history and retrospectively chosen event calendar. Overlapping horizons.',
        scope='Historical experiment only; coupled forecasting system and original historical parameters untouched.')
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame(sensitivity).round(5).to_string(index=False))


if __name__=='__main__':main()
