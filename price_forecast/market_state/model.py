"""Lagged market state in the real-growth model with three shock dummies.
Conditional scenarios, not a jointly solved sales/stock/price equilibrium.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'shock_normalization'))
import full_history as old
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
P,C,R=old.P,old.C,old.R
Q=pd.read_csv(ROOT/'inputs/sales.csv',index_col=0,parse_dates=True).sales_pd_thousand_m2*1000
S=pd.read_csv(ROOT/'inputs/stock_snapshots.csv',index_col=0,parse_dates=True).unsold_m2.reindex(P.index)
F=pd.DataFrame({'early':P.index.isin(old.EARLY).astype(float),
                'recent':P.index.isin(old.WAVE+old.LATEST).astype(float),
                'data':P.index.isin(old.DATA_ANOMALIES).astype(float)},index=P.index)
X=pd.DataFrame(index=P.index)
X['mos_lag1']=np.log(S.shift(1)/Q.rolling(6).mean().shift(1))
X['mos_lag3']=X.mos_lag1.shift(2)
X['sales_yoy_lag1']=np.log(Q.rolling(3).mean()).diff(12).shift(1)
X['sales_momentum_lag1']=np.log(Q.rolling(3).mean()).diff(3).shift(1)
FEATURES=[None,*X.columns]
COMMON=pd.concat([R.rename('r'),R.shift().rename('lag'),X],axis=1).dropna().index


def fit(train,feature=None,season=False):
    active=[c for c in F if F.loc[train,c].nunique()>1]
    f=F.loc[train,active].to_numpy();fl=F.shift().loc[train,active].to_numpy()
    y=R.loc[train].to_numpy();lag=R.shift().loc[train].to_numpy()
    mean=float(X.loc[train,feature].mean()) if feature else 0.
    x=(X.loc[train,feature].to_numpy()-mean)[:,None] if feature else np.empty((len(train),0))
    seasonal=np.c_[np.sin(2*np.pi*train.month/12),np.cos(2*np.pi*train.month/12)] if season else np.empty((len(train),0))
    controls=np.c_[x,seasonal];k=len(active)
    def residual(z):return y-f@z[2:2+k]-z[0]-z[1]*(lag-fl@z[2:2+k])-controls@z[2+k:]
    size=2+k+controls.shape[1]
    starts=[np.r_[.3,phi,np.ones(k),np.zeros(controls.shape[1])] for phi in [.1,.6]]
    lower=np.full(size,-np.inf);upper=np.full(size,np.inf);lower[1]=-.95;upper[1]=.95
    fits=[least_squares(residual,start,bounds=(lower,upper),max_nfev=1500) for start in starts]
    opt=min([m for m in fits if m.success],key=lambda m:m.fun@m.fun)
    return dict(c=float(opt.x[0]),phi=float(opt.x[1]),beta=dict(zip(active,opt.x[2:2+k])),
                gamma=float(opt.x[2+k]) if feature else 0.,season_beta=opt.x[-2:].tolist() if season else [],
                x_mean=mean,feature=feature,season=season,n=len(train),
                rmse=float(np.sqrt(np.mean(opt.fun**2))))


def effect(m,origin):return sum(v*F.loc[origin,k] for k,v in m['beta'].items())


def predict(m,origin,dates,mode='held',delta=0.):
    state=float(R.loc[origin]-effect(m,origin));level=float(P.loc[origin]);levels=[];drivers=[]
    # First target's regressor uses data through origin at most.
    first=dates[0]
    if m['feature']:
        if first in X.index:initial=float(X.loc[first,m['feature']])
        else:
            initial=float(live_x(m['feature']))
    else:initial=0.
    for h,date in enumerate(dates,1):
        driver=initial+delta*min(h/6,1.)
        if mode=='realized_market_diagnostic':driver=float(X.loc[date,m['feature']]) if m['feature'] else 0.
        control=m['gamma']*(driver-m['x_mean'])
        if m['season']:
            control+=m['season_beta'][0]*np.sin(2*np.pi*date.month/12)+m['season_beta'][1]*np.cos(2*np.pi*date.month/12)
        state=m['c']+m['phi']*state+control
        # Future events not predicted. CPI here is actual in backtests, scenario in projection.
        level*=np.exp(state/100)*C.loc[date]/C.loc[date-pd.offsets.MonthBegin(1)]
        levels.append(level);drivers.append(driver)
    return pd.Series(levels,index=dates),drivers


def live_x(feature):
    if feature=='mos_lag1':return np.log(S.iloc[-1]/Q.iloc[-6:].mean())
    if feature=='mos_lag3':return X.mos_lag1.iloc[-2]
    if feature=='sales_yoy_lag1':return np.log(Q.iloc[-3:].mean()/Q.iloc[-15:-12].mean())
    if feature=='sales_momentum_lag1':return np.log(Q.iloc[-3:].mean()/Q.iloc[-6:-3].mean())
    raise ValueError(feature)


def annual(m,delta=0.):
    dates=pd.date_range('2026-10-01','2030-12-01',freq='MS')
    pred,_=predict(m,P.index[-1],dates,delta=delta);full=pd.concat([P,pred]);rows=[]
    for year in range(2026,2031):
        g=100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1)
        pi=100*(C.loc[f'{year}-12-01']/C.loc[f'{year-1}-12-01']-1)
        rows.append(dict(year=year,nominal_growth_pct=g,real_growth_pct=100*((1+g/100)/(1+pi/100)-1)))
    return rows


def main():
    summary=[];parameters={};forecasts=[];folds=[]
    for season in [False,True]:
        for feature in FEATURES:
            name=(feature or 'baseline')+('_season' if season else '')
            m=fit(COMMON,feature,season);parameters[name]=m
            summary.append(dict(model=name,**{k:v for k,v in m.items() if k not in ['beta','season_beta']}))
            for row in annual(m):forecasts.append(dict(model=name,case='current_state_held',**row))
            if feature in ['mos_lag1','mos_lag3']:
                for label,delta in [('sales_20pct_lower_stock_fixed',-np.log(.8)),('sales_20pct_higher_stock_fixed',-np.log(1.2))]:
                    for row in annual(m,delta):forecasts.append(dict(model=name,case=label,**row))
            for n in range(30,len(COMMON)):
                train=COMMON[:n];origin=train[-1];dates=COMMON[n:n+12]
                fm=fit(train,feature,season)
                pred,_=predict(fm,origin,dates)
                realized_market,_=predict(fm,origin,dates,mode='realized_market_diagnostic')
                for h in [1,6,12]:
                    if len(dates)<h:continue
                    error=100*(pred.iloc[h-1]-P.loc[dates[h-1]])/P.loc[origin]
                    folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=error,
                                      realized_market_diagnostic_error_pp=100*(realized_market.iloc[h-1]-P.loc[dates[h-1]])/P.loc[origin],
                                      gamma=fm['gamma'],phi=fm['phi']))
    f=pd.DataFrame(folds)
    metrics=f.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda s:np.sqrt(np.mean(s*s)),mae=lambda s:np.mean(abs(s))).reset_index()
    for season in [False,True]:
        ref='baseline'+('_season' if season else '')
        for h in [1,6,12]:
            b=metrics[(metrics.model==ref)&(metrics.horizon==h)].rmse.iloc[0]
            mask=(metrics.horizon==h)&metrics.model.str.endswith('_season').eq(season)
            metrics.loc[mask,'rmse_change_vs_matching_baseline_pct']=100*(metrics.loc[mask,'rmse']/b-1)
    metrics.to_csv(OUT/'backtest_metrics.csv',index=False)
    oracle=f.groupby(['model','horizon']).realized_market_diagnostic_error_pp.agg(n='size',rmse=lambda s:np.sqrt(np.mean(s*s)),mae=lambda s:np.mean(abs(s))).reset_index()
    oracle.to_csv(OUT/'realized_market_diagnostic_metrics.csv',index=False)
    f.to_csv(OUT/'backtest_folds.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'model_summary.csv',index=False)
    pd.DataFrame(forecasts).to_csv(OUT/'conditional_annual_forecasts.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    pd.concat([P.rename('price'),R.rename('real_log_pp'),Q.rename('sales_m2'),S.rename('unsold_begin_m2'),X,F],axis=1).to_csv(OUT/'historical_panel.csv',index_label='month')
    sample=[]
    for feature in ['mos_lag1','sales_yoy_lag1']:
        own=pd.concat([R,R.shift(),X[feature]],axis=1).dropna().index
        for label,dates in [('max_available',own),('common',COMMON),('from_2023',COMMON[COMMON>='2023-01-01'])]:
            m=fit(dates,feature)
            a=annual(m)
            sample.append(dict(feature=feature,sample=label,n=len(dates),gamma=m['gamma'],real_growth2027=a[1]['real_growth_pct']))
    pd.DataFrame(sample).to_csv(OUT/'sample_sensitivity.csv',index=False)
    stability=f[f.horizon==1].groupby('model').gamma.agg(min='min',max='max',negative_share=lambda s:np.mean(s<0))
    stability.to_csv(OUT/'coefficient_stability.csv')
    # No later data may influence an earlier parameter estimate.
    train=COMMON[:30];before=fit(train,'mos_lag1');saved=R.copy();R.loc[R.index>train[-1]]=123.
    after=fit(train,'mos_lag1');R[:]=saved
    assert np.allclose([before['c'],before['phi'],before['gamma']],[after['c'],after['phi'],after['gamma']])
    assert np.isfinite(pd.DataFrame(forecasts).select_dtypes('number')).all().all()
    diag=dict(common_start=str(COMMON[0].date()),common_end=str(COMMON[-1].date()),n=len(COMMON),checks_passed=True,
              future_assumptions='Current market driver held constant, not an approved sales/stock forecast. MOS stress moves log MOS by -log(0.8) over 6 months; opposite case -log(1.2).',
              yoy_assumption='Held annual sales-growth driver is diagnostic, not a level sales forecast.',
              validation='Pseudo-out-of-sample: revised history and retrospectively selected shock dates. Actual future CPI supplied; future market regressor held at first target value known at origin. Overlapping horizons.',
              other='No sign constraints or desired annual price-growth targets. Early event dates remain provisional. Stock publication lags not reconstructed.')
    (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    print(pd.DataFrame(summary)[['model','n','gamma','phi','rmse']].round(4).to_string(index=False))
    print(metrics.query('horizon==12').round(3).to_string(index=False))
    print(pd.DataFrame(forecasts).query('year==2027').round(3).to_string(index=False))


if __name__=='__main__':main()
