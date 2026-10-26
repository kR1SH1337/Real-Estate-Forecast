"""Exploratory normalization, without forecast targets or deleting observations.
Calendar intervention effects are associations, not causal subsidy estimates.
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

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
D=pd.read_csv(ROOT.parent/'inputs/monthly_data.csv',index_col=0,parse_dates=True)
P=D['index'].dropna(); C=D.cpi_index_dec2020
R=(100*(np.log(P).diff()-np.log(C).diff())).reindex(P.index)
DATES=R.dropna().index[1:]
ENDS=pd.to_datetime(['2023-09-01','2023-12-01','2024-06-01','2026-01-01','2026-06-01','2026-09-01'])
MODELS=['baseline','huber','calendar','calendar_huber','regime','calendar_regime','calendar_season']


def features(index,width=2):
    f=pd.DataFrame(0.,index=index,columns=['pre','post','regime'])
    for end in ENDS:
        for j in range(width):
            date=end-pd.DateOffset(months=j)
            if date in f.index: f.loc[date,'pre']=1.
        for j in range(1,width+1):
            date=end+pd.DateOffset(months=j)
            if date in f.index: f.loc[date,'post']=1.
    f['regime']=(index>=pd.Timestamp('2024-07-01')).astype(float)
    f['sin12']=np.sin(2*np.pi*index.month/12)
    f['cos12']=np.cos(2*np.pi*index.month/12)
    return f


def estimate(train,model,width=2):
    allf=features(P.index,width)
    cols=(['pre','post'] if 'calendar' in model else [])+(['regime'] if 'regime' in model else [])+(['sin12','cos12'] if 'season' in model else [])
    # An unobserved regime change is not estimated or used in a historical forecast.
    cols=[c for c in cols if allf.loc[train,c].nunique()>1]
    f=allf.loc[train,cols].to_numpy()
    fl=allf.shift().loc[train,cols].to_numpy()
    y=R.loc[train].to_numpy(); lag=R.shift().loc[train].to_numpy()

    def errors(z):
        c,phi=z[:2]; beta=z[2:]
        return y-f@beta-c-phi*(lag-fl@beta)

    start=np.r_[y.mean(),.2,np.zeros(len(cols))]
    bounds=(np.r_[-np.inf,-.95,np.full(len(cols),-np.inf)],np.r_[np.inf,.95,np.full(len(cols),np.inf)])
    ols=least_squares(errors,start,bounds=bounds,max_nfev=1000)
    robust_scale=max(.1,1.4826*np.median(abs(ols.fun-np.median(ols.fun))))
    fit=least_squares(errors,ols.x,bounds=bounds,loss='huber',f_scale=1.345*robust_scale,max_nfev=1000) if 'huber' in model else ols
    assert fit.success
    return dict(c=float(fit.x[0]),phi=float(fit.x[1]),beta=dict(zip(cols,fit.x[2:])),
                width=width,scale=robust_scale,n=len(train),rmse=float(np.sqrt(np.mean(fit.fun**2))),
                event_pre_months=int(allf.loc[train,'pre'].sum()),event_post_months=int(allf.loc[train,'post'].sum()))


def effect(m,index):
    f=features(index,m['width'])
    return sum((v*f[k] for k,v in m['beta'].items()),start=pd.Series(0.,index=index))


def forecast(m,origin,dates):
    # No new future pre-deadline pulses; retain the payback window of past events.
    known_effect=effect(m,P.index)
    state=float(R.loc[origin]-known_effect.loc[origin]); level=float(P.loc[origin]); out=[]
    for date in dates:
        state=m['c']+m['phi']*state
        real=state+(m['beta'].get('regime',0.) if date>=pd.Timestamp('2024-07-01') else 0.)
        real+=m['beta'].get('sin12',0.)*np.sin(2*np.pi*date.month/12)+m['beta'].get('cos12',0.)*np.cos(2*np.pi*date.month/12)
        known_post=any(end<=origin and 1<=(date.year-end.year)*12+date.month-end.month<=m['width'] for end in ENDS)
        real+=m['beta'].get('post',0.)*known_post
        previous=date-pd.offsets.MonthBegin(1)
        level*=np.exp(real/100)*C.loc[date]/C.loc[previous]
        out.append(level)
    return pd.Series(out,index=dates)


def main():
    annual=[]; parameters=[]; audits=[]
    future=pd.date_range(P.index[-1]+pd.offsets.MonthBegin(1),'2030-12-01',freq='MS')
    models={name:estimate(DATES,name) for name in MODELS}
    for name,m in models.items():
        parameters.append(dict(model=name,**m))
        full=pd.concat([P,forecast(m,P.index[-1],future)])
        for year in range(2026,2031):
            growth=100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1)
            annual.append(dict(model=name,year=year,growth_pct=growth))
        if 'calendar' in name:
            f=features(P.index,m['width'])
            ev=m['beta'].get('pre',0.)*f.pre+m['beta'].get('post',0.)*f.post
            for date in R.dropna().index:
                audits.append(dict(model=name,month=str(date.date()),observed_real_log_pp=R.loc[date],
                                   estimated_calendar_component=ev.loc[date],
                                   adjusted_real_log_pp=R.loc[date]-ev.loc[date]))
    pd.DataFrame(annual).to_csv(OUT/'annual_forecasts.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    audit=pd.DataFrame(audits);audit.to_csv(OUT/'cleaning_audit.csv',index=False)
    # Effect on historical nominal year growth, calendar only, retaining every observation.
    cleaning=[]
    for name in ['calendar','calendar_huber','calendar_regime']:
        m=models[name];f=features(P.index)
        ev=effect(m,P.index)-m['beta'].get('regime',0.)*f.regime
        clean_g=100*np.log(P).diff()-ev
        for year in range(2022,2027):
            observed=100*np.expm1((100*np.log(P).diff()).loc[str(year)].sum()/100)
            adjusted=100*np.expm1(clean_g.loc[str(year)].sum()/100)
            cleaning.append(dict(model=name,year=year,months=int(P.loc[str(year)].size),observed_pct=observed,adjusted_pct=adjusted))
    pd.DataFrame(cleaning).to_csv(OUT/'historical_year_cleaning.csv',index=False)
    sensitivity=[]
    for width in [1,2,3]:
        for name in ['calendar','calendar_huber','calendar_regime']:
            m=estimate(DATES,name,width)
            full=pd.concat([P,forecast(m,P.index[-1],future)])
            for year in [2027,2030]:
                sensitivity.append(dict(model=name,width=width,year=year,growth_pct=100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1)))
    pd.DataFrame(sensitivity).to_csv(OUT/'window_sensitivity.csv',index=False)
    influence=[]
    for omitted_year in sorted(set(DATES.year)):
        m=estimate(DATES[DATES.year!=omitted_year],'baseline')
        full=pd.concat([P,forecast(m,P.index[-1],future)])
        influence.append(dict(omitted_year=int(omitted_year),growth2027_pct=100*(full.loc['2027-12-01']/full.loc['2026-12-01']-1),
                              note='Influence diagnostic only; omission is not a valid cleaning rule. Observed lag values retained.'))
    pd.DataFrame(influence).to_csv(OUT/'year_influence.csv',index=False)
    folds=[]
    for n in range(24,len(DATES)):
        train=DATES[:n];origin=train[-1]
        end=origin+pd.DateOffset(months=12)
        if end>P.index[-1]:continue
        dates=pd.date_range(origin+pd.offsets.MonthBegin(1),end,freq='MS')
        for name in MODELS:
            m=estimate(train,name)
            pred=forecast(m,origin,dates)
            for h in [6,12]:
                error=100*(pred.iloc[h-1]-P.loc[dates[h-1]])/P.loc[origin]
                folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=error))
    folds=pd.DataFrame(folds);folds.to_csv(OUT/'backtest_folds.csv',index=False)
    metrics=folds.groupby(['model','horizon']).error_pp.agg(n='size',rmse=lambda x:np.sqrt(np.mean(x*x)),mae=lambda x:np.mean(abs(x)))
    metrics.to_csv(OUT/'backtest_metrics.csv')
    fig,axs=plt.subplots(2,1,figsize=(10,8))
    axs[0].plot(R.index,R,color='gray',label='Observed monthly real growth')
    cal=models['calendar'];clean=R-effect(cal,P.index)
    axs[0].plot(clean.index,clean,label='Calendar adjusted growth',color='#007c91')
    axs[0].axhline(0,color='black',lw=.5);axs[0].set_ylabel('Monthly log pp');axs[0].legend()
    wide=pd.DataFrame(annual).pivot(index='year',columns='model',values='growth_pct')
    for name in MODELS:axs[1].plot(wide.index,wide[name],marker='o',label=name)
    axs[1].set(xticks=wide.index,ylabel='Price growth Dec/Dec (%)');axs[1].legend(ncol=3)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.svg');fig.savefig(OUT/'comparison.png',dpi=130);plt.close(fig)
    assert np.isfinite(wide).all().all()
    assert np.allclose(audit.observed_real_log_pp,audit.adjusted_real_log_pp+audit.estimated_calendar_component)
    # Future price observations must not affect estimates fitted at an earlier origin.
    test_train=DATES[:36];before=estimate(test_train,'calendar');saved=R.copy()
    R.loc[R.index>test_train[-1]]=999.
    after=estimate(test_train,'calendar');R[:]=saved
    assert np.allclose([before['c'],before['phi'],*before['beta'].values()],[after['c'],after['phi'],*after['beta'].values()])
    diagnostics=dict(n=len(DATES),targets_used=False,checks_passed=True,
                     backtest='Matched 32 overlapping 12-month origins; actual future CPI; no new future pre-deadline pulses; retain post-deadline tails of events already observed. Retrospectively assembled policy calendar, not vintage-real-time validation.',
                     calendar='Pooled pre/post effects; deadline windows chosen from policy timing, not price peaks. Window lengths remain assumptions. Causal subsidy impact not identified.')
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(wide.round(3).to_string());print(metrics.round(3).to_string());print(json.dumps(parameters,indent=2))


if __name__=='__main__':main()
