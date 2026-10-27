"""Exploratory ablation of target, market control and event correction.
Oracle event-calendar metrics are diagnostic, not usable forecast scores.
"""
import json
from pathlib import Path
import nominal_index_experiment as n
import numpy as np,pandas as pd
from scipy.optimize import least_squares
old=n.old
T={'growth':100*np.log(old.P/old.P.shift()),'log':n.TARGETS['log'],'raw':old.P}

def fit(train,kind,market,events):
 target=T[kind];D=old.F if kind=='growth' else n.SHIFTS
 active=[k for k in D if D.loc[train,k].nunique()>1] if events else []
 y=target.loc[train].to_numpy();lag=target.shift().loc[train].to_numpy();d=D.loc[train,active].to_numpy();dl=D.shift().loc[train,active].to_numpy();k=len(active)
 mean=float(old.X.loc[train,'mos_lag1'].mean()) if market else 0.;x=old.X.loc[train,'mos_lag1'].to_numpy()-mean if market else np.zeros(len(train))
 def residual(p):return y-d@p[2:2+k]-p[0]-p[1]*(lag-dl@p[2:2+k])-(p[-1]*x if market else 0.)
 opts=[]
 for rho in ([.1,.6] if kind=='growth' else [.5,.9,1.,1.05]):
  start=np.r_[np.mean(y-rho*lag),rho,np.ones(k),np.zeros(1 if market else 0)]
  bounds=(-np.inf,np.inf)
  if kind=='growth':
   low=np.full(len(start),-np.inf);high=np.full(len(start),np.inf);low[1]=-.95;high[1]=.95;bounds=(low,high)
  f=least_squares(residual,start,bounds=bounds,max_nfev=4000,ftol=1e-11,xtol=1e-11,gtol=1e-11)
  if f.success:opts.append(f)
 f=min(opts,key=lambda f:f.fun@f.fun)
 return dict(kind=kind,market=market,events=events,c=float(f.x[0]),rho=float(f.x[1]),beta=dict(zip(active,map(float,f.x[2:2+k]))),gamma=float(f.x[-1]) if market else 0.,mean=mean)

def predict(m,origin,dates,oracle=False):
 D=old.F if m['kind']=='growth' else n.SHIFTS
 effect=lambda date:sum(v*D.loc[date,k] for k,v in m['beta'].items())
 state=float(T[m['kind']].loc[origin]-effect(origin));level=float(old.P.loc[origin]);rows=[]
 x=float(old.X.loc[dates[0],'mos_lag1']) if m['market'] else 0.
 for date in dates:
  state=m['c']+m['rho']*state+m['gamma']*(x-m['mean'])
  if m['kind']=='growth':level*=np.exp((state+(effect(date) if oracle else 0.))/100)
  else:
   y=state+(effect(date) if oracle else effect(origin));level=old.P.iloc[0]*np.exp(y/100) if m['kind']=='log' else y
  assert np.isfinite(level) and level>0
  rows.append(level)
 return np.array(rows)
def main():
    out=Path(__file__).resolve().parent/'nominal_ablation_results';out.mkdir(exist_ok=True)
    folds=[];models={}
    for kind in ['growth','log','raw']:
     for market in [False,True]:
      for events in [False,True]:
       name=f'{kind}_market{int(market)}_events{int(events)}';models[name]=fit(old.COMMON,kind,market,events)
       for nn in range(30,len(old.COMMON)):
        train=old.COMMON[:nn];origin=train[-1];dates=old.COMMON[nn:nn+12];m=fit(train,kind,market,events);p=predict(m,origin,dates);oracle=predict(m,origin,dates,True)
        for h in [1,6,12]:
         if len(dates)<h:continue
         ds=dates[:h];events_future=old.F.loc[ds].sum();recent_available='recent' in m['beta'];fact=old.P.loc[ds[-1]]
         folds.append(dict(model=name,origin=str(origin.date()),horizon=h,error_pp=100*(p[h-1]-fact)/old.P.loc[origin],oracle_error_pp=100*(oracle[h-1]-fact)/old.P.loc[origin],future_event_months=int(old.F.loc[ds].any(axis=1).sum()),future_recent_events=int(events_future.recent),recent_coefficient_available=recent_available,rho=m['rho'],gamma=m['gamma']))
    f=pd.DataFrame(folds)
    a=f.groupby(['model','horizon']).agg(n=('error_pp','size'),rmse=('error_pp',lambda v:np.sqrt(np.mean(v*v))),bias=('error_pp','mean'),mae=('error_pp',lambda v:np.mean(abs(v))),oracle_rmse=('oracle_error_pp',lambda v:np.sqrt(np.mean(v*v)))).reset_index()
    print(a.round(4).to_string(index=False))
    print('EVENT WINDOWS');print(f[f.model=='growth_market1_events1'].groupby('horizon').agg(n=('origin','size'),no_event_windows=('future_event_months',lambda v:sum(v==0)),recent_beta_available=('recent_coefficient_available','sum'),min_events=('future_event_months','min'),max_events=('future_event_months','max')).to_string())
    f.to_csv(out/'backtest_folds.csv',index=False);a.to_csv(out/'backtest_metrics.csv',index=False);(out/'parameters.json').write_text(json.dumps(models,indent=2))

    non_event=f[f.horizon==1].groupby(['model','future_event_months']).error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v)),bias='mean').reset_index()
    non_event.to_csv(out/'monthly_event_split.csv',index=False)
    calendar=f[f.model=='growth_market1_events1'].groupby('horizon').agg(n=('origin','size'),no_event_windows=('future_event_months',lambda v:int(sum(v==0))),recent_beta_available=('recent_coefficient_available','sum'),min_events=('future_event_months','min'),max_events=('future_event_months','max')).reset_index()
    calendar.to_csv(out/'event_windows.csv',index=False)

    subset=f[(f.horizon==1)&(f.future_event_months==0)].groupby('model').error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
    mask=f[(f.model=='growth_market0_events1')&(f.horizon==1)][['origin','future_event_months']]
    for path,selected in [(n.ROOT/'results/backtest_folds.csv',['baseline','mos_lag1']),(n.OUT/'backtest_folds.csv',['nominal_drift','nominal_random_walk'])]:
        reference=pd.read_csv(path)
        reference=reference[(reference.horizon==1)&reference.model.isin(selected)].merge(mask,on='origin')
        reference=reference[reference.future_event_months==0].groupby('model').error_pp.agg(n='size',rmse=lambda v:np.sqrt(np.mean(v*v))).reset_index()
        subset=pd.concat([subset,reference],ignore_index=True)
    subset.to_csv(out/'non_event_monthly_comparison.csv',index=False)
    assert (calendar.set_index('horizon').loc[[6,12],'no_event_windows']==0).all()

    # Matching level variants must reproduce the preceding nominal experiment.
    previous=pd.read_csv(n.OUT/'backtest_metrics.csv')
    names={'raw_market1_events1':'nominal_index_market','raw_market0_events1':'nominal_index_no_market','log_market1_events1':'nominal_log_market','log_market0_events1':'nominal_log_no_market'}
    for name,reference in names.items():
        assert np.allclose(a[a.model==name].sort_values('horizon').rmse,previous[previous.model==reference].sort_values('horizon').rmse)
    assert np.isfinite(a[['rmse','bias','mae','oracle_rmse']]).all().all()
    (out/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,target='Nominal levels, log levels or monthly log growth',validation='Same expanding windows, actual nominal outcomes, no CPI, first known market held fixed, no future events in standard forecasts',oracle='Future event calendar supplied only in oracle_error_pp. Event magnitudes use training coefficients; unseen groups have no estimated coefficient. This is not a usable forecast.',interpretation='Exploratory comparisons on one short, overlapping evaluation period; not independent model selection or a validation of a latent shock-free price path.',production_model_changed=False),indent=2))

if __name__=='__main__':main()
