"""OLS versus unsmoothed exact-ML ARIMAX, identical data and rolling windows."""
from pathlib import Path
from math import lgamma,exp,sqrt,pi,atan,erfc
import json
import numpy as np
import pandas as pd

p=Path(__file__).parent
q=pd.read_csv(p/'current_quarterly.csv',index_col=0,parse_dates=True)
y=q.cash_th_m2.to_numpy()
x=pd.read_csv(p/'keyrate_quarterly.csv',index_col=0,parse_dates=True).iloc[:,0].to_numpy()
X=np.column_stack([np.ones(len(x)),x])
b=np.linalg.lstsq(X,y,rcond=None)[0]
res=y-X@b;n=len(y);df=n-2;sse=res@res
cov=sse/df*np.linalg.inv(X.T@X);se=np.sqrt(np.diag(cov))
def student_p(t,df):
    # Integrate transformed Student-t density on a finite interval; two-sided tail.
    a=atan(abs(t)/sqrt(df));N=16384
    z=np.linspace(a,pi/2,N+1);f=np.cos(z)**(df-1)
    integral=(pi/2-a)/(3*N)*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum())
    return 2*exp(lgamma((df+1)/2)-.5*np.log(pi)-lgamma(df/2))*integral
assert abs(student_p(0,df)-1)<1e-10
params=[]
for name,value,s in zip(['constant','key_rate'],b,se):
    params.append(dict(parameter=name,estimate=value,std_error=s,t=value/s,df=df,p_value=student_p(value/s,df)))
pd.DataFrame(params).to_csv(p/'without_ar_parameters.csv',index=False,encoding='utf-8-sig')
ll=-.5*n*(np.log(2*pi)+1+np.log(sse/n))
armodel=json.loads((p/'unsmoothed_models.json').read_text())['raw']
arll=-.5*(n*(np.log(2*pi)+1)+armodel['profile_objective'])
arfit=pd.read_csv(p/'unsmoothed_fit.csv').query("model=='raw'").iloc[0]
ols=dict(model='OLS_no_AR',n=n,R2_all=1-sse/np.sum((y-y.mean())**2),R2_excl_first=1-np.sum(res[1:]**2)/np.sum((y[1:]-y[1:].mean())**2),RMSE_excl_first=np.sqrt(np.mean(res[1:]**2)),MAE_excl_first=np.mean(abs(res[1:])),loglik=ll,AIC=-2*ll+2*3,BIC=-2*ll+np.log(n)*3)
ar_pred=armodel['alpha']+armodel['beta']*x
ar_pred[1:]+=armodel['phi']*(y[:-1]-armodel['alpha']-armodel['beta']*x[:-1])
ar=dict(model='ARIMAX_raw',n=n,R2_all=1-np.sum((y-ar_pred)**2)/np.sum((y-y.mean())**2),R2_excl_first=arfit.R2_one_step_excl_first,RMSE_excl_first=arfit.RMSE,MAE_excl_first=arfit.MAE,loglik=arll,AIC=-2*arll+2*4,BIC=-2*arll+np.log(n)*4)
lr=2*(arll-ll)
metrics=dict(OLS=ols,ARIMAX=ar,LR_stat=lr,LR_pvalue_chi2_1=erfc(sqrt(max(0,lr)/2)))
assert lr>=-1e-7
(p/'without_ar_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
rows=[]
for horizon in [1,4]:
    for t in range(12,n-horizon+1):
        btrain=np.linalg.lstsq(X[:t],y[:t],rcond=None)[0]
        pred=X[t:t+horizon]@btrain
        actual=y[t:t+horizon]
        rows.append(dict(model='OLS_no_AR',horizon=horizon,train_end=str(q.index[t-1].date()),target_end=str(q.index[t+horizon-1].date()),actual=actual.sum(),forecast=pred.sum(),error=pred.sum()-actual.sum()))
base=pd.read_csv(p/'unsmoothed_backtest_detail.csv')
tests=pd.concat([base,pd.DataFrame(rows)],ignore_index=True)
tests.to_csv(p/'without_ar_backtest_detail.csv',index=False,encoding='utf-8-sig')
summary=[]
for (name,h),g in tests.groupby(['model','horizon']):
    summary.append(dict(model=name,horizon=h,n=len(g),RMSE=np.sqrt(np.mean(g.error**2)),MAE=np.mean(abs(g.error)),WAPE_pct=100*abs(g.error).sum()/abs(g.actual).sum(),bias_pct=100*g.error.sum()/abs(g.actual).sum()))
pd.DataFrame(summary).to_csv(p/'without_ar_backtest.csv',index=False,encoding='utf-8-sig')
print('PARAMETERS\n',pd.DataFrame(params).to_string(index=False))
print('METRICS\n',json.dumps(metrics,indent=2))
print('BACKTEST\n',pd.DataFrame(summary).round(4).to_string(index=False))

