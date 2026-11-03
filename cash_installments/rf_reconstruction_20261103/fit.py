from pathlib import Path
OUT=Path(__file__).parent
import pandas as pd
import numpy as np

k=pd.read_csv(OUT/'keyrate.csv');k.columns=['date','rate'];k['date']=pd.to_datetime(k.date,dayfirst=True)
k=k.set_index('date').sort_index().reindex(pd.date_range('2021-01-01','2026-09-30')).ffill().bfill()
ks=k.rate.resample('QE').mean()
ks.to_csv(OUT/'keyrate_quarterly.csv',encoding='utf-8-sig')
rows=[]
for file in ['current','before_logged_corrections']:
 q=pd.read_csv(OUT/(file+'_quarterly.csv'),index_col=0,parse_dates=True)
 for end in ['2025-12-31','2026-03-31','2026-06-30','2026-09-30']:
  for variant in ['base','full_MA5','quarter_price','floor_zero']:
   z=q.loc[:end].copy()
   if variant=='quarter_price':z['MA5']=z.cash_q_price_th_m2.rolling(5,min_periods=1).mean()
   if variant=='floor_zero':z['MA5']=z.cash_th_m2.clip(lower=0).rolling(5,min_periods=1).mean()
   if variant=='full_MA5':z=z.iloc[4:]
   y=z.MA5;X=np.column_stack([np.ones(len(y)-1),ks.reindex(z.index).values[1:],y.values[:-1]])
   co=np.linalg.lstsq(X,y.values[1:],rcond=None)[0]
   err=y.values[1:]-X@co
   rows.append(dict(source=file,end=end,variant=variant,model='OLS_lag_y',n=len(err),alpha=co[0],beta=co[1],phi=co[2],RMSE=np.sqrt(np.mean(err**2)),MAE=np.mean(abs(err))))
pd.DataFrame(rows).to_csv(OUT/'model_sensitivity_OLS.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).query("variant=='base'").round(4).to_string(index=False))
def exact_ar1(y,x):
 # Exact Gaussian ML, stationary AR(1) errors; profile out regression and variance.
 y=np.asarray(y);x=np.column_stack([np.ones(len(y)),np.asarray(x)])
 def obj(phi,details=False):
  yy=np.r_[y[0]*np.sqrt(1-phi**2),y[1:]-phi*y[:-1]]
  xx=np.vstack([x[0]*np.sqrt(1-phi**2),x[1:]-phi*x[:-1]])
  coef=np.linalg.lstsq(xx,yy,rcond=None)[0];e=yy-xx@coef
  value=len(y)*np.log(e@e/len(y))-np.log(1-phi**2)
  return (value,coef,e) if details else value
 grid=np.linspace(-.99999,.99999,2001);i=np.argmin([obj(p) for p in grid]);a=grid[max(0,i-1)];b=grid[min(len(grid)-1,i+1)]
 for _ in range(100):
  l=a+(b-a)*.38196601125;r=a+(b-a)*.61803398875
  if obj(l)<obj(r):b=r
  else:a=l
 phi=(a+b)/2;value,coef,e=obj(phi,True)
 residual=y-x@coef
 forecast=x@coef;forecast[1:]+=phi*residual[:-1]
 return dict(alpha=coef[0],beta=coef[1],phi=phi,RMSE=np.sqrt(np.mean((y[1:]-forecast[1:])**2)),profile_objective=value)
ar=[]
for file in ['current','before_logged_corrections']:
 q=pd.read_csv(OUT/(file+'_quarterly.csv'),index_col=0,parse_dates=True)
 for end in ['2025-12-31','2026-03-31','2026-06-30','2026-09-30']:
  z=q.loc[:end].copy();y=z.MA5
  fit=exact_ar1(y,ks.reindex(z.index))
  ar.append(dict(source=file,end=end,n=len(y),**fit))
pd.DataFrame(ar).to_csv(OUT/'model_sensitivity_ARIMA.csv',index=False,encoding='utf-8-sig');print(pd.DataFrame(ar).round(4).to_string(index=False))

