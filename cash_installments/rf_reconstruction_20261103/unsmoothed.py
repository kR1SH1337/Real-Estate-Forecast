"""Compare the same exact-ML ARIMAX on raw quarterly area and trailing MA5."""
from pathlib import Path
from math import erfc,sqrt
import ast,json
import numpy as np
import pandas as pd

p=Path(__file__).parent
# Reuse the exact likelihood fitter without executing the historical batch script.
tree=ast.parse((p/'fit.py').read_text(encoding='utf-8'))
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_ar1')
exec(compile(ast.Module(body=[fun],type_ignores=[]),'fit.py','exec'))
q=pd.read_csv(p/'current_quarterly.csv',index_col=0,parse_dates=True)
x=pd.read_csv(p/'keyrate_quarterly.csv',index_col=0,parse_dates=True).iloc[:,0].to_numpy()
raw=q.cash_th_m2.to_numpy();smooth=q.MA5.to_numpy()
np.testing.assert_allclose(smooth,pd.Series(raw).rolling(5,min_periods=1).mean())

def inference(y,fit):
 c,b,phi=[fit[z] for z in ['alpha','beta','phi']]
 u=y-c-b*x; e=u[1:]-phi*u[:-1]
 s2=((1-phi**2)*u[0]**2+e@e)/len(y)
 theta=np.array([c/1000,b/100,phi,np.log(s2)])
 def nll(t):
  u=y-t[0]*1000-t[1]*100*x;phi=t[2];e=u[1:]-phi*u[:-1]
  ss=(1-phi**2)*u[0]**2+e@e
  return .5*(len(y)*(np.log(2*np.pi)+t[3])-np.log(1-phi**2)+ss/np.exp(t[3]))
 def cov(step):
  h=np.eye(4)*step;H=np.zeros((4,4));v=nll(theta)
  for i in range(4):
   H[i,i]=(nll(theta+h[i])-2*v+nll(theta-h[i]))/step**2
   for j in range(i):
    H[i,j]=H[j,i]=(nll(theta+h[i]+h[j])-nll(theta+h[i]-h[j])-nll(theta-h[i]+h[j])+nll(theta-h[i]-h[j]))/(4*step**2)
  assert np.linalg.eigvalsh(H).min()>0
  return np.sqrt(np.diag(np.linalg.inv(H)))*[1000,100,1,1]
 ses=[cov(h) for h in [2e-4,1e-4,5e-5]]
 np.testing.assert_allclose(ses[0],ses[2],rtol=1e-4)
 rows=[]
 for name,value,se in zip(['constant','key_rate','AR1'],[c,b,phi],ses[1][:3]):
  rows.append(dict(parameter=name,estimate=value,std_error=se,p_value=erfc(abs(value/se)/sqrt(2)),ci95_low=value-1.95996398454*se,ci95_high=value+1.95996398454*se))
 return rows

models={};parameters=[];fitrows=[]
for name,y in [('MA5',smooth),('raw',raw)]:
 f=exact_ar1(y,x);models[name]=f
 par=inference(y,f)
 parameters.extend([dict(model=name,**r) for r in par])
 pred=f['alpha']+f['beta']*x
 pred[1:]+=f['phi']*(y[:-1]-f['alpha']-f['beta']*x[:-1])
 err=y[1:]-pred[1:]
 r2=1-(err@err)/np.sum((y[1:]-y[1:].mean())**2)
 fitrows.append(dict(model=name,n=len(y),R2_one_step_excl_first=r2,MAE=np.mean(abs(err)),**f))
pd.DataFrame(parameters).to_csv(p/'unsmoothed_parameters.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(fitrows).to_csv(p/'unsmoothed_fit.csv',index=False,encoding='utf-8-sig')

def predict(train_raw,train_x,future_x,use_ma5):
 y=pd.Series(train_raw).rolling(5,min_periods=1).mean().to_numpy() if use_ma5 else train_raw
 f=exact_ar1(y,train_x);u=y[-1]-f['alpha']-f['beta']*train_x[-1]
 values=list(train_raw);res=[]
 for h,rate in enumerate(future_x,1):
  val=f['alpha']+f['beta']*rate+f['phi']**h*u
  if use_ma5:val=5*val-sum(values[-4:])
  values.append(val);res.append(val)
 return np.array(res)

tests=[]
for horizon in [1,4]:
 for t in range(12,len(raw)-horizon+1):
  actual=raw[t:t+horizon]
  predictions={name:predict(raw[:t],x[:t],x[t:t+horizon],ma) for name,ma in [('MA5',True),('raw',False)]}
  predictions['last_quarter']=np.repeat(raw[t-1],horizon)
  predictions['seasonal_naive']=raw[t-4:t-4+horizon]
  for name,pred in predictions.items():
   tests.append(dict(model=name,horizon=horizon,train_end=str(q.index[t-1].date()),target_end=str(q.index[t+horizon-1].date()),actual=actual.sum(),forecast=pred.sum(),error=(pred-actual).sum()))
test=pd.DataFrame(tests)
test.to_csv(p/'unsmoothed_backtest_detail.csv',index=False,encoding='utf-8-sig')
summary=[]
for (name,h),g in test.groupby(['model','horizon']):
 summary.append(dict(model=name,horizon=h,n=len(g),RMSE=np.sqrt(np.mean(g.error**2)),MAE=np.mean(abs(g.error)),WAPE_pct=100*abs(g.error).sum()/abs(g.actual).sum(),bias_pct=100*g.error.sum()/abs(g.actual).sum()))
pd.DataFrame(summary).to_csv(p/'unsmoothed_backtest.csv',index=False,encoding='utf-8-sig')
(p/'unsmoothed_models.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
print('PARAMETERS\n',pd.DataFrame(parameters).round(6).to_string(index=False))
print('FIT\n',pd.DataFrame(fitrows).round(6).to_string(index=False))
print('BACKTEST (all errors on original area, th. m2)\n',pd.DataFrame(summary).round(4).to_string(index=False))

