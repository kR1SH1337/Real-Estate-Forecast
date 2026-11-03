"""ARX(1) with lagged dependent variable versus matched-sample static OLS."""
from pathlib import Path
from math import lgamma,exp,sqrt,pi,atan
import ast,json
import numpy as np
import pandas as pd

p=Path(__file__).parent
tree=ast.parse((p/'without_ar.py').read_text(encoding='utf-8'))
fn=next(z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name=='student_p')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'without_ar.py','exec'))
q=pd.read_csv(p/'current_quarterly.csv',index_col=0,parse_dates=True)
y=q.cash_th_m2.to_numpy()
x=pd.read_csv(p/'keyrate_quarterly.csv',index_col=0,parse_dates=True).iloc[:,0].to_numpy()
target=y[1:];n=len(target)
designs={'static_matched':np.column_stack([np.ones(n),x[1:]]),
         'lagged_sales':np.column_stack([np.ones(n),x[1:],y[:-1]])}
metrics=[];params=[];coefs={}
for name,X in designs.items():
    b=np.linalg.lstsq(X,target,rcond=None)[0];coefs[name]=b.tolist()
    residual=target-X@b;sse=residual@residual;k=X.shape[1];df=n-k
    np.testing.assert_allclose((X.T@residual)/(np.linalg.norm(X,axis=0)*np.linalg.norm(residual)),0,atol=1e-12)
    se=np.sqrt(np.diag(sse/df*np.linalg.inv(X.T@X)))
    for label,v,s in zip(['constant','key_rate','sales_lag_1'],b,se):
        params.append(dict(model=name,parameter=label,estimate=v,std_error=s,t=v/s,df=df,p_value=student_p(v/s,df)))
    r2=1-sse/np.sum((target-target.mean())**2)
    ll=-.5*n*(np.log(2*pi)+1+np.log(sse/n))
    metrics.append(dict(model=name,n=n,R2=r2,adjusted_R2=1-(1-r2)*(n-1)/df,RMSE=np.sqrt(sse/n),MAE=np.mean(abs(residual)),AIC=-2*ll+2*(k+1),BIC=-2*ll+np.log(n)*(k+1)))
pd.DataFrame(params).to_csv(p/'lagged_sales_parameters.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(metrics).to_csv(p/'lagged_sales_fit.csv',index=False,encoding='utf-8-sig')
tests=[]
for h in [1,4]:
    for t in range(12,len(y)-h+1):
        for name,X in designs.items():
            # rows 0:t-1 represent target quarters 1:t, no future sales used.
            b=np.linalg.lstsq(X[:t-1],y[1:t],rcond=None)[0]
            previous=y[t-1];forecast=[]
            for rate in x[t:t+h]:
                pred=b[0]+b[1]*rate
                if name=='lagged_sales':pred+=b[2]*previous
                forecast.append(pred);previous=pred
            actual=y[t:t+h].sum();predsum=sum(forecast)
            tests.append(dict(model=name,horizon=h,train_end=str(q.index[t-1].date()),target_end=str(q.index[t+h-1].date()),actual=actual,forecast=predsum,error=predsum-actual))
base=pd.read_csv(p/'without_ar_backtest_detail.csv')
detail=pd.concat([base,pd.DataFrame(tests)],ignore_index=True)
detail.to_csv(p/'lagged_sales_backtest_detail.csv',index=False,encoding='utf-8-sig')
summary=[]
for (name,h),g in detail.groupby(['model','horizon']):
    summary.append(dict(model=name,horizon=h,n=len(g),RMSE=np.sqrt(np.mean(g.error**2)),MAE=np.mean(abs(g.error)),WAPE_pct=100*abs(g.error).sum()/abs(g.actual).sum(),bias_pct=100*g.error.sum()/abs(g.actual).sum()))
pd.DataFrame(summary).to_csv(p/'lagged_sales_backtest.csv',index=False,encoding='utf-8-sig')
(p/'lagged_sales_models.json').write_text(json.dumps(coefs,indent=2),encoding='utf-8')
assert len(pd.DataFrame(tests).query("model=='lagged_sales' and horizon==1"))==11
assert len(pd.DataFrame(tests).query("model=='lagged_sales' and horizon==4"))==8
print('PARAMETERS\n',pd.DataFrame(params).round(6).to_string(index=False))
print('FIT\n',pd.DataFrame(metrics).round(6).to_string(index=False))
print('BACKTEST\n',pd.DataFrame(summary).round(4).to_string(index=False))

