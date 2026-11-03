"""Compare MA5-ARIMAX and static raw OLS forecasts on the same scenario."""
from pathlib import Path
import pandas as pd
import numpy as np

p=Path(__file__).parent
q=pd.read_csv(p/'current_quarterly.csv',index_col=0,parse_dates=True)
f=pd.read_csv(p/'forecast_quarterly_2026_2030.csv',index_col=0,parse_dates=True)
b=pd.read_csv(p/'without_ar_parameters.csv').set_index('parameter').estimate
X=np.column_stack([np.ones(len(q)),pd.read_csv(p/'keyrate_quarterly.csv',index_col=0).iloc[:,0]])
np.testing.assert_allclose(np.linalg.lstsq(X,q.cash_th_m2,rcond=None)[0],b.loc[['constant','key_rate']])
f=f.rename(columns={'cash_forecast_th_m2':'ARIMAX_MA5_th_m2'})
f['OLS_KS_th_m2']=b['constant']+b['key_rate']*f.key_rate_pct
f['difference_th_m2']=f.OLS_KS_th_m2-f.ARIMAX_MA5_th_m2
f.to_csv(p/'forecast_comparison_quarterly.csv',encoding='utf-8-sig')
rows=[];prev_ar=prev_ols=q.loc['2025','cash_th_m2'].sum()/1000
for year in range(2026,2031):
    history=q.loc['2026','cash_th_m2'].sum()/1000 if year==2026 else 0
    ar=f.loc[str(year),'ARIMAX_MA5_th_m2'].sum()/1000+history
    ols=f.loc[str(year),'OLS_KS_th_m2'].sum()/1000+history
    rows.append(dict(year=year,actual_mln_m2=history,ARIMAX_MA5_mln_m2=ar,OLS_KS_mln_m2=ols,difference_mln_m2=ols-ar,difference_pct=(ols/ar-1)*100,ARIMAX_yoy_pct=(ar/prev_ar-1)*100,OLS_yoy_pct=(ols/prev_ols-1)*100))
    prev_ar,prev_ols=ar,ols
a=pd.DataFrame(rows)
a.to_csv(p/'forecast_comparison_annual.csv',index=False,encoding='utf-8-sig')
np.testing.assert_allclose(a.ARIMAX_MA5_mln_m2,pd.read_csv(p/'forecast_annual_2026_2030.csv').total_mln_m2)
back=pd.read_csv(p/'without_ar_backtest.csv')
back=back[back.model.isin(['MA5','OLS_no_AR'])]
back.to_csv(p/'forecast_comparison_backtest.csv',index=False,encoding='utf-8-sig')
assert len(f)==17 and len(a)==5
print('ANNUAL\n',a.round(6).to_string(index=False))
print('2026 Q4\n',f.head(1).round(6).to_string())
print('BACKTEST\n',back.round(4).to_string(index=False))

