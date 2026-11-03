import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path(__file__).parent
s=pd.DataFrame(json.loads((OUT/'sales_monthly.json').read_text()))
s['period']=pd.to_datetime(s.period)
for c in s.columns.drop('period'): s[c]=pd.to_numeric(s[c])
s=s.set_index('period')
inputs=pd.read_csv(OUT/'mortgage_inputs.csv',parse_dates=['period']).set_index('period')
s=s.join(inputs,validate='one_to_one')
print('PV missing',s[s.pv_pct.isna()].index.tolist())
print('PV tail',s.pv_pct.tail(12).to_string())
def calc(d,label):
 d=d.copy();d['sales_mln']=(d.gross_rub-d.cancel_rub)/1e6;d['sales_th_m2']=(d.gross_m2-d.cancel_m2)/1000
 d['mortgage_mln']=d.ddu_mln/(1-d.pv_pct/100)
 d['cash_mln']=d.sales_mln-d.mortgage_mln;d['cash_share']=d.cash_mln/d.sales_mln
 d['cash_th_m2']=d.cash_share*d.sales_th_m2
 d['price_th_rub_m2']=d.sales_mln/d.sales_th_m2
 d.to_csv(OUT/(label+'_monthly.csv'),encoding='utf-8-sig')
 q=d.resample('QE').agg({'sales_mln':'sum','sales_th_m2':'sum','ddu_mln':'sum','mortgage_mln':lambda x:x.sum(min_count=3),'cash_mln':lambda x:x.sum(min_count=3),'cash_th_m2':lambda x:x.sum(min_count=3),'pv_pct':'count'})
 q['cash_share']=q.cash_mln/q.sales_mln
 q['cash_q_price_th_m2']=q.cash_share*q.sales_th_m2
 q['MA5']=q.cash_th_m2.rolling(5,min_periods=1).mean()
 q.to_csv(OUT/(label+'_quarterly.csv'),encoding='utf-8-sig')
 print(label,q[['sales_mln','cash_mln','cash_th_m2','MA5']].round(2).to_string())
 return d,q
d,q=calc(s,'current')
old=s.copy()
old.loc['2021-02-01','cancel_rub']+=213917842008.09;old.loc['2021-02-01','cancel_m2']+=1
old.loc['2025-04-01','gross_rub']+=78530593743.5-2505877482.33;old.loc['2025-04-01','gross_m2']+=65589.5-1329.83
old.loc['2025-02-01','gross_rub']+=33246276338.85+3149240846.77-872460342.51-257949699.23
old.loc['2025-02-01','gross_m2']+=46121.47+4873.56-1539.2-234.7
od,oq=calc(old,'before_logged_corrections')
diff=pd.DataFrame({'cash_current':q.cash_th_m2,'cash_before':oq.cash_th_m2,'delta':q.cash_th_m2-oq.cash_th_m2,'MA5_current':q.MA5,'MA5_before':oq.MA5,'MA5_delta':q.MA5-oq.MA5})
diff.to_csv(OUT/'corrections_impact.csv',encoding='utf-8-sig');print('IMPACT',diff.round(2).to_string())
print('negative months',d[d.cash_mln<0][['sales_mln','mortgage_mln','cash_mln']].round(2).to_string())
print('AGGREGATION DIFF',(q.cash_q_price_th_m2-q.cash_th_m2).round(2).to_string())

