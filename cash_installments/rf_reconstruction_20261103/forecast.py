"""Forecast cleaned RF cash/installments model through 2030; invert trailing MA5."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

p=Path(__file__).parent
f=pd.read_csv(p/'model_sensitivity_ARIMA.csv').query("source=='current' and end=='2026-09-30'").iloc[0]
q=pd.read_csv(p/'current_quarterly.csv',index_col=0,parse_dates=True)
k=pd.read_csv(p/'keyrate_quarterly.csv',index_col=0,parse_dates=True).iloc[:,0]
source=pd.read_csv(p/'ks_forecast_source.csv',index_col=0,parse_dates=True)
monthly=source.ks_new_applied.loc['2026-10-01':'2030-12-01']
assert monthly.index.equals(pd.date_range('2026-10-01','2030-12-01',freq='MS'))
days=pd.Series(monthly.index.days_in_month,index=monthly.index)
rates=(monthly*days).resample('QE').sum()/days.resample('QE').sum()
last_error=q.MA5.iloc[-1]-f.alpha-f.beta*k.iloc[-1]
history=q.cash_th_m2.tolist()
rows=[]
for h,(dt,rate) in enumerate(rates.items(),1):
    smoothed=f.alpha+f.beta*rate+f.phi**h*last_error
    original=5*smoothed-sum(history[-4:])
    history.append(original)
    np.testing.assert_allclose(np.mean(history[-5:]),smoothed)
    rows.append(dict(period=dt,key_rate_pct=rate,MA5_forecast_th_m2=smoothed,cash_forecast_th_m2=original))
forecast=pd.DataFrame(rows).set_index('period')
forecast.to_csv(p/'forecast_quarterly_2026_2030.csv',encoding='utf-8-sig')
annual=[]
previous=q.loc['2025','cash_th_m2'].sum()/1000
for year in range(2026,2031):
    actual=q.loc[str(year),'cash_th_m2'].sum()/1000 if year==2026 else 0
    projected=forecast.loc[str(year),'cash_forecast_th_m2'].sum()/1000
    total=actual+projected
    annual.append(dict(year=year,actual_mln_m2=actual,forecast_mln_m2=projected,total_mln_m2=total,yoy_pct=(total/previous-1)*100))
    previous=total
a=pd.DataFrame(annual)
a.to_csv(p/'forecast_annual_2026_2030.csv',index=False,encoding='utf-8-sig')
assert len(forecast)==17 and len(a)==5
assert (forecast.cash_forecast_th_m2>0).all()
print(a.round(6).to_string(index=False))
print('QUARTERS',forecast.round(3).to_string())
(p/'forecast_assumptions.json').write_text(json.dumps(dict(model='Regression with stationary AR(1) errors, exact Gaussian ML, cleaned data',parameters={z:float(f[z]) for z in ['alpha','beta','phi']},last_observation='2026-09-30',end='2030-12-31',key_rate_source='https://github.com/kR1SH1337/Real-Estate-Forecast/blob/main/cash_installments/scenarios/ks_2026-11-06/ks_monthly_audit.csv',key_rate_2030_provisional=True,monthly_rate_aggregation='calendar-day weighted quarterly mean',forecast_formula='MA5_h = alpha + beta * KS_h + phi**h * last_regression_error',MA5_inversion='cash_t = 5 * MA5_t - sum(cash_(t-1) through cash_(t-4))',units='thousand square meters quarterly; million square meters annually',scope='RF cash and installments combined, excluding mortgage',annual_2026='Actual Q1-Q3 plus forecast Q4'),indent=2),encoding='utf-8')

