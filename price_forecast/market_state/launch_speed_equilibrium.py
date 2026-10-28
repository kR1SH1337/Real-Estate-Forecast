"""Three launch adjustment speeds, with all four market blocks solved jointly.
Price-model vintages are shown separately; coefficients are never selected by output.
"""
from pathlib import Path
import json,copy,hashlib
import coupled_forecast as c
import numpy as np,pandas as pd
class System(c.CoupledSystem):
 def __init__(self,kind,speed,omega=.5):
  super().__init__();self.kind=kind;self.snapshot['assumptions']['launch_adjustment_speed']=speed
  self.snapshot['assumptions']['launch_future_sales_weight']=omega
  if kind=='growth57':
   m=json.load(open(c.ROOT/'direct_sales_growth_results/parameters.json'))['ratio6_all_events']
   self.parameters=dict(c=m['c'],phi=m['phi'],beta=m['beta'],gamma=m['coefficients']['ratio6'],x_mean=m['means']['ratio6'],feature='mos_lag1',season=False,season_beta=[])
  if kind=='level57':
   self.level=json.load(open(c.ROOT/'sales_window_from2022_results/parameters.json'))['raw_avg6']
 def evaluate(self,guess):
  if self.kind!='level57':return super().evaluate(guess)
  ch=self.sales(guess);mkt=self.market(ch.total_sales_thousand_m2.to_numpy());p=self.level
  h=c.link.price;J=100*(h.P/h.C.reindex(h.P.index))/(h.P.iloc[0]/h.C.loc[h.P.index[0]])
  shift=sum(v*h.F[k].loc[:h.P.index[-1]].sum() for k,v in p['beta'].items());z=float(J.iloc[-1]-shift);prev=float(J.iloc[-1]);rows=[]
  for date in c.DATES:
   x=float(mkt['drivers'].loc[date,'mos_lag1']);z=p['c']+p['rho']*z+p['gamma']*(x-p['x_mean']);real=z+shift
   assert real>0
   price=h.P.iloc[0]*(real/100)*h.C.loc[date]/h.C.loc[h.P.index[0]]
   rows.append(dict(month=date,price_index=price,real_log_pp=100*np.log(real/prev)));prev=real
  return dict(channels=ch,market=mkt,model_price=pd.DataFrame(rows).set_index('month'))

def main():
 out=c.ROOT/'launch_speed_results';out.mkdir(exist_ok=True)
 rows=[];monthly=[];traces=[];by_class=[];checks=[];parameters={};scenarios=[]
 for kind in ['growth54','growth57','level57']:
  for speed in [.5,.75,1.]:
   system=System(kind,speed)
   relevant=copy.deepcopy(system.level if kind=='level57' else system.parameters)
   solution=system.solve(tolerance=1e-8)
   ev=solution['evaluation'];annual=c.link.yearly(pd.DataFrame({'price_index':solution['index']})).join(ev['channels'])
   for col in ['launches_annual','completions_annual','portfolio_closing','unsold_closing','sellout']:annual[col]=ev['market']['annual'][col]
   assert relevant==(system.level if kind=='level57' else system.parameters)
   assert np.allclose(system.sales(solution['index']).total_sales_thousand_m2,annual.total_sales_thousand_m2)
   assert np.allclose(annual.cash_total_thousand_m2,system.annual_channels.cash_total_thousand_m2)
   assert np.allclose(annual.mortgage_transactions_bln_rub,system.annual_channels.mortgage_transactions_bln_rub)
   parameters[kind]=relevant
   scenario=f'{kind}_alpha{int(speed*100)}'
   for year,row in annual.iterrows():
    rows.append(dict(scenario=scenario,model=kind,launch_speed=speed,year=int(year),**row.to_dict()))
   month=pd.DataFrame({'price_index':solution['index'],'model_price_index':ev['model_price'].price_index,'sales_m2':ev['market']['q'].loc[c.DATES],'unsold_begin_m2':ev['market']['stock'].loc[c.DATES],'months_of_sales_driver':np.exp(ev['market']['drivers'].loc[c.DATES,'mos_lag1'])})
   month.index.name='month';monthly.append(month.reset_index().assign(scenario=scenario,model=kind,launch_speed=speed))
   traces.append(solution['trace'].assign(scenario=scenario))
   by_class.append(ev['market']['by_class'].assign(scenario=scenario))
   snap=copy.deepcopy(system.snapshot)
   snap['assumptions']['sales']=annual.total_sales_thousand_m2.tolist()
   snap['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
   snap['forecast_scenario_provenance']={'price_model':kind,'launch_adjustment_speed':speed,'four_blocks_solved':True,'status':'Sensitivity, not selected base forecast'}
   reproduced=pd.DataFrame(c.portfolio.forecast(snap))
   assert np.allclose(reproduced.unsold_closing,ev['market']['by_class'].unsold_closing)
   scenarios.append(dict(scenario=scenario,snapshot=snap))
   check=dict(scenario=scenario,iterations=solution['iterations'],max_price_log_residual=solution['residual'])
   if speed==1.:
    alternative=system.solve(damping=.25,start_factor=1.15,tolerance=1e-8)
    deviation=float(np.max(abs(np.log(alternative['index']/solution['index']))))
    assert deviation<1e-6;check['alternative_start_deviation']=deviation;check['alternative_iterations']=alternative['iterations']
   checks.append(check)
 result=pd.DataFrame(rows)
 baseline_saved=pd.read_csv(c.ROOT/'coupled_results/annual_equilibrium.csv').set_index('year')
 baseline=result[result.scenario=='growth54_alpha50'].set_index('year')
 assert np.allclose(baseline.nominal_growth_pct,baseline_saved.nominal_growth_pct,atol=1e-6)
 assert np.allclose(baseline.total_sales_thousand_m2,baseline_saved.total_sales_thousand_m2,atol=1e-5)
 for kind in ['growth54','growth57','level57']:
  opening=result[(result.model==kind)&(result.year==2026)]
  assert np.ptp(opening.nominal_growth_pct)<1e-6
 result.to_csv(out/'annual_comparison.csv',index=False)
 pd.concat(monthly,ignore_index=True).to_csv(out/'monthly_paths.csv',index=False)
 pd.concat(traces,ignore_index=True).to_csv(out/'iteration_traces.csv',index=False)
 pd.concat(by_class,ignore_index=True).to_csv(out/'portfolio_by_class.csv',index=False)
 pd.DataFrame(checks).to_csv(out/'convergence_checks.csv',index=False)
 (out/'portfolio_scenarios.json').write_text(json.dumps(scenarios,ensure_ascii=False,indent=2))
 (out/'parameters.json').write_text(json.dumps(parameters,indent=2))
 (out/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,launch_speeds=[.5,.75,1.],launch_sales_ratio=1.149,
  price_models={'growth54':'Original production monthly real-growth model, Apr2022-Sep2026 N54','growth57':'Same form re-estimated Jan2022-Sep2026 N57, near-zero positive market coefficient','level57':'Real-index level with cumulative events, Jan2022-Sep2026 N57'},
  coefficient_changes_within_launch_scenario=False,autonomous_price_correction=0.,fixed_inputs='Mortgage nominal values, cash area, CPI, launch ratio, class allocation, completion profiles and delays, price anchor, terminal Q2031=Q2030.',
  solver='All future monthly prices solved jointly with endogenous mortgage areas, launches and cohort inventories. Alternate initialization/damping checked at alpha=1 for every price model.',
  level_seed='Level model uses the original growth path only as a numerical starting guess; its evaluation and final fixed point use level coefficients exclusively.',
  period_warning='Different price samples are sensitivity, not a parameter selection by desired output; p-values and historical forecast limitations remain unchanged.',
  future_launches='2026 launches fixed; alpha changes only 2027 onward.',status='Research sensitivity; no final price model or launch speed selected'),indent=2))
 print(result[['model','launch_speed','year','nominal_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing']].round(4).to_string(index=False))


if __name__=='__main__':main()
