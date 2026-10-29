"""User-selected 10% Dec/Dec 2026 target with endogenous sales and portfolio."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from launch_speed_equilibrium import System,c
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'target_2026_results'
TARGET=.10
class TargetSystem(System):
 def __init__(self):
  super().__init__('growth54',.75,.5,channel_path=ROOT/'inputs/channel_forecast_2026-10-28_v3.json',anchor='september_2026')
  delays=self.snapshot['assumptions']['completion_delays']
  self.snapshot['assumptions']['completion_sold_rates']=[1-delays[0]]+[.95]*4
  self.snapshot['assumptions']['completion_unsold_rates']=[1-delays[0]]+[.5]*4
 def evaluate(self,guess):
  channels=self.sales(guess);market=self.market(channels.total_sales_thousand_m2.to_numpy())
  raw=c.link.project(self.parameters,market['drivers']);q4=raw.loc['2026'];h=c.link.price;p=self.parameters
  endpoint=float(h.P.loc['2025-12-01'])*(1+TARGET)
  delta=100*np.log(endpoint/q4.price_index.iloc[-1])/3
  state=float(h.R.iloc[-1]-h.effect(p,h.P.index[-1]));level=float(h.P.iloc[-1]);rows=[]
  for date in c.DATES:
   x=float(market['drivers'].loc[date,p['feature']]);direct=p['gamma']*(x-p['x_mean'])
   model_state=p['c']+p['phi']*state+direct
   if date.year==2026:
    # Equal monthly shift relative to the unadjusted Q4 path at the same market.
    # The corrected December state feeds the original recurrence from January.
    model_state=float(raw.loc[date,'real_log_pp']);state=model_state+delta
   else:state=model_state
   inflation=100*np.log(h.C.loc[date]/h.C.loc[date-pd.offsets.MonthBegin(1)])
   level*=np.exp((inflation+state)/100)
   rows.append(dict(month=date,price_index=level,real_log_pp=state,inflation_log_pp=inflation,market_driver=x,market_direct_component_log_pp=direct,expert_shift_log_pp=delta if date.year==2026 else 0.,unadjusted_same_market_state_log_pp=model_state))
  frame=pd.DataFrame(rows).set_index('month');assert np.isclose(frame.loc['2026-12-01','price_index'],endpoint,rtol=1e-12)
  assert np.ptp(frame.loc['2026','expert_shift_log_pp'])<1e-12
  assert (frame.loc['2027':,'expert_shift_log_pp']==0).all()
  return dict(channels=channels,market=market,model_price=frame)
def main():
 OUT.mkdir(exist_ok=True);rows=[];monthly=[];ports=[];checks=[];snapshots=[];traces=[]
 ref=pd.read_csv(ROOT/'reanchored_price_results/annual_equilibrium.csv').set_index('year')
 for case in ['unconstrained','target10']:
  s=TargetSystem()
  if case=='unconstrained':s.evaluate=lambda guess:System.evaluate(s,guess)
  params=copy.deepcopy(s.parameters);sol=s.solve(tolerance=1e-10);ev=sol['evaluation']
  annual=c.link.yearly(pd.DataFrame({'price_index':sol['index']})).join(ev['channels'])
  for col in ['launches_annual','completions_annual','portfolio_closing','unsold_closing','sellout','balance_error','unallocated_sales']:annual[col]=ev['market']['annual'][col]
  assert s.parameters==params
  assert np.allclose(annual.mortgage_transactions_bln_rub,s.annual_channels.mortgage_transactions_bln_rub)
  assert np.allclose(annual.cash_total_thousand_m2,s.annual_channels.cash_total_thousand_m2)
  assert annual.unallocated_sales.max()<1e-7
  if case=='unconstrained':
   for col in ['nominal_growth_pct','total_sales_thousand_m2','unsold_closing']:assert np.allclose(annual[col],ref[col],atol=1e-5)
  else:assert abs(annual.loc[2026,'nominal_growth_pct']-100*TARGET)<1e-8
  for year,r in annual.iterrows():rows.append(dict(scenario=case,year=int(year),**r.to_dict()))
  monthly.append(ev['model_price'].assign(price_thousand_rub_m2=sol['index']*s.scale,scenario=case).rename_axis('month').reset_index())
  ports.append(ev['market']['by_class'].assign(scenario=case));traces.append(sol['trace'].assign(scenario=case))
  snap=copy.deepcopy(s.snapshot);snap['assumptions']['sales']=annual.total_sales_thousand_m2.tolist();snap['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
  snap['forecast_scenario_provenance']={'price_model':'growth54','price_anchor':'september_2026','alpha':.75,'omega':.5,'target_nominal_growth2026':TARGET if case=='target10' else None,'expert_target_only_in_Q4_2026':True,'case':case}
  replay=pd.DataFrame(c.portfolio.forecast(snap));assert np.allclose(replay.unsold_closing,ev['market']['by_class'].unsold_closing)
  snapshots.append(dict(scenario=case,snapshot=snap))
  check=dict(scenario=case,iterations=sol['iterations'],max_price_log_residual=sol['residual'])
  if case=='target10':
   alt=s.solve(damping=.25,start_factor=1.15,tolerance=1e-10);deviation=float(np.max(abs(np.log(alt['index']/sol['index']))));assert deviation<1e-6;check['alternative_start_deviation']=deviation
   correction=float(ev['model_price'].loc['2026-10-01','expert_shift_log_pp'])
  checks.append(check)
 data=pd.DataFrame(rows);data.to_csv(OUT/'annual_comparison.csv',index=False);latest=data[data.scenario=='target10'].set_index('year');latest.to_csv(OUT/'annual_equilibrium.csv')
 pd.concat(monthly,ignore_index=True).to_csv(OUT/'monthly_paths.csv',index=False);pd.concat(ports,ignore_index=True).to_csv(OUT/'portfolio_by_class.csv',index=False);pd.concat(traces,ignore_index=True).to_csv(OUT/'iteration_traces.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'convergence_checks.csv',index=False)
 (OUT/'portfolio_scenarios.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2));(OUT/'portfolio_snapshot_equilibrium.json').write_text(json.dumps(snapshots[-1]['snapshot'],ensure_ascii=False,indent=2))
 (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,user_selected_target_nominal2026=TARGET,monthly_Q4_shift_log_pp=correction,actual_JanSep2026_unchanged=True,price_coefficients_unchanged=True,future_expert_shifts=0,recurrence='Equal shift to three unadjusted Q4 real log growth rates at each market iteration. Corrected December state feeds unmodified AR equation January2027 onward.',fixed='Latest workbook v3 mortgage nominal, cash/oneoff area, CPI, September2026 price anchor, alpha75 omega50 and sold95/unsold50 completion rule unchanged.',caveat='Expert forecast constraint for 2026, not improved statistical fit or revised observed history.'),indent=2))
 print(data[['scenario','year','nominal_growth_pct','price_denominator_thousand_rub_m2','total_sales_thousand_m2','launches_annual','portfolio_closing','unsold_closing','sellout']].round(6).to_string(index=False))
if __name__=='__main__':main()
