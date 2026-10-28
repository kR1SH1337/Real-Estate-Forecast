"""Completion-delay sensitivity with endogenous prices and sales; alpha=.75, omega=.5."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from launch_speed_equilibrium import System,c
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'completion_delay_results'

def main():
 OUT.mkdir(exist_ok=True)
 rows=[];monthly=[];portfolios=[];checks=[];snapshots=[];traces=[]
 reference=pd.read_csv(ROOT/'omega_results/annual_comparison.csv')
 for kind in ['growth54','growth57','level57']:
  for extra in [0.,.1,.2]:
   s=System(kind,.75,.5)
   old=copy.deepcopy(s.snapshot['assumptions']['completion_delays'])
   delays=[old[0]]+[x+extra for x in old[1:]]
   assert all(0<=x<=1 for x in delays)
   s.snapshot['assumptions']['completion_delays']=delays
   params=copy.deepcopy(s.level if kind=='level57' else s.parameters)
   result=s.solve(tolerance=1e-8);ev=result['evaluation']
   annual=c.link.yearly(pd.DataFrame({'price_index':result['index']})).join(ev['channels'])
   for col in ['launches_annual','completions_annual','portfolio_closing','unsold_closing','sellout','unallocated_sales','balance_error']:
    annual[col]=ev['market']['annual'][col]
   bc=ev['market']['by_class'].copy()
   bc['unsold_exiting_at_completion']=bc.completions_period-bc.sold_exiting_at_completion
   annual['unsold_exiting_at_completion']=bc[bc.housing_class=='Всего'].set_index('year').unsold_exiting_at_completion
   assert params==(s.level if kind=='level57' else s.parameters)
   assert np.allclose(annual.mortgage_transactions_bln_rub,s.annual_channels.mortgage_transactions_bln_rub)
   assert np.allclose(annual.cash_total_thousand_m2,s.annual_channels.cash_total_thousand_m2)
   assert annual.unallocated_sales.max()<1e-7
   if extra==0:
    ref=reference[(reference.model==kind)&(reference.omega==.5)].set_index('year')
    for col in ['nominal_growth_pct','total_sales_thousand_m2','unsold_closing']:assert np.allclose(annual[col],ref[col],atol=1e-5)
   scenario=f'{kind}_delay_plus{round(extra*100)}pp'
   for y,r in annual.iterrows():rows.append(dict(scenario=scenario,model=kind,extra_delay_pp=extra*100,year=int(y),**r.to_dict()))
   bc['scenario']=scenario;portfolios.append(bc)
   monthly.append(pd.DataFrame({'price_index':result['index'],'model_price_index':ev['model_price'].price_index,'sales_m2':ev['market']['q'].loc[c.DATES],'unsold_begin_m2':ev['market']['stock'].loc[c.DATES]}).rename_axis('month').reset_index().assign(scenario=scenario))
   traces.append(result['trace'].assign(scenario=scenario))
   snap=copy.deepcopy(s.snapshot);snap['assumptions']['sales']=annual.total_sales_thousand_m2.tolist();snap['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
   snap['forecast_scenario_provenance']={'price_model':kind,'alpha':.75,'omega':.5,'extra_delay_pp_from2027':extra*100,'status':'Sensitivity, not selected forecast'}
   replay=pd.DataFrame(c.portfolio.forecast(snap));assert np.allclose(replay.unsold_closing,bc.unsold_closing)
   snapshots.append(dict(scenario=scenario,snapshot=snap))
   check=dict(scenario=scenario,iterations=result['iterations'],max_price_log_residual=result['residual'],max_unallocated_sales=float(annual.unallocated_sales.max()))
   if extra==.2:
    alt=s.solve(damping=.25,start_factor=1.15,tolerance=1e-8)
    deviation=float(np.max(abs(np.log(alt['index']/result['index']))));assert deviation<1e-6
    check['alternative_start_deviation']=deviation
   checks.append(check)
 result=pd.DataFrame(rows)
 for kind in ['growth54','growth57','level57']:
  for col in ['nominal_growth_pct','total_sales_thousand_m2','unsold_closing']:
   assert np.ptp(result[(result.model==kind)&(result.year==2026)][col])<1e-6
 result.to_csv(OUT/'annual_comparison.csv',index=False)
 pd.concat(monthly,ignore_index=True).to_csv(OUT/'monthly_paths.csv',index=False)
 pd.concat(portfolios,ignore_index=True).to_csv(OUT/'portfolio_by_class.csv',index=False)
 pd.concat(traces,ignore_index=True).to_csv(OUT/'iteration_traces.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'convergence_checks.csv',index=False)
 (OUT/'portfolio_scenarios.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2))
 (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,alpha=.75,omega=.5,base_delays=old,extra_delays_pp_from2027=[0,10,20],year2026_unchanged=True,all_sales_allocated=True,scope='Uniform delay of due sold and unsold cohorts; no selective low-sellout delay estimated. Delayed cohorts re-enter next-year due pool and may be delayed again.',fixed='Mortgage nominal, cash area, CPI, price coefficients, launch rule, class rules, profiles, terminal sales unchanged.',caveat='Sensitivity, not empirical delay estimate or selected baseline. Numerical equilibrium does not validate price coefficients.'),indent=2))
 print(result[['model','extra_delay_pp','year','nominal_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing','completions_annual','unsold_exiting_at_completion']].round(4).to_string(index=False))
if __name__=='__main__':main()
