"""Latest mortgage and one-off inputs with selective completion; all blocks jointly solved."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from launch_speed_equilibrium import System,c
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revised_mortgage_results'

def main():
 OUT.mkdir(exist_ok=True)
 rows=[];monthly=[];portfolios=[];checks=[];snapshots=[];traces=[]
 reference=pd.read_csv(ROOT/'updated_channels_results/annual_comparison.csv')
 old_channels=ROOT/'inputs/channel_forecast_2026-10-28.json'
 new_channels=ROOT/'inputs/channel_forecast_2026-10-28_v3.json'
 old_payload=json.loads(old_channels.read_text());new_payload=json.loads(new_channels.read_text())
 for name,fields in [('mortgage_only',['mortgage_transactions_bln_rub']),('oneoff_only',['oneoff_real_bln_rub','cash_total_thousand_m2'])]:
  partial=copy.deepcopy(old_payload)
  partial['scenario_override']={'case':name,'updated_fields':fields,'updated_source_sha256':new_payload['source_sha256'],'note':'Other fields retained from previous inputs; reference prices/sales serve only as solver seed.'}
  for o,n in zip(partial['annual'],new_payload['annual']):
   for field in fields:o[field]=n[field]
  (OUT/(name+'_inputs.json')).write_text(json.dumps(partial,ensure_ascii=False,indent=2))
 for kind in ['growth54','growth57','level57']:
  for case in (['previous','updated','mortgage_only','oneoff_only'] if kind=='growth54' else ['previous','updated']):
   path=old_channels if case=='previous' else new_channels if case=='updated' else OUT/(case+'_inputs.json')
   selective=True
   s=System(kind,.75,.5,channel_path=path)
   old=copy.deepcopy(s.snapshot['assumptions']['completion_delays'])
   if selective:
    s.snapshot['assumptions']['completion_sold_rates']=[1-old[0]]+[.95]*4
    s.snapshot['assumptions']['completion_unsold_rates']=[1-old[0]]+[.5]*4
   params=copy.deepcopy(s.level if kind=='level57' else s.parameters)
   result=s.solve(tolerance=1e-8);ev=result['evaluation']
   annual=c.link.yearly(pd.DataFrame({'price_index':result['index']})).join(ev['channels'])
   for col in ['launches_annual','completions_annual','portfolio_closing','unsold_closing','sellout','unallocated_sales','balance_error']:
    annual[col]=ev['market']['annual'][col]
   bc=ev['market']['by_class'].copy()
   bc['unsold_exiting_at_completion']=bc.completions_period-bc.sold_exiting_at_completion
   annual['unsold_exiting_at_completion']=bc[bc.housing_class=='Всего'].set_index('year').unsold_exiting_at_completion
   for col in ['scheduled_sold','scheduled_unsold','delayed_sold','delayed_unsold']:
    annual[col]=bc[bc.housing_class=='Всего'].set_index('year')[col]
   assert np.allclose(bc.scheduled_sold,bc.sold_exiting_at_completion+bc.delayed_sold)
   assert np.allclose(bc.scheduled_unsold,bc.unsold_exiting_at_completion+bc.delayed_unsold)
   if selective:
    future=bc[bc.year>=2027]
    assert np.allclose(future.sold_exiting_at_completion,.95*future.scheduled_sold)
    assert np.allclose(future.unsold_exiting_at_completion,.5*future.scheduled_unsold)
   assert params==(s.level if kind=='level57' else s.parameters)
   assert np.allclose(annual.mortgage_transactions_bln_rub,s.annual_channels.mortgage_transactions_bln_rub)
   assert np.allclose(annual.cash_total_thousand_m2,s.annual_channels.cash_total_thousand_m2)
   assert annual.unallocated_sales.max()<1e-7
   if case=='previous':
    ref=reference[(reference.model==kind)&(reference.input_case=='updated')].set_index('year')
    for col in ['nominal_growth_pct','total_sales_thousand_m2','unsold_closing']:assert np.allclose(annual[col],ref[col],atol=1e-5)
   scenario=f'{kind}_{case}'
   for y,r in annual.iterrows():rows.append(dict(scenario=scenario,model=kind,input_case=case,year=int(y),**r.to_dict()))
   bc['scenario']=scenario;portfolios.append(bc)
   monthly.append(pd.DataFrame({'price_index':result['index'],'model_price_index':ev['model_price'].price_index,'sales_m2':ev['market']['q'].loc[c.DATES],'unsold_begin_m2':ev['market']['stock'].loc[c.DATES]}).rename_axis('month').reset_index().assign(scenario=scenario))
   traces.append(result['trace'].assign(scenario=scenario))
   snap=copy.deepcopy(s.snapshot);snap['assumptions']['sales']=annual.total_sales_thousand_m2.tolist();snap['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
   snap['forecast_scenario_provenance']={'price_model':kind,'alpha':.75,'omega':.5,'selective_completion':True,'channel_input_case':case,'channel_input_file':str(path.relative_to(ROOT)),'status':'Sensitivity, not selected forecast'}
   replay=pd.DataFrame(c.portfolio.forecast(snap));assert np.allclose(replay.unsold_closing,bc.unsold_closing)
   snapshots.append(dict(scenario=scenario,snapshot=snap))
   check=dict(scenario=scenario,iterations=result['iterations'],max_price_log_residual=result['residual'],max_unallocated_sales=float(annual.unallocated_sales.max()))
   if case=='updated':
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
 latest=result[(result.model=='growth54')&(result.input_case=='updated')].set_index('year')
 latest.to_csv(OUT/'annual_equilibrium.csv',index_label='year')
 selected=next(x for x in snapshots if x['scenario']=='growth54_updated')
 (OUT/'portfolio_snapshot_equilibrium.json').write_text(json.dumps(selected['snapshot'],ensure_ascii=False,indent=2))
 (OUT/'portfolio_scenarios.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2))
 (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,alpha=.75,omega=.5,base_delays=old,completion_sold_rates_from2027=.95,completion_unsold_rates_from2027=.5,year2026_unchanged=True,all_sales_allocated=True,input_file='inputs/channel_forecast_2026-10-28_v3.json',scope='Sold/unsold status measured after current-year sales allocation. Complete 95% of scheduled sold and 50% of scheduled unsold from 2027. Remaining areas re-enter next-year pool; no completed-unsold stock estimated. This is an aggregate approximation, not project-level completion decisions.',fixed='CPI, price coefficients, launch rule, completion rule 95% sold/50% unsold, actual9m2026, price anchor, class rules, profiles and terminal Q2031=Q2030 unchanged. Mortgage nominal and one-off/cash areas updated from saved XLSX annual cells.',caveat='Sensitivity, not empirical delay estimate or selected baseline. Numerical equilibrium does not validate price coefficients.'),indent=2))
 print(result[['model','input_case','year','nominal_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing','completions_annual','unsold_exiting_at_completion']].round(4).to_string(index=False))
if __name__=='__main__':main()
