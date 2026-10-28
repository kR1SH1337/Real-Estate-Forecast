"""Future-sales weights at user-fixed alpha=0.75, all four blocks jointly solved."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from launch_speed_equilibrium import System,c

ROOT=Path(__file__).resolve().parent
POLICY=ROOT/'inputs/launch_policy_2026-10-28.json'
OUT=ROOT/'omega_results'


def main():
    OUT.mkdir(exist_ok=True)
    policy=json.loads(POLICY.read_text());alpha=policy['launch_adjustment_speed']
    assert alpha==.75
    rows=[];monthly=[];traces=[];portfolios=[];snapshots=[];checks=[];parameters={}
    archived=pd.read_csv(ROOT/'launch_speed_results/annual_comparison.csv')
    for kind in ['growth54','growth57','level57']:
        for omega in policy['future_sales_weights_tested']:
            system=System(kind,alpha,omega)
            params=copy.deepcopy(system.level if kind=='level57' else system.parameters)
            result=system.solve(tolerance=1e-8);ev=result['evaluation']
            annual=c.link.yearly(pd.DataFrame({'price_index':result['index']})).join(ev['channels'])
            for col in ['launches_annual','completions_annual','portfolio_closing','unsold_closing','sellout']:annual[col]=ev['market']['annual'][col]
            assert params==(system.level if kind=='level57' else system.parameters)
            assert np.allclose(system.sales(result['index']).total_sales_thousand_m2,annual.total_sales_thousand_m2)
            assert np.allclose(annual.cash_total_thousand_m2,system.annual_channels.cash_total_thousand_m2)
            assert np.allclose(annual.mortgage_transactions_bln_rub,system.annual_channels.mortgage_transactions_bln_rub)
            q=annual.total_sales_thousand_m2.to_numpy();expected=[system.snapshot['assumptions']['launches_2026']]
            for i in range(1,5):
                nextq=q[i+1] if i<4 else q[-1]
                expected.append((1-alpha)*expected[-1]+alpha*policy['launch_sales_ratio']*((1-omega)*q[i]+omega*nextq))
            assert np.allclose(expected,annual.launches_annual)
            if omega==.5:
                old=archived[(archived.model==kind)&(archived.launch_speed==alpha)].set_index('year')
                for col in ['nominal_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing']:assert np.allclose(annual[col],old[col],atol=1e-5)
            scenario=f'{kind}_alpha75_omega{int(omega*100)}'
            for year,row in annual.iterrows():rows.append(dict(scenario=scenario,model=kind,alpha=alpha,omega=omega,year=int(year),**row.to_dict()))
            month=pd.DataFrame({'price_index':result['index'],'model_price_index':ev['model_price'].price_index,'sales_m2':ev['market']['q'].loc[c.DATES],
                'unsold_begin_m2':ev['market']['stock'].loc[c.DATES],'months_of_sales_driver':np.exp(ev['market']['drivers'].loc[c.DATES,'mos_lag1'])})
            month.index.name='month';monthly.append(month.reset_index().assign(scenario=scenario))
            traces.append(result['trace'].assign(scenario=scenario));portfolios.append(ev['market']['by_class'].assign(scenario=scenario))
            snapshot=copy.deepcopy(system.snapshot);snapshot['assumptions']['sales']=q.tolist();snapshot['assumptions']['sales_2031']=float(q[-1])
            snapshot['forecast_scenario_provenance']={'price_model':kind,'alpha_selected':alpha,'omega_sensitivity':omega,'four_blocks_solved':True,'status':'Omega not yet selected'}
            reproduced=pd.DataFrame(c.portfolio.forecast(snapshot))
            assert np.allclose(reproduced.unsold_closing,ev['market']['by_class'].unsold_closing)
            snapshots.append(dict(scenario=scenario,snapshot=snapshot));parameters[kind]=params
            check=dict(scenario=scenario,iterations=result['iterations'],max_price_log_residual=result['residual'])
            if omega==1.:
                alternative=system.solve(damping=.25,start_factor=1.15,tolerance=1e-8)
                deviation=float(np.max(abs(np.log(alternative['index']/result['index']))))
                assert deviation<1e-6;check['alternative_initialization_deviation']=deviation
            checks.append(check)
    annual_all=pd.DataFrame(rows)
    for kind in ['growth54','growth57','level57']:
        assert np.ptp(annual_all[(annual_all.model==kind)&(annual_all.year==2026)].nominal_growth_pct)<1e-6
    annual_all.to_csv(OUT/'annual_comparison.csv',index=False)
    pd.concat(monthly,ignore_index=True).to_csv(OUT/'monthly_paths.csv',index=False)
    pd.concat(traces,ignore_index=True).to_csv(OUT/'iteration_traces.csv',index=False)
    pd.concat(portfolios,ignore_index=True).to_csv(OUT/'portfolio_by_class.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'convergence_checks.csv',index=False)
    (OUT/'portfolio_scenarios.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2))
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,alpha=.75,alpha_selected_by_user=True,omega_cases=policy['future_sales_weights_tested'],omega_selected=False,
        formula='L_y=(1-alpha)*L_previous+alpha*1.149*((1-omega)*Q_y+omega*Q_next)',
        fixed='All price coefficients, mortgage nominal amounts, cash areas, CPI, launch/sales ratio, completion delays/profiles, class allocation and price anchor unchanged within each price model.',
        price_models='growth54 original production growth; growth57 Jan2022 re-estimation; level57 Jan2022 real-index level. Results never pooled across specifications.',
        terminal='Q2031=Q2030: omega has no direct effect on desired launches in 2030; earlier launches and endogenous price-sales feedback still matter.',
        implementation='Optional launch_future_sales_weight in local price_forecast/market_state/portfolio_engine.py, default 0.5 preserves archived results. Other portfolio directory not modified.',
        solver='Full future monthly price fixed point with endogenous sales and cohort portfolios. Archived omega0.5 alpha0.75 reproduced; alternative starts at omega1 checked.',
        stock_note='Higher inventory relative to a scenario is distinct from inventory growing over time.',
        caveat='Numerical equilibrium is not statistical validation. Original weak market inference remains.'),indent=2))
    print(annual_all[['model','omega','year','nominal_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing']].round(4).to_string(index=False))


if __name__=='__main__':main()
