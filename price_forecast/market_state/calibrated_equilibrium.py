"""Scenario calibration with a four-block equilibrium solved at every trial.
Only a disclosed autonomous future price-growth correction is calibrated.
"""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar,brentq
import coupled_forecast as core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'

ROOT=Path(__file__).resolve().parent
YEARS=core.YEARS


class CalibratedSystem(core.CoupledSystem):
    def __init__(self,kappa=0.,anchor='annual_2025'):
        super().__init__(anchor)
        self.estimated_parameters=copy.deepcopy(self.parameters)
        self.kappa=float(kappa)
        # This changes the scenario's future equation, not the fitted historical file.
        self.parameters['c']=self.estimated_parameters['c']+self.kappa


def summary(solution):
    evaluation=solution['evaluation']
    annual=core.link.yearly(pd.DataFrame({'price_index':solution['index']}))
    annual=annual.join(evaluation['channels'])
    for column in ['launches_annual','completions_annual','portfolio_closing','sellout','unsold_closing']:
        annual[column]=evaluation['market']['annual'][column]
    return annual


def main():
    out=ROOT/'calibrated_results';out.mkdir(exist_ok=True)
    cache={};outer=[]
    def solved(kappa):
        key=float(kappa)
        if key not in cache:
            system=CalibratedSystem(key)
            result=system.solve(tolerance=1e-8)
            cache[key]=(system,result,summary(result))
        return cache[key]
    baseline_system,baseline_result,baseline=solved(0.)
    targets=baseline_system.annual_channels.reference_sales_thousand_m2
    # 2026 is excluded: observed nine months and the conversion-price issue are preserved.
    fit_years=[2027,2028,2029,2030]
    def objective(kappa):
        _,result,annual=solved(kappa)
        error=np.log(annual.loc[fit_years,'total_sales_thousand_m2']/targets.loc[fit_years])
        score=float(np.mean(error**2))
        outer.append(dict(kappa_log_pp_month=float(kappa),mean_squared_log_sales_error=score,
                          rms_log_sales_error_pct=100*np.sqrt(score),inner_iterations=result['iterations']))
        return score
    fit=minimize_scalar(objective,bounds=(-.5,0.),method='bounded',options={'xatol':1e-6,'maxiter':60})
    assert fit.success
    sales_kappa=float(fit.x)
    def premium_gap(kappa):
        _,_,annual=solved(kappa)
        return float(annual.loc[2030,'nominal_growth_pct']-annual.loc[2030,'inflation_pct']-1.8)
    premium_kappa=float(brentq(premium_gap,-.5,0.,xtol=1e-8))
    scenarios={'baseline':0.,'sales_guidance_fit':sales_kappa,'premium_1_8pp_2030':premium_kappa}
    results=[];parameters={};diagnostics={}
    for name,kappa in scenarios.items():
        system,result,annual=solved(kappa)
        parameters[name]={'kappa_log_pp_month':kappa,'estimated_parameters':system.estimated_parameters,
                          'effective_future_parameters':system.parameters}
        for year,row in annual.iterrows():
            record=dict(scenario=name,year=int(year),kappa_log_pp_month=kappa,**row.to_dict())
            record['target_sales_thousand_m2']=float(targets.loc[year])
            record['sales_gap_to_target_pct']=100*(row.total_sales_thousand_m2/targets.loc[year]-1)
            record['nominal_premium_pp']=row.nominal_growth_pct-row.inflation_pct
            for col in ['nominal_growth_pct','real_growth_pct','total_sales_thousand_m2','launches_annual','completions_annual','portfolio_closing','unsold_closing']:
                record['change_vs_baseline_'+col]=row[col]-baseline.loc[year,col]
            results.append(record)
        diagnostics[name]={'iterations':result['iterations'],'max_price_log_residual':result['residual']}
        result['trace'].to_csv(out/f'{name}_inner_trace.csv',index=False)
        market=result['evaluation']['market']
        market['by_class'].to_csv(out/f'{name}_portfolio_by_class.csv',index=False)
        snapshot=copy.deepcopy(system.snapshot)
        snapshot['assumptions']['sales']=annual.total_sales_thousand_m2.tolist()
        snapshot['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
        snapshot['calibration_provenance']={'scenario':name,'kappa_log_pp_month':kappa,'all_four_blocks_solved':True}
        (out/f'{name}_portfolio_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
        month=result['evaluation']['model_price'].copy()
        month['solved_price_index']=result['index']
        month['sales_m2']=market['q'].loc[core.DATES]
        month['unsold_begin_m2']=market['stock'].loc[core.DATES]
        month.to_csv(out/f'{name}_monthly.csv',index_label='month')
        # Four-block consistency and fixed-exogenous-input checks.
        checked=system.sales(result['index'])
        assert np.allclose(checked.total_sales_thousand_m2,annual.total_sales_thousand_m2,rtol=1e-12)
        assert np.allclose(annual.cash_total_thousand_m2,system.annual_channels.cash_total_thousand_m2)
        assert np.allclose(annual.mortgage_transactions_bln_rub,system.annual_channels.mortgage_transactions_bln_rub)
        assert abs(system.parameters['gamma']-system.estimated_parameters['gamma'])<1e-15
        assert system.parameters['beta']==system.estimated_parameters['beta']
        assert abs(system.parameters['phi']-system.estimated_parameters['phi'])<1e-15
        assert np.max(abs(np.log(result['evaluation']['model_price'].price_index/result['index'])))<1e-8
        reproduced=pd.DataFrame(core.portfolio.forecast(snapshot))
        assert np.allclose(reproduced.unsold_closing,market['by_class'].unsold_closing)
        if name!='baseline':
            alt=system.solve(damping=.25,tolerance=1e-8,start_factor=1.15)
            assert np.max(abs(np.log(alt['index']/result['index'])))<1e-6
            diagnostics[name]['alternative_start_iterations']=alt['iterations']
    comparison=pd.DataFrame(results)
    comparison.to_csv(out/'annual_comparison.csv',index=False)
    comparison[comparison.scenario=='sales_guidance_fit'].to_csv(out/'annual_sales_guidance.csv',index=False)
    comparison[comparison.scenario=='premium_1_8pp_2030'].to_csv(out/'annual_premium_1_8.csv',index=False)
    pd.DataFrame(outer).to_csv(out/'outer_calibration_trace.csv',index=False)
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2))
    # Publish a response curve; these points are sensitivity, not alternative estimates of data.
    grid=[]
    for kappa in [0.,-.05,-.1,-.15,-.2,-.25,-.3,-.35]:
        _,result,annual=solved(kappa)
        for year,row in annual.iterrows():grid.append(dict(kappa=kappa,year=int(year),**row.to_dict()))
    pd.DataFrame(grid).to_csv(out/'kappa_response_grid.csv',index=False)
    _,_,selected=solved(sales_kappa)
    assert (selected.nominal_growth_pct<baseline.nominal_growth_pct).all()
    assert (selected.total_sales_thousand_m2>baseline.total_sales_thousand_m2).all()
    assert (selected.loc[2027:,'launches_annual']>baseline.loc[2027:,'launches_annual']).all()
    assert selected.loc[2030,'unsold_closing']>baseline.loc[2030,'unsold_closing']
    assert np.isclose(comparison.query("scenario=='premium_1_8pp_2030' and year==2030").nominal_premium_pp.iloc[0],1.8,atol=1e-6)
    diagnostics.update({'as_of':'2026-10-27','checks_passed':True,'bounds_kappa':[-.5,0.],
        'primary_scenario':'sales_guidance_fit','outer_target_years':fit_years,
        'outer_loss':'Mean squared log ratio of equilibrium sales to workbook guidance, equal weight by year',
        'calibrated_parameter':'One constant future autonomous real-price-growth correction, applied from October 2026; no historical coefficient is re-estimated.',
        'alternative':'One correction solving 2030 nominal growth minus inflation =1.8 percentage points; not real growth of 1.8 percent.',
        '2026_note':'Excluded from calibration objective; actual nine months preserved and forecast quarter recomputed.',
        'interpretation':'Explicit scenario shift, not an empirically identified causal effect or improved forecast accuracy. Target sales not fixed inside the four-block solver.',
        'stock_note':'2026 launches are fixed. Later stock response depends on cohorts, sales, unsold exits and corrections; k>1 does not prove launches grow faster in percentage terms.'})
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    labels={'baseline':'Без поправки','sales_guidance_fit':'Ориентиры продаж','premium_1_8pp_2030':'Премия 1,8 п.п. в 2030'}
    for col,ax,title,div in [('real_growth_pct',axs[0,0],'Реальный рост цен, %',1.),
        ('total_sales_thousand_m2',axs[0,1],'Продажи, млн м²',1000.),
        ('launches_annual',axs[1,0],'Запуски, млн м²',1000.),
        ('unsold_closing',axs[1,1],'Непроданный остаток, млн м²',1000.)]:
        for scenario,label in labels.items():
            a=comparison[comparison.scenario==scenario]
            ax.plot(a.year,a[col]/div,marker='o',label=label)
        ax.set(title=title,xticks=YEARS);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'calibration.svg');fig.savefig(out/'calibration.png',dpi=130);plt.close(fig)
    print('KAPPAS',scenarios)
    print(comparison[['scenario','year','nominal_growth_pct','real_growth_pct','total_sales_thousand_m2','launches_annual','unsold_closing','sales_gap_to_target_pct']].round(3).to_string(index=False))


if __name__=='__main__':main()
