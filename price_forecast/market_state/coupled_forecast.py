"""Fixed-point sales/portfolio/price system with fixed estimated coefficients.
Annual mortgage transaction amounts include the down payment. Cash areas stay fixed.
"""
from pathlib import Path
import argparse,copy,json,hashlib
import numpy as np
import pandas as pd
import linked_forecast as link
import portfolio_engine as portfolio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'
ROOT=Path(__file__).resolve().parent
YEARS=list(range(2026,2031))
DATES=pd.date_range('2026-10-01','2030-12-01',freq='MS')


class CoupledSystem:
    def __init__(self,anchor='annual_2025'):
        self.channel_path=ROOT/'inputs/channel_forecast_2026-10-27.json'
        self.channels=json.loads(self.channel_path.read_text())
        self.snapshot=json.loads(link.DEFAULT_SNAPSHOT.read_text())
        self.parameters=json.loads((ROOT/'results/parameters.json').read_text())['mos_lag1']
        self.annual_channels=pd.DataFrame(self.channels['annual']).set_index('year')
        self.anchor=anchor
        if anchor=='annual_2025':
            base_index=np.average(link.price.P.loc['2025'],weights=link.price.Q.loc['2025'])
            self.scale=self.channels['price_anchor_2025_mean_thousand_rub_m2']/base_index
            assert np.isclose(np.average(link.price.P.loc['2025']*self.scale,weights=link.price.Q.loc['2025']),
                              self.channels['price_anchor_2025_mean_thousand_rub_m2'])
        elif anchor=='september_2026':
            self.scale=self.channels['price_anchor_september_2026_thousand_rub_m2']/link.price.P.iloc[-1]
        else:raise ValueError(anchor)
        self.actual=self.channels['actual_2026_9m']
        assert np.isclose(sum(self.snapshot['actual9']),self.actual['mortgage_thousand_m2']+self.actual['cash_thousand_m2'])
        assert np.isclose(link.price.Q.loc['2026'].sum()/1000,sum(self.snapshot['actual9']))

    def sales(self,index):
        absolute=index*self.scale
        rows=[]
        for year,row in self.annual_channels.iterrows():
            if year==2026:
                quarter=absolute.loc['2026-10-01':'2026-12-01']
                denominator=float(np.average(quarter,weights=[d.days_in_month for d in quarter.index]))
                future_nominal=row.mortgage_transactions_bln_rub-self.actual['mortgage_transactions_bln_rub']
                future_area=1000*future_nominal/denominator
                mortgage=self.actual['mortgage_thousand_m2']+future_area
            else:
                denominator=float(absolute.loc[str(year)].mean())
                future_nominal=row.mortgage_transactions_bln_rub
                future_area=mortgage=1000*future_nominal/denominator
            total=mortgage+row.cash_total_thousand_m2
            rows.append(dict(year=int(year),price_denominator_thousand_rub_m2=denominator,
                denominator_period='Q4 only, day weighted' if year==2026 else 'full year, monthly mean',
                mortgage_transactions_bln_rub=float(row.mortgage_transactions_bln_rub),
                future_mortgage_transactions_bln_rub=float(future_nominal),
                future_mortgage_thousand_m2=float(future_area),mortgage_thousand_m2=float(mortgage),
                cash_total_thousand_m2=float(row.cash_total_thousand_m2),total_sales_thousand_m2=float(total)))
        frame=pd.DataFrame(rows).set_index('year')
        assert np.allclose(frame.future_mortgage_thousand_m2*frame.price_denominator_thousand_rub_m2/1000,frame.future_mortgage_transactions_bln_rub)
        return frame

    def market(self,sales):
        data=copy.deepcopy(self.snapshot)
        data['assumptions']['sales']=list(map(float,sales))
        data['assumptions']['sales_2031']=float(sales[-1])
        rows=pd.DataFrame(portfolio.forecast(data))
        if rows.unallocated_sales.max()>1e-7:raise ValueError('Sales could not be allocated to cohorts')
        annual=rows[rows.housing_class=='Всего'].set_index('year').sort_index()
        q,s,drivers,anchors,_,_=link.build_market(annual,data)
        return dict(annual=annual,by_class=rows,q=q,stock=s,drivers=drivers,anchors=anchors)

    def evaluate(self,guess):
        channels=self.sales(guess)
        market=self.market(channels.total_sales_thousand_m2.to_numpy())
        model_price=link.project(self.parameters,market['drivers'])
        return dict(channels=channels,market=market,model_price=model_price)

    def solve(self,damping=.5,tolerance=1e-8,max_iter=150,start_factor=1.):
        # A one-way run based on the latest workbook totals supplies a starting path only.
        initial_market=self.market(self.annual_channels.reference_sales_thousand_m2.to_numpy())
        one_way=link.project(self.parameters,initial_market['drivers'])
        guess=one_way.price_index.copy()*start_factor
        trace=[];previous_sales=None
        for iteration in range(1,max_iter+1):
            evaluated=self.evaluate(guess)
            implied=evaluated['model_price'].price_index
            residual=float(np.max(abs(np.log(implied/guess))))
            sales=evaluated['channels'].total_sales_thousand_m2.to_numpy()
            sales_change=float(np.max(abs(sales/previous_sales-1))) if previous_sales is not None else None
            for year in YEARS:
                mask=guess.index.year==year
                trace.append(dict(iteration=iteration,year=year,max_price_log_residual=residual,
                    max_sales_relative_change=sales_change,
                    year_price_mean_thousand_rub_m2=float((guess.loc[mask]*self.scale).mean()),
                    sales_thousand_m2=float(evaluated['channels'].loc[year,'total_sales_thousand_m2']),
                    unsold_closing_thousand_m2=float(evaluated['market']['annual'].loc[year,'unsold_closing'])))
            if residual<tolerance and (sales_change is None or sales_change<tolerance):
                return dict(index=guess,evaluation=evaluated,trace=pd.DataFrame(trace),iterations=iteration,
                            residual=residual,one_way=one_way,one_way_market=initial_market)
            guess=np.exp((1-damping)*np.log(guess)+damping*np.log(implied))
            previous_sales=sales
        raise RuntimeError(f'No convergence after {max_iter} iterations; residual={residual}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--damping',type=float,default=.5)
    parser.add_argument('--tolerance',type=float,default=1e-8)
    args=parser.parse_args()
    if not 0<args.damping<=1:raise ValueError('Damping must be in (0,1]')
    out=ROOT/'coupled_results';out.mkdir(exist_ok=True)
    system=CoupledSystem();original=json.dumps(system.parameters,sort_keys=True)
    result=system.solve(args.damping,args.tolerance)
    ev=result['evaluation'];annual=link.yearly(pd.DataFrame({'price_index':result['index']}))
    source=system.annual_channels
    annual=annual.join(ev['channels'])
    for column in ['launches_annual','completions_annual','portfolio_closing','sellout','unsold_closing']:
        annual[column]=ev['market']['annual'][column]
    annual['workbook_reference_sales_thousand_m2']=source.reference_sales_thousand_m2
    annual['workbook_reference_price_thousand_rub_m2']=source.reference_annual_price_thousand_rub_m2
    annual['sales_change_vs_workbook_pct']=100*(annual.total_sales_thousand_m2/annual.workbook_reference_sales_thousand_m2-1)
    one_way_annual=link.yearly(result['one_way'])
    annual['one_way_nominal_growth_pct']=one_way_annual.nominal_growth_pct
    annual['one_way_real_growth_pct']=one_way_annual.real_growth_pct
    annual.to_csv(out/'annual_equilibrium.csv',index_label='year')
    result['trace'].to_csv(out/'iteration_trace.csv',index=False)
    ev['market']['by_class'].to_csv(out/'portfolio_by_class.csv',index=False)
    export_snapshot=copy.deepcopy(system.snapshot)
    export_snapshot['assumptions']['sales']=annual.total_sales_thousand_m2.tolist()
    export_snapshot['assumptions']['sales_2031']=float(annual.total_sales_thousand_m2.iloc[-1])
    export_snapshot['coupling_provenance']={'source':'price_forecast/market_state/coupled_forecast.py',
        'status':'Numerically converged conditional price-sales-portfolio scenario',
        'source_excel_sha256':system.channels['source_sha256']}
    (out/'portfolio_snapshot_equilibrium.json').write_text(json.dumps(export_snapshot,ensure_ascii=False,indent=2))
    reproduced=pd.DataFrame(portfolio.forecast(export_snapshot))
    assert np.allclose(reproduced.portfolio_closing,ev['market']['by_class'].portfolio_closing)
    assert np.allclose(reproduced.unsold_closing,ev['market']['by_class'].unsold_closing)
    monthly=pd.DataFrame({'price_index':result['index'],'model_implied_price_index':ev['model_price'].price_index})
    monthly['price_thousand_rub_m2']=monthly.price_index*system.scale
    monthly['sales_m2']=ev['market']['q'].loc[DATES]
    monthly['unsold_begin_m2']=ev['market']['stock'].loc[DATES]
    monthly['months_of_sales_price_driver']=np.exp(ev['market']['drivers'].loc[DATES,'mos_lag1'])
    monthly.to_csv(out/'monthly_equilibrium.csv',index_label='month')
    tests=[]
    # Re-solving from different starting prices/damping tests numerical stability, not accuracy.
    for damping,factor in [(.25,1.),(1.,1.),(.5,.85),(.5,1.15)]:
        other=system.solve(damping,args.tolerance,start_factor=factor)
        deviation=float(np.max(abs(np.log(other['index']/result['index']))))
        assert deviation<1e-6
        tests.append(dict(damping=damping,start_factor=factor,iterations=other['iterations'],max_price_log_deviation_from_main=deviation))
    pd.DataFrame(tests).to_csv(out/'convergence_checks.csv',index=False)
    other_system=CoupledSystem('september_2026')
    other=other_system.solve(args.damping,args.tolerance)
    sensitivity=link.yearly(pd.DataFrame({'price_index':other['index']})).join(other['evaluation']['channels'])
    sensitivity.to_csv(out/'september_price_anchor_sensitivity.csv',index_label='year')
    # Given its own price output, the published sales must satisfy the financial identity.
    checked=system.sales(result['index'])
    assert np.allclose(checked.total_sales_thousand_m2,annual.total_sales_thousand_m2,rtol=1e-12)
    assert np.isfinite(annual.select_dtypes('number')).all().all()
    assert json.dumps(system.parameters,sort_keys=True)==original
    assert np.allclose(annual.cash_total_thousand_m2,source.cash_total_thousand_m2)
    diagnostics=dict(as_of='2026-10-27',converged=True,iterations=result['iterations'],
        max_price_log_residual=result['residual'],tolerance=args.tolerance,damping=args.damping,
        coefficients_refitted=False,price_anchor='2025 annual average unit price from source workbook mapped to the sales-weighted 2025 mean model index',
        price_anchor_thousand_rub_m2=system.channels['price_anchor_2025_mean_thousand_rub_m2'],
        mortgage_nominals_held_fixed_during_iterations=True,cash_and_oneoff_held_fixed_during_iterations=True,actual_9m_preserved=True,
        price_parameter_sha256=hashlib.sha256((ROOT/'results/parameters.json').read_bytes()).hexdigest(),
        future_launches='Recalculated with current-year and next-year sales using the unchanged portfolio engine; terminal 2031 sales equal 2030.',
        numerator='Mortgage transaction value includes down payment. Full annual nominal for 2027+; forecast Q4 nominal only for 2026.',
        denominator='2026 forecast Q4 mean proxy price weighted by calendar days; 2027+ full-year arithmetic mean monthly proxy price.',
        caveats='A numerical fixed point is not evidence of econometric forecast accuracy. Unit-price/index anchoring and annual-to-monthly bridge are assumptions. Earlier dummy dates remain provisional. Mortgage ruble amounts are fixed, with no extra affordability elasticity.',checks_passed=True)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2))
    (out/'parameters_used.json').write_text(json.dumps(system.parameters,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].plot(annual.index,annual.one_way_real_growth_pct,'o--',label='Без обратной связи')
    axs[0].plot(annual.index,annual.real_growth_pct,'o-',label='Согласованный расчёт')
    axs[0].set(ylabel='Реальный рост цен, декабрь к декабрю, %',xticks=YEARS);axs[0].legend(fontsize=8)
    axs[1].plot(annual.index,annual.workbook_reference_sales_thousand_m2/1000,'o--',label='Исходный Excel')
    axs[1].plot(annual.index,annual.total_sales_thousand_m2/1000,'o-',label='Согласованный расчёт')
    axs[1].set(ylabel='Продажи, млн м² за год',xticks=YEARS);axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'coupled_forecast.svg');fig.savefig(out/'coupled_forecast.png',dpi=130);plt.close(fig)
    print(annual[['nominal_growth_pct','real_growth_pct','total_sales_thousand_m2','unsold_closing','sales_change_vs_workbook_pct']].round(3).to_string())
    print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
    print('ALTERNATIVE ANCHOR',sensitivity[['nominal_growth_pct','total_sales_thousand_m2']].round(3).to_string())


if __name__=='__main__':main()
