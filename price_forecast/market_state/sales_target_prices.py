"""Inverse annual prices conditional on fixed sales targets and fixed channels.
Does not manufacture a monthly index path or overwrite the endogenous solution.
"""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
import coupled_forecast as coupled

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'sales_target_results';out.mkdir(exist_ok=True)
    system=coupled.CoupledSystem()
    channels=system.annual_channels.copy()
    mortgage_area=channels.reference_sales_thousand_m2-channels.cash_total_thousand_m2
    assert (mortgage_area>0).all()
    prices=1000*channels.mortgage_transactions_bln_rub/mortgage_area
    frame=pd.DataFrame({'target_sales_thousand_m2':channels.reference_sales_thousand_m2,
        'mortgage_thousand_m2':mortgage_area,'cash_and_oneoff_thousand_m2':channels.cash_total_thousand_m2,
        'mortgage_transactions_bln_rub':channels.mortgage_transactions_bln_rub,
        'required_annual_price_thousand_rub_m2':prices})
    prior=system.channels['price_anchor_2025_mean_thousand_rub_m2']
    for year in coupled.YEARS:
        frame.loc[year,'annual_price_change_pct']=100*(prices.loc[year]/prior-1)
        prior=prices.loc[year]
    market=system.market(channels.reference_sales_thousand_m2.to_numpy())
    unforced=coupled.link.project(system.parameters,market['drivers'])
    absolute_unforced=unforced.price_index*system.scale
    for col in ['launches_annual','completions_annual','portfolio_closing','sellout','unsold_closing']:
        frame[col]=market['annual'][col]
    market['by_class'].to_csv(out/'portfolio_by_class.csv',index=False)
    snapshot=copy.deepcopy(system.snapshot)
    snapshot['assumptions']['sales']=channels.reference_sales_thousand_m2.tolist()
    snapshot['assumptions']['sales_2031']=float(channels.reference_sales_thousand_m2.iloc[-1])
    (out/'portfolio_snapshot_sales_targets.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    future_area=mortgage_area.loc[2026]-system.actual['mortgage_thousand_m2']
    future_nominal=channels.loc[2026,'mortgage_transactions_bln_rub']-system.actual['mortgage_transactions_bln_rub']
    assert future_area>0 and future_nominal>0
    quarter_price=1000*future_nominal/future_area
    for year in coupled.YEARS:
        values=absolute_unforced.loc[str(year)]
        model_denominator=float(np.average(values,weights=[d.days_in_month for d in values.index])) if year==2026 else float(values.mean())
        target_denominator=float(quarter_price if year==2026 else prices.loc[year])
        frame.loc[year,'required_conversion_price_thousand_rub_m2']=target_denominator
        frame.loc[year,'unforced_model_conversion_price_thousand_rub_m2']=model_denominator
        frame.loc[year,'required_price_level_adjustment_pct']=100*(target_denominator/model_denominator-1)
    frame.to_csv(out/'annual_required_prices.csv',index_label='year')
    september_observed=system.channels['price_anchor_september_2026_thousand_rub_m2']
    september_proxy=float(coupled.link.price.P.iloc[-1]*system.scale)
    check=dict(actual_mortgage_9m_thousand_m2=system.actual['mortgage_thousand_m2'],
        actual_mortgage_9m_transactions_bln=system.actual['mortgage_transactions_bln_rub'],
        required_q4_mortgage_thousand_m2=float(future_area),q4_transactions_bln=float(future_nominal),
        required_q4_effective_price_thousand_rub_m2=float(quarter_price),
        observed_september_average_price_thousand_rub_m2=september_observed,
        difference_to_observed_september_pct=100*(quarter_price/september_observed-1),
        september_model_proxy_thousand_rub_m2=september_proxy,
        difference_to_model_proxy_pct=100*(quarter_price/september_proxy-1))
    (out/'2026_fact_consistency.json').write_text(json.dumps(check,indent=2))
    assert np.allclose(mortgage_area*prices/1000,channels.mortgage_transactions_bln_rub)
    assert np.allclose(mortgage_area+channels.cash_total_thousand_m2,channels.reference_sales_thousand_m2)
    assert np.allclose(prices,channels.reference_annual_price_thousand_rub_m2)
    diagnostics=dict(as_of='2026-10-27',mode='Sales targets are fixed; annual prices follow the transaction identity',
        target_source='Latest uploaded workbook, reference_sales_thousand_m2',
        mortgage_nominals_fixed=True,cash_oneoff_fixed=True,checks_passed=True,
        annual_price_definition='Transaction-value/area effective annual price required by the mortgage conversion, not automatically a constant-quality December/December index.',
        monthly_index_calibrated=False,
        status='Annual inverse scenario complete; monthly price-index path and 2026 linkage require a decision. Prior endogenous coupled solution remains separate.')
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(frame[['target_sales_thousand_m2','required_annual_price_thousand_rub_m2','annual_price_change_pct']].round(3).to_string())
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()
