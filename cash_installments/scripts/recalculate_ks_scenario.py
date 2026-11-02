"""Recalculate the cash model under an uploaded monthly KS scenario.

Only future rates change. Historical rates, sales, prices and fitted coefficients
are preserved. Results are a separate scenario; the baseline is never overwritten.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cash_model import MONTHS_RU, read_sources, fit_model, forecasts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ks',type=Path,default=ROOT/'scenarios/ks_2026-11-06/source.xlsx')
    parser.add_argument('--output',type=Path,default=ROOT/'scenarios/ks_2026-11-06')
    args=parser.parse_args()
    data,old_rates=read_sources(ROOT/'data/cash_sales.csv',ROOT/'data/macro_scenario.xlsx',ROOT/'data/price_index.xlsx')
    model=fit_model(data.real,data.ks_avg3)
    book=openpyxl.load_workbook(args.ks,read_only=True,data_only=True)
    try:
        sheet=book['КС_на дату и среднее']
        if sheet.cell(3,4).value!='сред. за период':raise ValueError('Expected monthly average KS in column D')
        version=sheet.cell(2,2).value
        records=[]
        for row in sheet.iter_rows(min_row=4,max_col=4,values_only=True):
            if row[1] is None:continue
            month,year=str(row[1]).split()
            date=pd.Timestamp(int(year),MONTHS_RU.index(month)+1,1)
            value=row[3]
            if not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Missing cached monthly KS')
            records.append({'date':date,'ks_end_percent':float(row[2]),'ks_month_average_percent':float(value)})
        source=pd.DataFrame(records).set_index('date')
        if source.index.has_duplicates:raise ValueError('Duplicate month in KS file')
        expected=pd.date_range(source.index.min(),source.index.max(),freq='MS')
        if not source.index.equals(expected):raise ValueError('KS source has a calendar gap')
        comparison=book['Сравнение']
        warnings=[]
        for col in range(3,comparison.max_column+1):
            label=comparison.cell(3,col).value
            if not isinstance(label,int) or not 2026<=label<=2030:continue
            monthly=source.loc[source.index.year==label]
            end_summary=float(comparison.cell(5,col).value)
            mean_summary=float(comparison.cell(9,col).value)
            mean_monthly=float(np.average(monthly.ks_month_average_percent,weights=monthly.index.days_in_month))
            end_monthly=float(monthly.ks_end_percent.iloc[-1])
            if abs(end_summary-end_monthly)>1e-6 or abs(mean_summary-mean_monthly)>1e-6:
                warnings.append({'year':label,'summary_year_end':end_summary,'monthly_year_end':end_monthly,
                    'summary_year_average':mean_summary,'monthly_day_weighted_average':mean_monthly,
                    'material':bool(abs(end_summary-end_monthly)>.01 or abs(mean_summary-mean_monthly)>.01),
                    'resolution':'Use detailed monthly column D; do not invent a monthly path from annual endpoints.'})
    finally:book.close()
    cutoff=data.index[-1]
    # Preserve the exact old historical definition of KS; changing month-end history
    # to monthly-average history would be a separate re-estimation exercise.
    new_rates=old_rates.copy()
    relevant=source.loc[(source.index>cutoff)&(source.index<=pd.Timestamp('2030-12-01')),'ks_month_average_percent']
    new_rates.loc[relevant.index]=relevant
    np.testing.assert_allclose(new_rates.loc[:cutoff],old_rates.loc[:cutoff],equal_nan=True)
    horizon=(2030-cutoff.year)*12+12-cutoff.month
    old_monthly,old_annual=forecasts(data,old_rates,model,horizon,282700)
    new_monthly,new_annual=forecasts(data,new_rates,model,horizon,282700)
    disputed_years=[w['year'] for w in warnings if w['material']]
    new_annual['provisional']=new_annual.provisional | new_annual.year.isin(disputed_years)
    new_annual['ks_path_requires_confirmation']=new_annual.year.isin(disputed_years)
    new_monthly['ks_path_requires_confirmation']=new_monthly.index.year.isin(disputed_years)
    saved=pd.read_csv(ROOT/'reference_results/forecast_annual.csv')
    np.testing.assert_allclose(old_annual.real_bln_rub,saved.real_bln_rub,atol=1e-7)
    comparison=old_annual[['year','real_bln_rub','area_mln_m2']].merge(
        new_annual[['year','real_bln_rub','area_mln_m2']],on='year',suffixes=('_old','_new'))
    comparison['change_real_bln_rub']=comparison.real_bln_rub_new-comparison.real_bln_rub_old
    comparison['change_area_thousand_m2']=(comparison.area_mln_m2_new-comparison.area_mln_m2_old)*1000
    comparison['change_pct']=(comparison.real_bln_rub_new/comparison.real_bln_rub_old-1)*100
    # Independent linear-response check, including lag spillover across year boundaries.
    delta_x=new_monthly.ks_avg3-old_monthly.ks_avg3
    for year in comparison.year:
        expected_delta=float(model['coef'][1]*delta_x.loc[delta_x.index.year==year].sum())
        actual_delta=float(comparison.loc[comparison.year.eq(year),'change_real_bln_rub'].iloc[0])
        np.testing.assert_allclose(actual_delta,expected_delta,atol=1e-9)
    audit=pd.DataFrame({'ks_old':old_rates,'ks_new_applied':new_rates})
    audit=audit.loc[source.index.min():'2030-12-01']
    audit['source_month_average']=source.ks_month_average_percent.reindex(audit.index)
    audit['source_month_end']=source.ks_end_percent.reindex(audit.index)
    audit['new_scenario_applied']=audit.index>cutoff
    annual_ks=[]
    for year in range(2026,2031):
        part=audit.loc[audit.index.year==year]
        annual_ks.append({'year':year,'old_ks_monthly_mean':float(part.ks_old.mean()),
                         'new_ks_monthly_mean_applied':float(part.ks_new_applied.mean()),
                         'source_month_average_simple_mean':float(part.source_month_average.mean())})
    result={
        'calculated_on':'2026-11-02','scenario_version_label':str(pd.Timestamp(version).date()),
        'last_sales_actual':str(cutoff.date()),'coefficient_alpha':float(model['coef'][0]),
        'coefficient_beta':float(model['coef'][1]),'usable_sales_observations':int(model['valid'].sum()),
        'scope':'Cash/installments only, excluding mortgage and separate expert deposit inflow.',
        'base_price_rub_m2':282700,
        'source_file_original':'Сравнение КС (ФК 25.09 и 06.11)(1).xlsx',
        'source_sha256':hashlib.sha256(args.ks.read_bytes()).hexdigest(),
        'source_sheet':'КС_на дату и среднее','source_column':'D: сред. за период',
        'source_value_mode':'Saved cached values; external links were not refreshed.',
        'history_policy':'Preserve old KS through September 2026, sales, index and coefficients; replace monthly KS only from October 2026.',
        'baseline_policy':'Previous published cash-model scenario, not the 25.09 summary-only scenario in the uploaded comparison sheet.',
        'source_inconsistencies':warnings,
        'source_history_month_end_vs_average_max_difference_pp':float(
            (source.loc[:cutoff,'ks_month_average_percent']-old_rates.reindex(source.loc[:cutoff].index)).abs().max()),
        'checks':['Published baseline forecast reproduced',
                  'Historical KS unchanged',
                  'Annual forecast changes equal beta times changes in lagged KS sums'],
        'baseline_sources_sha256':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                                   for name in ['cash_sales.csv','macro_scenario.xlsx','price_index.xlsx']}}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,frame,index in [('forecast_annual.csv',new_annual,False),('forecast_monthly.csv',new_monthly,True),
                            ('comparison_annual.csv',comparison,False),('ks_monthly_audit.csv',audit,True),
                            ('ks_annual.csv',pd.DataFrame(annual_ks),False)]:
        frame.to_csv(args.output/name,index=index,encoding='utf-8-sig')
    (args.output/'provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(comparison.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
    print('SOURCE INCONSISTENCIES',json.dumps(warnings,ensure_ascii=False))


if __name__=='__main__':main()
