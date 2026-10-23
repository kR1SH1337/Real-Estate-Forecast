#!/usr/bin/env python3
"""Moscow residential launches: statistical benchmark and expectations scenario.

Run from repository root: python launches/model.py
Requires numpy. All areas are thousand square metres.
Real data are supplied locally with --input-json and are not bundled.
A synthetic demonstration is available in examples/synthetic_input.json.
--input-json accepts an updated snapshot with the same structure.
--adjustment, --anticipation and --ratio change explicit scenario assumptions.
No future realised sales are used for historical one-quarter-ahead validation.
The expectations scenario is NOT an estimated or backtested causal relationship.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = MODEL_ROOT / 'data' / 'snapshot_2026-10-22.json'


def qi(key):
    return int(key[:4]) * 4 + int(key[-1]) - 1


def fit(rows):
    x = np.array([[1.0, r[1], r[2]] for r in rows])
    y = np.array([r[0] for r in rows])
    return np.linalg.lstsq(x, y, rcond=None)[0]


def run(data, adjustment=0.5, anticipation=0.5, ratio=None):
    if not 0 <= adjustment <= 1 or not 0 <= anticipation <= 1:
        raise ValueError('Adjustment and anticipation must be between 0 and 1.')
    q = {qi(k): dict(v) for k, v in data['quarterly'].items()}
    annual_sales = {int(k): float(v) for k, v in data['annual_sales_forecast'].items()}
    actual_launch = {t: v['launch'] for t, v in q.items() if 'launch' in v}
    actual_sales = {t: v['sales'] for t, v in q.items() if 'sales' in v}
    rows = []
    times = []
    for t in sorted(actual_launch):
        if all(i in actual_launch for i in (t-1,t-4,t-5)) and all(i in actual_sales for i in (t-1,t-5)):
            rows.append([actual_launch[t]-actual_launch[t-4],
                         actual_launch[t-1]-actual_launch[t-5],
                         actual_sales[t-1]-actual_sales[t-5]])
            times.append(t)
    coef = fit(rows)
    # Repeat the previous comparison on exactly the original nine test quarters.
    historical_rows = [r for t,r in zip(times,rows) if t <= 2026*4+1]
    historical_times = [t for t in times if t <= 2026*4+1]
    errors = []
    for i in range(8, len(historical_rows)):
        c = fit(historical_rows[:i]); r = historical_rows[i]
        errors.append(float(c @ [1,r[1],r[2]] - r[0]))
    recent_ratios = []
    seasonality = np.zeros(4)
    for year in (2023,2024,2025):
        launches = sum(actual_launch[year*4+j] for j in range(4))
        sales = sum(actual_sales[year*4+j] for j in range(4))
        recent_ratios.append(launches/sales)
        seasonality += np.array([actual_sales[year*4+j]/sales for j in range(4)])/3
    k = float(np.mean(recent_ratios)) if ratio is None else ratio
    if k <= 0:
        raise ValueError('Launches/sales ratio must be positive.')
    sales_path = dict(actual_sales)
    for year,total in sorted(annual_sales.items()):
        future = [year*4+j for j in range(4) if year*4+j not in actual_sales]
        remaining = total - sum(actual_sales.get(year*4+j,0) for j in range(4))
        if future and remaining < 0:
            raise ValueError('Annual sales forecast is below year-to-date actual sales.')
        norm = sum(seasonality[t % 4] for t in future)
        for t in future:
            sales_path[t] = remaining * seasonality[t % 4] / norm
    last_actual = max(actual_launch)
    launch_path = dict(actual_launch)
    invalid_quarters = []
    for t in range(last_actual+1, max(annual_sales)*4+4):
        value = launch_path[t-4] + float(coef @ [1,
            launch_path[t-1]-launch_path[t-5], sales_path[t-1]-sales_path[t-5]])
        if value < 0:
            invalid_quarters.append({'year': t//4, 'quarter': t%4+1, 'raw_value': value})
        launch_path[t] = value
    statistical = {y: sum(launch_path[y*4+j] for j in range(4)) for y in annual_sales}
    first = min(annual_sales)
    if any(r['year'] == first for r in invalid_quarters):
        raise ValueError('Negative launch forecast in the anchor year; review short-term model.')

    def scenario(lam, theta, target_ratio):
        path = {first: statistical[first]}
        for year in sorted(annual_sales):
            if year == first:
                continue
            # Terminal boundary: sales beyond the forecast horizon are held flat.
            next_sales = annual_sales.get(year+1, annual_sales[year])
            expected = (1-theta)*annual_sales[year] + theta*next_sales
            target = target_ratio*expected
            path[year] = (1-lam)*path[year-1] + lam*target
        return path

    central = scenario(adjustment, anticipation, k)
    no_anticipation = scenario(adjustment, 0, k)
    annual = []
    for year in sorted(annual_sales):
        next_sales = annual_sales.get(year+1, annual_sales[year])
        expected = (1-anticipation)*annual_sales[year] + anticipation*next_sales
        annual.append({'year':year,'sales':annual_sales[year],
            'statistical_launches': None if any(r['year']==year for r in invalid_quarters) else statistical[year],
            'raw_statistical_annual_sum':statistical[year],
            'scenario_without_anticipation':no_anticipation[year],
            'expected_sales':expected,'target_launches':k*expected,
            'forward_scenario_launches':central[year],
            'anticipation_effect':central[year]-no_anticipation[year]})
    sensitivity = {
        'adjustment': {str(lam): scenario(lam,anticipation,k) for lam in (0.25,0.5,0.75)},
        'ratio': {str(round(v,5)): scenario(adjustment,anticipation,v)
                  for v in (min(recent_ratios),k,max(recent_ratios))}}
    fitted_quarterly = [
        {'quarter': f'{t//4}Q{t%4+1}',
         'actual': actual_launch[t],
         'fitted': actual_launch[t-4] + float(coef @ [1,r[1],r[2]])}
        for t,r in zip(times,rows)]
    return {'units':data['units'],'coefficients':coef.tolist(),
        'coefficient_order':['intercept','lag_launch_yoy_difference','lag_sales_yoy_difference'],
        'historical_validation': {'n':len(errors),'mae':float(np.mean(np.abs(errors)))},
        'assumptions':{'annual_adjustment':adjustment,'next_year_sales_weight':anticipation,
            'target_launches_sales_ratio':k,'ratio_calibration_years':[2023,2024,2025],
            'historical_annual_ratios':recent_ratios,
            'terminal_sales_assumption':'2031 sales equal 2030 sales',
            'quarterly_sales_allocation':'Average quarterly sales shares, 2023-2025; actuals retained',
            '2026_anchor':'Nine-month launches actual + statistical Q4 forecast',
            'status':'Expert scenario; forward mechanism not independently backtested',
            'inventory_balance':'Not included; requires unsold exits at completion and consistent scope'},
        'annual':annual,'sensitivity':sensitivity,'invalid_statistical_quarters':invalid_quarters,
        'fitted_quarterly': fitted_quarterly,
        'short_term_forecast_quarterly': [
            {'quarter': f'{t//4}Q{t%4+1}', 'launch': launch_path[t]}
            for t in range(last_actual+1, first*4+4)],
        'source_notes':data['source_notes']}


def create_chart(data, result, output):
    """Plot full-sample quarterly fits and the separate annual forward scenario."""
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def date(key):
        return datetime(int(key[:4]), 1 + 3*(int(key[-1])-1), 1)

    actual = sorted((k,v['launch']) for k,v in data['quarterly'].items() if 'launch' in v)
    fitted = result['fitted_quarterly']
    forecast = result['short_term_forecast_quarterly']
    annual = result['annual']
    synthetic = any('synthetic' in str(s).lower() for s in data['source_notes'])
    navy, teal, orange = '#193b61', '#16877e', '#d57b27'
    fig, axes = plt.subplots(1,2,figsize=(14.5,6.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.25,wspace=.24)
    title = 'Синтетический пример · не прогноз рынка' if synthetic else 'Москва · запуски новых проектов'
    fig.suptitle(title,x=.065,y=.96,ha='left',fontsize=19,fontweight='bold',color=navy)
    for ax in axes:
        ax.set_facecolor('#f5f7fa')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.tick_params(labelsize=10)
    ax=axes[0]
    ax.plot([date(k) for k,v in actual],[v/1000 for k,v in actual],
            color=navy,lw=2,label='Фактические запуски')
    ax.plot([date(r['quarter']) for r in fitted],[r['fitted']/1000 for r in fitted],
            color=teal,lw=1.8,ls='--',label='Оценки на обучающей истории')
    if forecast:
        keys=[actual[-1][0]]+[r['quarter'] for r in forecast]
        values=[actual[-1][1]]+[r['launch'] for r in forecast]
        ax.plot([date(k) for k in keys],[v/1000 for v in values],color=orange,
                lw=2.4,marker='o',ms=4,label='Краткосрочный прогноз')
        ax.axvspan(date(actual[-1][0]),date(forecast[-1]['quarter']),color=orange,alpha=.08)
    ax.set_title('Кварталы: факт и статистическая модель',loc='left',fontsize=12,pad=14)
    ax.set_ylabel('Млн м² за квартал')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    all_quarterly = ([v/1000 for k,v in actual] + [r['fitted']/1000 for r in fitted]
                     + [r['launch']/1000 for r in forecast])
    ax.set_ylim(min(0,min(all_quarterly)*1.12),max(all_quarterly)*1.16)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)

    ax=axes[1]
    yearly={}
    for key,value in actual:
        yearly.setdefault(int(key[:4]),[]).append(value)
    complete=[(year,sum(values)/1000) for year,values in sorted(yearly.items()) if len(values)==4]
    ax.plot([y for y,v in complete],[v for y,v in complete],color=navy,lw=2,
            marker='o',ms=4,label='Фактические годовые запуски')
    years=[r['year'] for r in annual]
    values=[r['forward_scenario_launches']/1000 for r in annual]
    bridge=complete[-1:] if complete and complete[-1][0]==years[0]-1 else []
    ax.plot([y for y,v in bridge]+years,[v for y,v in bridge]+values,color=orange,
            lw=2.4,marker='o',ms=5,label='2026: дооценка; далее сценарий')
    ax.axvspan(years[0]-.5,years[-1]+.3,color=orange,alpha=.08)
    for year,value in zip(years,values):
        ax.annotate(f'{value:.2f}'.replace('.',','),(year,value),xytext=(0,11),
                    textcoords='offset points',ha='center',fontsize=10,color=orange)
    ax.set_xlim(min(y for y,v in complete)-.25 if complete else years[0]-.5,years[-1]+.4)
    ax.set_xticks(range(min(y for y,v in complete) if complete else years[0],years[-1]+1,2))
    ax.set_title('Годы: факт и сценарий восстановления',loc='left',fontsize=12,pad=14)
    ax.set_ylabel('Млн м² за год')
    all_annual = [v for y,v in complete] + values
    ax.set_ylim(min(0,min(all_annual)*1.12),max(all_annual)*1.16)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)
    footer=('Слева: оценки по коэффициентам всей истории, не ретропрогноз. '
            'Справа: с 2027 года — экспертная подстройка к ожидаемым продажам.\n'
            f'Параметры: λ={result["assumptions"]["annual_adjustment"]:g}; '
            f'θ={result["assumptions"]["next_year_sales_weight"]:g}; '
            f'k={result["assumptions"]["target_launches_sales_ratio"]:.3f}. '
            'Баланс непроданного предложения пока не включён.')
    fig.text(.065,.035,footer,ha='left',va='bottom',fontsize=9,color='#526171',linespacing=1.7)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,facecolor='white')
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json', type=Path)
    parser.add_argument("--output-json", type=Path, default=MODEL_ROOT / "results" / "forecast.json")
    parser.add_argument('--adjustment', type=float, default=0.5)
    parser.add_argument('--anticipation', type=float, default=0.5)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--chart', type=Path, help='PNG chart path; defaults to output JSON stem + _chart.png')
    parser.add_argument('--no-chart', action='store_true', help='Skip chart generation')
    args = parser.parse_args()
    input_path = args.input_json or DEFAULT_INPUT
    if not input_path.is_file():
        parser.error("Supply --input-json with your local snapshot. For a synthetic demonstration use launches/examples/synthetic_input.json from the repository root.")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    result = run(data,args.adjustment,args.anticipation,args.ratio)
    text = json.dumps(result,ensure_ascii=False,indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n",encoding="utf-8")
    if not args.no_chart:
        chart = args.chart or args.output_json.with_name(args.output_json.stem + '_chart.png')
        create_chart(data,result,chart)
        print(f'Chart: {chart}',file=sys.stderr)
    print(text)


if __name__ == '__main__':
    main()
