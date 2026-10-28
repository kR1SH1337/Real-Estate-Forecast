"""Historical diagnostic for the proposed annual sales-stock scenario form.
Coefficients are initialized from monthly OLS, not expert-calibrated or fitted to
four annual observations. This is not validation of unspecified expert values.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import no_price_lag_experiment as helper

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'scenario_function_history_results'


def main():
    OUT.mkdir(exist_ok=True)
    core=helper.core;old=core.old
    m=helper.fit_noar(core.SAMPLE,dict(events=[],controls=['sales_now','stock_lag1']))
    a=12*m['c']/100;bq=12*m['coefficients']['sales_now']/100;bu=-12*m['coefficients']['stock_lag1']/100
    qref=12*np.exp(m['means']['sales_now']);uref=np.exp(m['means']['stock_lag1'])
    fitted=pd.Series(m['c'],index=old.P.index)
    for col,coef in m['coefficients'].items():fitted+=coef*(core.X[col]-m['means'][col])
    original_growth=pd.read_csv(core.OUT/'historical_annual_comparison.csv')
    original_level=pd.read_csv(ROOT/'sales_window_from2022_results/historical_annual_comparison.csv')
    rows=[]
    for year in range(2022,2026):
        dates=old.P.loc[str(year)].index
        q=old.Q.loc[dates];u=old.S.shift().loc[dates]
        assert len(q)==12 and q.notna().all() and u.notna().all() and fitted.loc[dates].notna().all()
        inflation=float(old.C.loc[f'{year}-12-01']/old.C.loc[f'{year-1}-12-01'])
        actual=float(100*(old.P.loc[f'{year}-12-01']/old.P.loc[f'{year-1}-12-01']-1))
        realannual=a+bq*np.log(float(q.sum())/qref)-bu*np.log(float(u.mean())/uref)
        realmonthly=float(fitted.loc[dates].sum()/100)
        # With geometric monthly inputs the converted annual equation is exact.
        geometric=a+bq*np.log(float(12*np.exp(np.log(q).mean()))/qref)-bu*np.log(float(np.exp(np.log(u).mean()))/uref)
        assert np.isclose(geometric,realmonthly,atol=1e-10)
        rows.append(dict(year=year,actual_growth_pct=actual,
            original_growth_ratio_pct=float(original_growth[(original_growth.model=='ratio6_all_events')&(original_growth.year==year)].fitted_nominal_growth_pct.iloc[0]),
            original_real_index_pct=float(original_level[(original_level.model=='raw_avg6')&(original_level.year==year)].fitted_accumulated_growth_pct.iloc[0]),
            new_monthly_ols_pct=float(100*(inflation*np.exp(realmonthly)-1)),
            proposed_annual_initialization_pct=float(100*(inflation*np.exp(realannual)-1)),
            annual_sales_mln_m2=float(q.sum()/1e6),mean_lagged_unsold_mln_m2=float(u.mean()/1e6),inflation_pct=float(100*(inflation-1))))
    errors={}
    for field in ['original_growth_ratio_pct','original_real_index_pct','new_monthly_ols_pct','proposed_annual_initialization_pct']:
        e=np.array([r[field]-r['actual_growth_pct'] for r in rows])
        errors[field]=dict(historical_annual_mae_pp=float(np.mean(abs(e))),historical_annual_rmse_pp=float(np.sqrt(np.mean(e*e))))
    result=dict(comparison=rows,monthly_ols_parameters=m,initial_annual_parameters=dict(a=a,b_Q=bq,b_U=bu,Q_reference_m2=qref,U_reference_m2=uref),historical_descriptive_errors=errors,
        interpretation='The annual scenario equation has not received expert calibration. These are starting coefficients converted from unrestricted monthly OLS, not a final forecast or a validation of expert elasticities.',
        aggregation='Monthly geometric inputs give exact aggregation. Proposed annual inputs are total sales and arithmetic mean monthly lagged stock; this diagnostic therefore differs from compounding the monthly fitted values.',
        checks_passed=True)
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# Сценарная функция и исходные модели на истории','',
        '28.10.2026. У предложенной калибруемой функции ещё нет выбранных экспертных коэффициентов. Ниже проверена её форма со стартовыми коэффициентами из месячной истории. Это не оценка качества ещё не заданной экспертной калибровки.','',
        'Период оценки коэффициентов: январь 2022 — сентябрь 2026, 57 месяцев. Новые коэффициенты получены МНК по месячным реальным логарифмическим приростам без AR и без всех dummy; отдельные регрессоры — log текущих продаж и log остатка с прежним лагом. Они не оценивались на четырёх годовых наблюдениях.','',
        '## Сравнение накопленного номинального прироста декабрь к декабрю','',
        '| Год | Факт | Исходные приросты + U/продажи + dummy | Реальный индекс + рынок + dummy | Новая месячная форма без AR/dummy | Предложенная годовая форма: стартовая инициализация |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        fields=['actual_growth_pct','original_growth_ratio_pct','original_real_index_pct','new_monthly_ols_pct','proposed_annual_initialization_pct']
        lines+=['| '+str(r['year'])+' | '+' | '.join(f'{r[f]:.2f}%' for f in fields)+' |']
    lines+=['','Это описание истории при известных исторических входах, а не годовые прогнозы из предыдущего декабря. Исходные варианты используют фактическое предыдущее наблюдение и восстановленные шоковые компоненты; новые формы не имеют dummy. Для исходных моделей накоплены одномесячные оценки со сложным процентом.','',
        '## Что именно сделано с годовой функцией','',
        'Форма: `s_y = a + b_Q log(Q_y/Q*) − b_U log(Ubar_y/U*)`, где s — годовой логарифмический реальный прирост, Q — годовые продажи, Ubar — средний из месячных остатков с прежним лагом. Номинальный прирост: `(1 + CPI_growth) × exp(s_y) − 1`.','',
        f'Стартовые значения: a = {a:.6f}, b_Q = {bq:.6f}, b_U = {bu:.6f}; Q* = {qref/1e6:.3f} млн м² в год, U* = {uref/1e6:.3f} млн м². Эти значения не объявляются экспертно обоснованными.',
        '',
        'Преобразование из месячной оценки: a = 12c/100, b_Q = 12β_Q/100, b_U = −12β_U/100, Q* = 12 exp(mean log q), U* = exp(mean log U). Для геометрических средних месячных входов такая агрегация точна. При обычном годовом объёме продаж и арифметическом среднем остатка появляется отличие из-за неравномерности месяцев; поэтому обе колонки показаны отдельно.','',
        '## Насколько совпадает с историей','',
        '| Модель | Средняя абсолютная годовая ошибка, п.п. |','|---|---:|']
    for name,key in [('Исходные приросты с отношением','original_growth_ratio_pct'),('Реальный индекс + рынок','original_real_index_pct'),('Новая месячная форма без AR/dummy','new_monthly_ols_pct'),('Предложенная годовая форма: стартовые коэффициенты','proposed_annual_initialization_pct')]:
        lines+=['| '+name+f" | {errors[key]['historical_annual_mae_pp']:.2f} |"]
    lines+=['','Эти ошибки рассчитаны на четырёх годах с полновыборочными коэффициентами и не являются независимой прогнозной точностью. Новая форма не стала точнее сама по себе. Её возможное преимущество для рабочего сценария — прозрачная задаваемая реакция на спрос и предложение; улучшение исторической точности из этого не следует.','',
        '## Почему особенно трудно объяснить 2025 год','',
        '| Год | Продажи, млн м² | Средний остаток с лагом, млн м² | Фактический рост цен |','|---|---:|---:|---:|']
    for r in rows:lines+=['| '+str(r['year'])+f" | {r['annual_sales_mln_m2']:.3f} | {r['mean_lagged_unsold_mln_m2']:.3f} | {r['actual_growth_pct']:.2f}% |"]
    lines+=['','Между 2024 и 2025 продажи немного снизились, остаток снизился умеренно, но цены существенно ускорились при более низкой инфляции. Одна пара агрегированных количественных факторов не объясняет такой переход при текущих оценённых коэффициентах. Это не доказывает истинности прежних dummy и не исключает влияния кредитных условий, состава сделок и других факторов.','',
        'Сценарий без новых ипотечных шоков не обязан воспроизводить все реализовавшиеся шоковые эпизоды. Но ненаблюдаемый контрфактический ряд нельзя объявлять известным фактом. При дальнейшей экспертной калибровке её источники и чувствительность должны быть раскрыты отдельно. Основной совместный прогноз не изменён.','',
        '## Воспроизведение','',
        '```bash','python price_forecast/market_state/scenario_function_history.py','```','',
        'Проверены полнота 12 месяцев в каждом году и точное совпадение месячной агрегации с годовой формой при геометрических входах. Все точные значения, источники стартовых коэффициентов и описательные ошибки находятся в [comparison.json](comparison.json).']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(errors,indent=2))


if __name__=='__main__':main()
