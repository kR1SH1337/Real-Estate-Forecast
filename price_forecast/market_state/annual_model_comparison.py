"""All saved price specifications: compound fitted monthly changes by calendar year.

Historical fit and genuine saved December-origin backtests are separate tables.
No partial-year actual is labelled December/December.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import model as market
import real_index_experiment as real
import nominal_index_experiment as nominal

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
OUT=ROOT/'annual_model_comparison'
YEARS=[2022,2023,2024,2025]
P,C,R=market.P,market.C,market.R
INFL=100*np.log(C/C.shift()).reindex(P.index)
NOM=100*np.log(P/P.shift())
ROWS=[];MONTHLY={}


def load(path):return json.loads((BASE/path).read_text())


def register(key,label,family,fitted,source,n=None,aliases=None):
    fitted=fitted.reindex(P.index)
    row=dict(id=key,model=label,family=family,n=n,source=source,aliases=aliases or [])
    for year in YEARS:
        z=fitted.loc[str(year)]
        row[str(year)]=float(100*np.expm1(z.sum()/100)) if len(z)==12 and z.notna().all() else None
    assert all(row[str(y)] is not None for y in [2023,2024,2025]),key
    row['mean_absolute_historical_error_pp']=float(np.mean([abs(row[str(y)]-100*(P.loc[f'{y}-12-01']/P.loc[f'{y-1}-12-01']-1)) for y in [2023,2024,2025]]))
    ROWS.append(row);MONTHLY[key]=fitted


def growth_fit(m,effect=None,control=None,real_target=True,dates=None):
    target=R if real_target else NOM
    effect=pd.Series(0.,index=P.index) if effect is None else effect
    control=pd.Series(0.,index=P.index) if control is None else control
    fit=m['c']+m.get('phi',m.get('rho'))*(target.shift()-effect.shift())+effect+control
    if real_target:fit+=INFL
    if dates is not None:fit=fit.where(fit.index.isin(dates))
    return fit


def effect_from(m,frame):
    return sum((v*frame[k] for k,v in m.get('beta',{}).items()),start=pd.Series(0.,index=P.index))


def main():
    OUT.mkdir(exist_ok=True);ROWS.clear();MONTHLY.clear()
    # Earlier statistical family, including all saved guidance weights.
    guided=pd.read_csv(BASE/'statistical/results/weight_sensitivity.csv')
    for _,z in guided.iterrows():
        w=float(z.weight);m={'c':(1-z.phi)*z.mu_log_pp_month,'phi':z.phi}
        register(f'guide_{w:g}',f'Реальные приросты: вес ориентиров {w:g}'+(' (выбранный)' if z.selected else ''),'Мягкие ориентиры',growth_fit(m),'statistical/results/weight_sensitivity.csv',67)
    # Calendar, Huber, regime and seasonality before expert wave identification.
    import full_history
    import expert_wave
    import importlib.util
    spec=importlib.util.spec_from_file_location('historical_calendar',BASE/'shock_normalization/model.py')
    calendar=importlib.util.module_from_spec(spec);spec.loader.exec_module(calendar)
    for m in load('shock_normalization/results/parameters.json'):
        name=m['model'];ef=calendar.effect(m,P.index)
        labels={'baseline':'AR без очистки','huber':'AR без очистки, Хьюбер','calendar':'Календарные окна','calendar_huber':'Календарные окна, Хьюбер','regime':'Смена режима','calendar_regime':'Календарь + режим','calendar_season':'Календарь + сезонность'}
        register('calendar_'+name,labels[name],'Первоначальные спецификации',growth_fit(m,ef),'shock_normalization/results/parameters.json',m['n'])
    for name,m in load('shock_normalization/expert_wave_results/parameters.json').items():
        ef=pd.Series(0.,index=P.index)
        if name=='expert_wave':ef=m['beta']*pd.Series(((P.index>=pd.Timestamp(m['start']))&(P.index<=pd.Timestamp(m['end']))).astype(float),index=P.index)
        register('wave_'+name,'AR без очистки' if name=='baseline' else 'Одна dummy: сентябрь 2025 — январь 2026','Экспертная волна',growth_fit(m,ef),'shock_normalization/expert_wave_results/parameters.json',m['n'])
    # Full-history shock specifications and standardized ridge sensitivities.
    labels={'unadjusted':'Без очистки','confirmed_2025_wave':'Только волна 2025–2026','all_user_confirmed':'Все подтверждённые эпизоды с 2025','provisional_full_history':'Все эпизоды, одна dummy','provisional_full_history_two_coefficients':'Волна и прочие эпизоды: две dummy','provisional_full_history_by_era':'До 2025 / с 2025: две dummy','provisional_full_history_by_era_plus_data':'Две эпохи + аномалии данных: три dummy'}
    era=pd.DataFrame({'pre_2025':P.index.isin(full_history.EARLY).astype(float),'from_2025':P.index.isin(full_history.WAVE+full_history.LATEST).astype(float)},index=P.index)
    for name,m in load('shock_normalization/full_history_results/parameters.json').items():
        if name=='confirmed_2025_wave':months=full_history.WAVE
        elif name=='all_user_confirmed':months=full_history.WAVE+full_history.LATEST
        else:months=full_history.SCENARIOS['provisional_full_history']
        ff=era.copy();ff['shock']=P.index.isin(months).astype(float);ff['long_wave']=P.index.isin(full_history.WAVE).astype(float);ff['other_episodes']=P.index.isin([d for d in months if d not in full_history.WAVE]).astype(float);ff['sales_data_2025']=P.index.isin(full_history.DATA_ANOMALIES).astype(float)
        register('full_'+name,labels[name],'Очистка всей истории',growth_fit(m,effect_from(m,ff)),'shock_normalization/full_history_results/parameters.json',m['n'])
    for name,m in load('shock_normalization/robustness_results/parameters.json').items():
        label={'ols':'OLS, две dummy','huber':'Хьюбер, две dummy'}.get(name,'Ridge-type λ='+name.replace('ridge_',''))
        register('robust_'+name,label,'Робастность',growth_fit(m,effect_from(m,era)),'shock_normalization/robustness_results/parameters.json',m['n'])
    # All market regressors and seasonal variants, original matching sample.
    names={'baseline':'Без рынка','mos_lag1':'Остаток / продажи, лаг 1','mos_lag3':'Остаток / продажи, лаг 3','sales_yoy_lag1':'Продажи год к году','sales_momentum_lag1':'Импульс продаж за 3 месяца'}
    for name,m in load('market_state/results/parameters.json').items():
        ef=effect_from(m,market.F);control=pd.Series(0.,index=P.index)
        if m['feature']:control+=m['gamma']*(market.X[m['feature']]-m['x_mean'])
        if m['season']:control+=m['season_beta'][0]*np.sin(2*np.pi*P.index.month/12)+m['season_beta'][1]*np.cos(2*np.pi*P.index.month/12)
        aliases=[]
        if name=='mos_lag1':aliases=['linked_forecast: одинаковая историческая функция','coupled_forecast: одинаковая историческая функция','calibrated_equilibrium: поправки действуют только с октября 2026']
        register('market_'+name,names[name.replace('_season','')]+(' + сезонность' if m['season'] else ''),'Реальные приросты, три dummy',growth_fit(m,ef,control,dates=market.COMMON),'market_state/results/parameters.json',m['n'],aliases)
    # Real-index levels: convert a one-step fitted level to its monthly change.
    for name,m in load('market_state/real_index_results/parameters.json').items():
        if name.startswith('growth'):continue  # identical to registered market variants
        target=real.LOGREAL if m['transform']=='log' else real.REAL
        ef=effect_from(m,real.SHIFTS)
        y=m['c']+m['rho']*(target.shift()-ef.shift())+ef
        if m['feature']:y+=m['gamma']*(market.X[m['feature']]-m['x_mean'])
        realfit=100*np.exp(y/100) if m['transform']=='log' else y
        pf=P.iloc[0]*realfit/100*C.reindex(P.index)/C.loc[P.index[0]]
        fit=100*np.log(pf/P.shift())
        label=('Логарифм реального индекса' if m['transform']=='log' else 'Реальный индекс')+(' + рынок' if m['feature'] else ', без рынка')
        register('real_'+name,label,'Уровень реального индекса',fit.where(fit.index.isin(market.COMMON)),'market_state/real_index_results/parameters.json',m['n'])
    # Window experiments: retain old estimates and add the common full-2022 refit.
    market.X['mos_avg3_lag1']=np.log(market.S.shift(1)/market.Q.rolling(3).mean().shift(1))
    for folder,nobs,include_six in [('sales_window_results',54,False),('sales_window_from2022_results',57,True)]:
        for name,m in load('market_state/'+folder+'/parameters.json').items():
            if name.endswith('avg6') and not include_six:continue
            target=real.LOGREAL if m['transform']=='log' else real.REAL
            ef=effect_from(m,real.SHIFTS)
            y=m['c']+m['rho']*(target.shift()-ef.shift())+ef+m['gamma']*(market.X[m['feature']]-m['x_mean'])
            rf=100*np.exp(y/100) if m['transform']=='log' else y
            pf=P.iloc[0]*rf/100*C.reindex(P.index)/C.loc[P.index[0]]
            fitted=100*np.log(pf/P.shift())
            fitted=fitted.where(fitted.index>=pd.Timestamp('2022-01-01')) if include_six else fitted.where(fitted.index.isin(market.COMMON))
            label=('Логарифм реального индекса' if m['transform']=='log' else 'Реальный индекс')+' + рынок, среднее '+('6' if name.endswith('avg6') else '3')+' мес.'+(', оценка с января 2022' if include_six else ', оценка с апреля 2022')
            register('window_'+folder+'_'+name,label,'Окно продаж и полный 2022 год',fitted,'market_state/'+folder+'/parameters.json',nobs)

    # Nominal ablations include all four earlier nominal-level models as aliases.
    nomlabels={'growth':'Номинальные приросты','log':'Логарифм номинального индекса','raw':'Номинальный индекс'}
    for name,m in load('market_state/nominal_ablation_results/parameters.json').items():
        kind=m['kind'];target=nominal.TARGETS['log'] if kind=='log' else (P if kind=='raw' else NOM)
        frame=market.F if kind=='growth' else nominal.SHIFTS
        ef=effect_from(m,frame)
        y=m['c']+m['rho']*(target.shift()-ef.shift())+ef
        if m['market']:y+=m['gamma']*(market.X['mos_lag1']-m['mean'])
        if kind=='growth':fit=y
        else:
            pf=P.iloc[0]*np.exp(y/100) if kind=='log' else y
            fit=100*np.log(pf/P.shift())
        label=nomlabels[kind]+(' + рынок' if m['market'] else ', без рынка')+(' + dummy' if m['events'] else ', без dummy')
        aliases=['nominal_index_experiment: та же спецификация'] if kind!='growth' and m['events'] else []
        register('nom_'+name,label,'Номинальные модели',fit.where(fit.index.isin(market.COMMON)),'market_state/nominal_ablation_results/parameters.json',54,aliases)
    pulse_history=ROOT/'sales_pulse_results/historical_fit.csv'
    if pulse_history.exists():
        ph=pd.read_csv(pulse_history,parse_dates=['month'])
        plabels={'pulse_now_all_events':'Текущий всплеск + все dummy + рынок','pulse_now_data_only':'Текущий всплеск + рынок, без ипотечных dummy','pulse_lag_all_events':'Всплеск с лагом + все dummy + рынок','pulse_lag_data_only':'Всплеск с лагом + рынок, без ипотечных dummy','pulse_now_no_market':'Текущий всплеск, без рынка и ипотечных dummy','cumulative_pulse_all_events':'Накопленный всплеск + все dummy + рынок','cumulative_pulse_data_only':'Накопленный всплеск + рынок, без ипотечных dummy'}
        for name,group in ph.groupby('model',sort=False):
            if name=='baseline':continue
            group=group.set_index('month')
            pf=P.iloc[0]*group.fitted_real_index/100*C.reindex(group.index)/C.loc[P.index[0]]
            fit=100*np.log(pf/P.shift().reindex(group.index))
            register('pulse_'+name,plabels[name],'Всплески продаж',fit,'market_state/sales_pulse_results/parameters.json',57)

    # Reference rules evaluated with full matching sample means, as historical fits.
    register('nominal_rw','Неизменный номинальный индекс','Простые ориентиры',pd.Series(0.,index=P.index),'nominal_index_experiment.py',54)
    register('inflation_only','Рост только на ИПЦ','Простые ориентиры',INFL,'statistical/model.py',54)
    register('nominal_drift','Средний номинальный прирост','Простые ориентиры',pd.Series(NOM.loc[market.COMMON].mean(),index=P.index),'nominal_index_experiment.py',54)
    register('real_drift','Средний реальный прирост + ИПЦ','Простые ориентиры',INFL+R.loc[market.COMMON].mean(),'real_index_experiment.py',54)
    # Saved genuine 12-month forecasts with a December origin, no refitting on future data.
    backtests=[]
    sources=['statistical/results/baseline_backtest_folds.csv','shock_normalization/results/backtest_folds.csv','market_state/results/backtest_folds.csv','market_state/real_index_results/backtest_folds.csv','market_state/nominal_index_results/backtest_folds.csv','market_state/nominal_ablation_results/backtest_folds.csv','market_state/sales_window_results/backtest_folds.csv','market_state/sales_window_from2022_results/backtest_folds.csv','market_state/sales_pulse_results/backtest_folds.csv']
    for source in sources:
        df=pd.read_csv(BASE/source)
        for _,row in df.iterrows():
            origin=pd.Timestamp(row.origin)
            if origin.month!=12 or ('horizon' in df and row.horizon!=12):continue
            end=origin+pd.DateOffset(years=1)
            if end>P.index[-1]:continue
            actual=100*(P.loc[end]/P.loc[origin]-1)
            estimate=float(row.forecast_growth_pct) if 'forecast_growth_pct' in df else actual+float(row.error_pp)
            backtests.append(dict(source=source,model=row.model,year=end.year,actual_growth_pct=float(actual),forecast_growth_pct=estimate,error_pp=estimate-actual))
    # Verify saved historical fits match reconstructed month changes.
    saved=pd.read_csv(ROOT/'nominal_index_results/historical_fit.csv',parse_dates=['month'])
    for previous,key in [('nominal_index_market','nom_raw_market1_events1'),('nominal_log_no_market','nom_log_market0_events1')]:
        a=saved[saved.model==previous].set_index('month')
        assert np.allclose(MONTHLY[key].loc[a.index],a.fitted_nominal_log_growth,atol=1e-6)
    actual={str(y):float(100*(P.loc[f'{y}-12-01']/P.loc[f'{y-1}-12-01']-1)) for y in YEARS}
    for y in YEARS:assert np.isclose(actual[str(y)],100*np.expm1(NOM.loc[str(y)].sum()/100))
    duplicates=[]
    for i,row in enumerate(ROWS):
        for previous in ROWS[:i]:
            ds=P.index[(P.index.year>=2023)&(P.index.year<=2025)]
            if np.allclose(MONTHLY[row['id']].loc[ds],MONTHLY[previous['id']].loc[ds],atol=1e-6,rtol=0.):
                row['same_historical_curve_as']=previous['id'];duplicates.append([row['id'],previous['id']]);break
    result=dict(method='Compound nominal one-step fitted monthly growth, each month conditioned on actual preceding data. Full-sample parameters, not an out-of-sample annual forecast.',formula='100*(exp(sum(fitted_monthly_log_pp)/100)-1)',years=YEARS,actual=actual,models=ROWS,saved_december_backtests=backtests,duplicates=duplicates,
                not_separate_historical_models=['Original mechanical scenario: forecast starts October 2026, no fitted history','Linked, coupled and future-kappa scenarios share market_mos_lag1 historical equation','Sales-target inversion and price corridors have no independent historical price equation','Sensitivity windows, omitted years and alternative initial values are checks, not additional main model families'],checks_passed=True)
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# Все модели цен: накопленный прирост декабрь к декабрю','',
           'Дата: 27.10.2026. Полные годы: 2022–2025. Для старых вариантов с оценкой с апреля 2022 вместо неполного 2022 показан прочерк. Последние варианты переоценены с января 2022 (57 месяцев). Все значения — номинальный прирост индекса, %. Факт рассчитан по отношению двух декабрьских значений. 2026 не включён: декабрь ещё не наблюдался.','',
           '**Первая таблица — описание истории.** Месячные оценки рассчитаны с фактическим предыдущим наблюдением и накоплены по формуле `100 × (exp(Σ fitted_log_pp / 100) − 1)`. Коэффициенты оценены на всей доступной истории каждой спецификации. Это соответствует накоплению месячных оценок, а не отношению двух отдельных fitted-значений декабря. Для моделей уровней сначала вычисляется fitted-изменение от фактического предыдущего месяца.','',
           'Шоки и фактический рыночный показатель подставлены в исторические оценки. У реальных моделей используется фактический ИПЦ. Эта таблица не является проверкой годового прогноза и не доказывает причинных эффектов. Разные обучающие выборки сохранены как в исходных экспериментах: 67, 57 или 54 месяца.','',
           '| Модель / вариант | N | 2022 | 2023 | 2024 | 2025 |','|---|---:|---:|---:|---:|---:|',
           '| **ФАКТ** | — | '+' | '.join(f'**{actual[str(y)]:.2f}%**' for y in YEARS)+' |']
    family=None
    for row in ROWS:
        if row['family']!=family:
            family=row['family'];lines+=['| **'+family+'** | | | | | |']
        label=row['model']+(' [повтор]' if 'same_historical_curve_as' in row else '')
        lines+=['| '+label+' | '+str(row['n'])+' | '+' | '.join((f"{row[str(y)]:.2f}%" if row[str(y)] is not None else '—') for y in YEARS)+' |']
    lines+=['','Повторы оставлены, чтобы были видны все ранее сохранённые спецификации; они не являются независимыми моделями. Параметры, источники, точные значения и связи повторов доступны в `comparison.json`.','',
            '## Сохранённые прогнозы из предыдущего декабря','',
            'Здесь только ранее сохранённые прогнозы, рассчитанные по прошлой истории. Модели с мягкими ориентирами, Хьюбер/ридж без сохранённого такого backtest не получают придуманной исторической прогнозной оценки. У реальных моделей фактический будущий ИПЦ известен условно. Разметка событий ретроспективна.','',
            '| Источник / модель | Год | Прогноз | Факт | Ошибка, п.п. |','|---|---:|---:|---:|---:|']
    for row in backtests:lines+=['| '+row['source'].split('/')[0]+'/'+row['model']+' | '+str(row['year'])+f" | {row['forecast_growth_pct']:.2f}% | {row['actual_growth_pct']:.2f}% | {row['error_pp']:+.2f} |"]
    lines+=['','## Что не образует отдельную историческую строку','']+[f'- {z}' for z in result['not_separate_historical_models']]
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(n_models=len(ROWS),n_saved_december_backtests=len(backtests),actual=actual,checks_passed=True),ensure_ascii=False))
    selected=['guide_0','guide_0.3','full_provisional_full_history_by_era_plus_data','robust_huber','robust_ridge_0.1','market_baseline','market_mos_lag1','market_mos_lag3','market_sales_yoy_lag1','market_mos_lag1_season','real_real_index_market','real_log_index_market','nom_raw_market1_events1','nom_log_market1_events1','nom_growth_market0_events1','nom_growth_market1_events1','nom_growth_market0_events0','nominal_drift']
    for key in selected:
        row=next(r for r in ROWS if r['id']==key)
        print(row['model'],*[(round(row[str(y)],2) if row[str(y)] is not None else None) for y in YEARS])


if __name__=='__main__':main()
