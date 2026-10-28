"""NLS coefficient inference and diagnostics for the two original price forms.

HAC: prediction Jacobian sandwich, Bartlett lags 6, n/(n-k) correction.
Small-sample t/F references are approximate; model/event selection is not covered.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import real_index_experiment as source

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'statistical_comparison_results'
old=source.old
NAMES=['constant','persistence','early','recent','data','market']


def evaluate(theta,y,lag,d,dl,x):
    c,phi=theta[:2];beta=theta[2:5];gamma=theta[5]
    fitted=c+phi*(lag-dl@beta)+d@beta+gamma*x
    jac=np.c_[np.ones(len(y)),lag-dl@beta,d-phi*dl,x]
    return fitted,y-fitted,jac


def hac_covariance(jac,residual,lags):
    n,k=jac.shape;bread=np.linalg.inv(jac.T@jac);scores=jac*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,lags+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(lags+1))*(cross+cross.T)
    covariance=bread@meat@bread*n/(n-k)
    return (covariance+covariance.T)/2


def coefficient_table(theta,cov,n,k):
    se=np.sqrt(np.diag(cov));t=theta/se;p=2*stats.t.sf(abs(t),n-k);crit=stats.t.ppf(.975,n-k)
    return [dict(parameter=name,estimate=float(b),se=float(s),t_stat=float(tt),p_value=float(pp),ci95_low=float(b-crit*s),ci95_high=float(b+crit*s)) for name,b,s,tt,pp in zip(NAMES,theta,se,t,p)]


def wald(theta,cov,indices,df_resid):
    values=theta[indices];sub=cov[np.ix_(indices,indices)]
    w=float(values@np.linalg.solve(sub,values));q=len(indices)
    return dict(wald=w,f_stat=w/q,df_num=q,df_denom=df_resid,p_value=float(stats.f.sf(w/q,q,df_resid)))


def ljung_box(residual,lags):
    e=residual-residual.mean();n=len(e);den=e@e
    acf=np.array([e[j:]@e[:-j]/den for j in range(1,lags+1)])
    q=float(n*(n+2)*np.sum(acf**2/(n-np.arange(1,lags+1))))
    return dict(statistic=q,df=lags-1,p_value=float(stats.chi2.sf(q,lags-1)))


def write_report(result):
    g,l=result['growth'],result['real_index']
    p=lambda x:'<0,001' if x<.001 else f'{x:.3f}'.replace('.',',')
    labels={'constant':'Константа','persistence':'Инерция','early':'Ранние события','recent':'События с 2025','data':'Аномалии данных','market':'Рыночный показатель'}
    lines=['# Статистическое сравнение двух ценовых функций','',
        '28.10.2026. Общая выборка: январь 2022 — сентябрь 2026, N = 57. В каждой функции шесть параметров, остаточных степеней свободы 51. Это последние переоценки с полным 2022 годом, а не прежние 54-месячные оценки с апреля 2022.','',
        'Первая функция объясняет реальные месячные логарифмические приросты и использует месячные dummy. Вторая объясняет уровень реального индекса (январь 2021 = 100) и накопленные индикаторы тех же событий. Рынок в обеих — log непроданного остатка / средних продаж шести завершённых месяцев с прежним лагом.','',
        '## Подгонка','',
        '| Показатель | Приросты + отношение | Реальный индекс + рынок |','|---|---:|---:|']
    for label,key in [('R²','r2'),('Скорректированный R²','adjusted_r2'),('Описательный R² после вычитания оценённых шоков','r2_shock_adjusted_descriptive'),('RMSE в единицах собственной целевой переменной','rmse_target'),('RMSE fitted месячных реальных приростов, лог. п.п.','common_real_monthly_growth_rmse_log_pp')]:
        lines.append(f'| {label} | {g[key]:.5f} | {l[key]:.5f} |')
    lines+=['','R² и RMSE собственных целей нельзя использовать для прямого ранжирования этих двух функций: объясняемые величины и их единицы различаются. Последняя строка переводит одномесячные fitted-значения уровня в приросты относительно фактического предыдущего индекса; это подгонка, не многомесячный прогноз. Описательный R² очищенного ряда использует вычтенные компоненты, оценённые той же моделью, и не является независимой проверкой.','',
        '## Коэффициенты: Newey–West HAC, 6 месяцев','',
        '| Параметр | Оценка: приросты | SE HAC6 | p-value | Оценка: уровень | SE HAC6 | p-value |','|---|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(g['coefficients']['6'],l['coefficients']['6']):
        lines.append(f"| {labels[a['parameter']]} | {a['estimate']:.5f} | {a['se']:.5f} | {p(a['p_value'])} | {b['estimate']:.5f} | {b['se']:.5f} | {p(b['p_value'])} |")
    lines+=['','p-value двусторонние для H0: соответствующий коэффициент равен нулю. Использована HAC sandwich-ковариация по аналитическому якобиану нелинейной функции прогноза, ядро Bartlett, 6 месячных лагов, поправка N/(N−K); приближённые t-распределение с 51 степенью свободы и F для совместных Wald-тестов. Это не обычная линейная регрессия на независимо оценённом очищенном ряду: шоковые коэффициенты и AR оценивались совместно, в ковариации учтены все шесть параметров.','',
        '**Главный результат:** рынок незначим в обеих функциях. Его 95% HAC-интервал включает как отрицательные, так и положительные значения. Незначимость не доказывает нулевого экономического эффекта, но не даёт статистического основания считать этот регрессор ключевым драйвером прогноза.','',
        '| Модель | 95% HAC6-интервал коэффициента рынка |','|---|---|']
    for name,r in [('Приросты',g),('Реальный индекс',l)]:
        a=r['coefficients']['6'][-1];lines.append(f"| {name} | [{a['ci95_low']:.4f}; {a['ci95_high']:.4f}] |")
    lines+=['','## Что добавляет рынок','',
        '| Показатель | Приросты | Реальный индекс |','|---|---:|---:|']
    for label,key in [('RMSE модели без рынка','rmse_without_market'),('Частный R² рынка: доля ошибки модели без рынка','market_partial_r2'),('Прибавка к общему R² от рынка','market_incremental_r2'),('ΔAIC: модель с рынком минус без рынка','delta_aic_market_vs_no_market'),('ΔBIC: модель с рынком минус без рынка','delta_bic_market_vs_no_market')]:
        lines.append(f'| {label} | {g[key]:.6f} | {l[key]:.6f} |')
    lines+=['','Положительные ΔAIC/ΔBIC означают, что добавление рынка не окупает дополнительный параметр по этим критериям. Каждый Δ вычислен только относительно модели с той же целевой переменной, разметкой и выборкой. AIC/BIC уровня и прироста напрямую не сопоставляются.','',
        '## Чувствительность p-value к ковариации','',
        '| Параметр / функция | IID | HAC0 | HAC3 | HAC6 | HAC12 | HC3 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in [('Приросты',g),('Уровень',l)]:
        for index in [2,3,4,5]:
            values=[p(r['coefficients'][method][index]['p_value']) for method in ['iid','0','3','6','12','HC3']]
            lines.append('| '+name+': '+labels[NAMES[index]]+' | '+' | '.join(values)+' |')
    lines+=['','HC3 корректирует влияние наблюдений с высоким локальным leverage, но сама по себе не корректирует автокорреляцию. HAC и HC3 — разные проверки чувствительности, не методы, среди которых следует выбирать минимальный p-value. У группы аномалий данных всего два отмеченных месяца; её значимость особенно чувствительна. По HAC6 она значима на 5%, по HC3 — нет в обеих функциях. У ранней группы в модели приростов ситуация также меняется. Рыночный коэффициент остаётся незначимым во всех приведённых вариантах.','',
        'Для рыночного коэффициента дополнительно проверена sandwich-ковариация с наблюдаемой гессиановой матрицей вместо Gauss–Newton bread. p-value практически тот же: '+f"{g['market_hac6_observed_hessian_sensitivity']['p_value']:.4f} и {l['market_hac6_observed_hessian_sensitivity']['p_value']:.4f}."+'','',
        '## Остатки и совместные тесты','',
        '| Диагностика | Приросты | Реальный индекс |','|---|---:|---:|']
    lines.append(f"| Durbin–Watson | {g['durbin_watson']:.3f} | {l['durbin_watson']:.3f} |")
    for label,key in [('Ljung–Box, 6 лагов: p','ljung_box_6'),('Ljung–Box, 12 лагов: p','ljung_box_12'),('Koenker–Breusch–Pagan: p','koenker_bp'),('Jarque–Bera: p','jarque_bera'),('Три dummy совместно, HAC6: p','joint_dummy_hac6'),('Три dummy совместно, HC3: p','joint_dummy_hc3'),('Все коэффициенты кроме константы, HAC6: p','joint_slopes_hac6'),('Все коэффициенты кроме константы, HC3: p','joint_slopes_hc3')]:
        lines.append(f'| {label} | {p(g[key]["p_value"])} | {p(l[key]["p_value"])} |')
    lines+=['','В этих приблизительных тестах нет отклонения отсутствия автокорреляции на 6/12 лагах и постоянства дисперсии на 5%; это не доказательство их отсутствия на короткой истории. Нормальность остатков отвергается у обеих функций. У Ljung–Box число степеней свободы равно числу лагов минус один AR-параметр. Koenker BP выполнен по регрессии квадратов остатков на столбцы локального якобиана, 5 степеней свободы.','',
        'Совместная значимость полной модели не означает значимости рынка: результат может определяться инерцией и событиями. Значимость AR в таблице проверяет коэффициент против нуля, **не** единичный корень. Нельзя использовать эти p-value как доказательство стационарности уровня.','',
        '## Границы интерпретации','',
        'Все p-value условны на уже выбранные спецификацию и календарь. Они не учитывают перебор десятков моделей, выбор дат по ретроспективной информации и неопределённость качества данных. HAC не исправляет эндогенность, пропущенные факторы и нестационарность. Коэффициенты dummy — ассоциации размеченных эпизодов, не доказанный причинный эффект льготной ипотеки.','',
        'Таким образом, модель реального индекса лучше подгоняет уровень и имеет значимую инерцию, но в обеих функциях самостоятельный вклад рынка не подтверждён. Высокий R² уровня не решает задачу надёжной оценки ценовой реакции на спрос и предложение.','',
        '## Воспроизведение','',
        '```bash','python price_forecast/market_state/statistical_comparison.py','```','',
        'Зависимости: прежние NumPy, pandas, SciPy. Производные NLS проверены конечными разностями, точки оценки и RMSE воспроизведены из сохранённых параметров. Реализация ковариаций выполнена в Python; Stata в этой среде не запускалась. Для независимой проверки подготовлен [stata_reproduce.do](stata_reproduce.do), совместимый с формульным nl и HAC nwest в Stata 13, запуск из корня репозитория.','',
        '- [Все коэффициенты, SE, p-value и 95% интервалы](coefficients.csv).','- [Все статистики в точности](statistics.json), [данные и лаги](model_data.csv), [настройки и ограничения](diagnostics.json).','',
        'Методические источники: [Stata 13, nl](https://www.stata.com/manuals13/rnl.pdf); [HAC Bartlett/Newey–West](https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_hac.html); [Ljung–Box и model_df](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/diagnostic.py).']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    commands=['version 13.0','clear all','set more off','* Run from the repository root. Not executed in Stata in this environment.',
              'import delimited using "price_forecast/market_state/statistical_comparison_results/model_data.csv", varnames(1) clear',
              'generate tm = monthly(substr(date,1,7), "YM")','format tm %tm','tsset tm','']
    for key,short in [('growth','g'),('real_index','l')]:
        theta=[r['estimate'] for r in result[key]['coefficients']['6']]
        c,ar,early,recent,data,market=theta
        commands+=['* '+key,
            f'nl ({key}_target = {{{short}c={c:.12g}}} + {{{short}ar={ar:.12g}}} * ( ///',
            f'    {key}_lag - {{{short}early={early:.12g}}}*{key}_early_lag ///',
            f'    - {{{short}recent={recent:.12g}}}*{key}_recent_lag - {{{short}data={data:.12g}}}*{key}_data_lag) ///',
            f'    + {{{short}early}}*{key}_early + {{{short}recent}}*{key}_recent + {{{short}data}}*{key}_data ///',
            f'    + {{{short}market={market:.12g}}}*market_centered), hasconstant({short}c) vce(hac nwest 6) iterate(4000)',
            f'estimates store {key}','']
    (OUT/'stata_reproduce.do').write_text('\n'.join(commands)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    sample=pd.concat([old.R.rename('growth'),old.R.shift().rename('lag'),source.REAL.rename('level'),source.REAL.shift().rename('laglevel'),old.X.mos_lag1],axis=1).dropna().loc['2022-01-01':].index
    assert len(sample)==57
    saved_growth=json.loads((ROOT/'direct_sales_growth_results/parameters.json').read_text())['ratio6_all_events']
    saved_level=json.loads((ROOT/'sales_window_from2022_results/parameters.json').read_text())['raw_avg6']
    common_x=old.X.loc[sample,'mos_lag1'];x=(common_x-common_x.mean()).to_numpy()
    output={};all_coefficients=[];data=pd.DataFrame(index=sample)
    data['market_centered']=x
    for kind,saved in [('growth',saved_growth),('real_index',saved_level)]:
        target=old.R if kind=='growth' else source.REAL
        frame=old.F if kind=='growth' else source.SHIFTS
        events=['early','recent','data'];d=frame.loc[sample,events].to_numpy();dl=frame.shift().loc[sample,events].to_numpy()
        y=target.loc[sample].to_numpy();lag=target.shift().loc[sample].to_numpy()
        theta=np.r_[saved['c'],saved['phi'] if kind=='growth' else saved['rho'],[saved['beta'][e] for e in events],saved['coefficients']['ratio6'] if kind=='growth' else saved['gamma']]
        fitted,e,jac=evaluate(theta,y,lag,d,dl,x);n,k=jac.shape
        assert np.linalg.matrix_rank(jac)==k
        assert np.isclose(np.sqrt(np.mean(e**2)),saved['rmse_log_pp'] if kind=='growth' else saved['rmse_target'])
        # Analytic derivative check of the actual nonlinear fitted equation.
        numerical=np.empty_like(jac)
        for j in range(k):
            h=1e-5*max(1.,abs(theta[j]));plus=theta.copy();minus=theta.copy();plus[j]+=h;minus[j]-=h
            numerical[:,j]=(evaluate(plus,y,lag,d,dl,x)[0]-evaluate(minus,y,lag,d,dl,x)[0])/(2*h)
        assert np.allclose(jac,numerical,rtol=1e-6,atol=1e-6)
        sse=float(e@e);tss=float(np.sum((y-y.mean())**2));clean=y-d@theta[2:5]
        iid=sse/(n-k)*np.linalg.inv(jac.T@jac)
        covariance={str(l):hac_covariance(jac,e,l) for l in [0,3,6,12]}
        coeff={str(l):coefficient_table(theta,v,n,k) for l,v in covariance.items()}
        coeff['iid']=coefficient_table(theta,iid,n,k)

        bread=np.linalg.inv(jac.T@jac)
        leverage=np.sum((jac@bread)*jac,axis=1)
        assert leverage.max()<1 and np.isclose(leverage.sum(),k)
        score_hc3=jac*(e/(1-leverage))[:,None]
        hc3=bread@(score_hc3.T@score_hc3)@bread
        coeff['HC3']=coefficient_table(theta,(hc3+hc3.T)/2,n,k)

        for method,rows in coeff.items():
            for row in rows:all_coefficients.append(dict(model=kind,covariance=method,**row))
        # Restricted fit without the market regressor, on exactly the same sample.
        null=old.fit(sample,None) if kind=='growth' else source.fit_level(sample,None,'raw')
        null_rmse=null['rmse'] if kind=='growth' else null['rmse_target'];sse_null=n*null_rmse**2
        bp_design=jac;aux_e2=e*e
        aux_fitted=bp_design@np.linalg.lstsq(bp_design,aux_e2,rcond=None)[0]
        bp_r2=1-np.sum((aux_e2-aux_fitted)**2)/np.sum((aux_e2-aux_e2.mean())**2)
        jb=stats.jarque_bera(e)
        # Optional observed-Hessian bread sensitivity for nonlinear-score inference.
        observed=jac.T@jac
        for j in range(3):
            observed[1,2+j]+=e@dl[:,j];observed[2+j,1]+=e@dl[:,j]
        observed_cov=np.linalg.inv(observed)@(jac.T@jac)@covariance['6']@(jac.T@jac)@np.linalg.inv(observed)
        if kind=='growth':common_growth_fitted=fitted
        else:
            assert (fitted>0).all()
            common_growth_fitted=100*np.log(fitted/source.REAL.shift().loc[sample].to_numpy())
        result=dict(n=n,k=k,df_resid=n-k,rmse_target=float(np.sqrt(sse/n)),mae_target=float(np.mean(abs(e))),r2=1-sse/tss,
            adjusted_r2=1-(sse/(n-k))/(tss/(n-1)),r2_shock_adjusted_descriptive=1-sse/float(np.sum((clean-clean.mean())**2)),
            common_real_monthly_growth_rmse_log_pp=float(np.sqrt(np.mean((common_growth_fitted-old.R.loc[sample].to_numpy())**2))),
            market_partial_r2=1-sse/sse_null,market_incremental_r2=(sse_null-sse)/tss,rmse_without_market=float(null_rmse),
            delta_aic_market_vs_no_market=float(n*np.log(sse/sse_null)+2),delta_bic_market_vs_no_market=float(n*np.log(sse/sse_null)+np.log(n)),
            durbin_watson=float(np.sum(np.diff(e)**2)/sse),ljung_box_6=ljung_box(e,6),ljung_box_12=ljung_box(e,12),
            koenker_bp=dict(statistic=float(n*bp_r2),df=k-1,p_value=float(stats.chi2.sf(n*bp_r2,k-1))),
            jarque_bera=dict(statistic=float(jb.statistic),p_value=float(jb.pvalue)),
            joint_dummy_hac6=wald(theta,covariance['6'],[2,3,4],n-k),joint_slopes_hac6=wald(theta,covariance['6'],[1,2,3,4,5],n-k),
            joint_dummy_hc3=wald(theta,hc3,[2,3,4],n-k),joint_slopes_hc3=wald(theta,hc3,[1,2,3,4,5],n-k),
            coefficients=coeff,market_hac6_observed_hessian_sensitivity=coefficient_table(theta,observed_cov,n,k)[-1],
            max_local_leverage=float(leverage.max()),jacobian_gram_condition_number=float(np.linalg.cond(jac.T@jac)),target='monthly real log growth, pp' if kind=='growth' else 'real price index level, Jan2021=100')
        output[kind]=result
        data[kind+'_target']=y;data[kind+'_lag']=lag;data[kind+'_fitted']=fitted;data[kind+'_residual']=e
        for j,event in enumerate(events):
            data[kind+'_'+event]=d[:,j];data[kind+'_'+event+'_lag']=dl[:,j]
    data.to_csv(OUT/'model_data.csv',index_label='date')
    pd.DataFrame(all_coefficients).to_csv(OUT/'coefficients.csv',index=False)
    (OUT/'statistics.json').write_text(json.dumps(output,indent=2))
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,period='2022-01 through 2026-09',sample_n=57,
        inference='Conditional nonlinear least-squares inference using prediction Jacobian and HAC sandwich; Bartlett weights, six monthly lags, n/(n-k) correction; approximate t/F reference with n-k=51.',
        sensitivity='Classical IID, HAC lag 0/3/6/12 and heteroskedastic HC3 leverage correction. Observed-Hessian bread sensitivity for market coefficient reported separately.',
        model_comparison='Raw/adjusted R2 target-dependent; do not rank growth and level models by R2. AIC/BIC deltas only relative to no-market counterpart with same target.',
        residual_tests='Ljung-Box degrees of freedom lag count minus one AR term, approximate; Koenker auxiliary regression of squared residual on prediction Jacobian, df5; Jarque-Bera approximate normality.',
        limitations='Short sample, sparse/retrospectively chosen event groups, revised data and extensive model selection not accounted for in p-values. Conditional significance, not causal identification. Persistence p tests zero, not a unit root. HAC does not establish stationarity or fix endogenous regressors.',
        implementation='NumPy/SciPy analytic Jacobian and covariance implementation; not estimated by Stata here.',production_model_changed=False),indent=2))
    write_report(output)
    print('COEFFICIENTS HAC6')
    print(pd.DataFrame(all_coefficients).query("covariance=='6'").round(6).to_string(index=False))
    print('SUMMARY')
    for kind,result in output.items():
        print(kind,json.dumps({key:value for key,value in result.items() if key not in ['coefficients']},indent=2))


if __name__=='__main__':main()
