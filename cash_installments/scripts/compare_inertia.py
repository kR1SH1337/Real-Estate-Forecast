"""Recursive expanding-history comparison; diagnostic forecasts, not main-model replacement."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cash_model import read_sources, fit_model, metrics

MODELS = ['static_huber', 'static_huber_matched', 'static_ols', 'ar1_huber', 'arx1_huber', 'arx1_ols', 'arx3_huber',
          'delta_arx1_huber', 'mean12', 'last_value']


def regress(y, features, robust=True):
    y=np.asarray(y,float);features=np.asarray(features,float)
    valid=np.isfinite(y)&np.isfinite(features).all(axis=1)
    y,features=y[valid],features[valid]
    if len(y)<features.shape[1]+6: raise ValueError('Insufficient usable observations')
    center=features.mean(axis=0);scale=features.std(axis=0)
    if (scale<1e-10).any():raise ValueError('Constant feature')
    X=np.column_stack([np.ones(len(y)),(features-center)/scale])
    if np.linalg.matrix_rank(X)!=X.shape[1]:raise ValueError('Rank-deficient design')
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    if robust:
        for iterations in range(1000):
            e=y-X@b;s=max(1.482602218505602*np.median(abs(e-np.median(e))),1e-6)
            w=np.minimum(1,1.345/np.maximum(abs(e/s),1e-15));rw=np.sqrt(w)
            candidate=np.linalg.lstsq(X*rw[:,None],y*rw,rcond=None)[0]
            if np.linalg.norm(candidate-b)<1e-9*(1+np.linalg.norm(b)):
                b=candidate;break
            b=candidate
        else:raise RuntimeError('Huber IRLS did not converge')
    coef=np.r_[b[0]-np.sum(b[1:]*center/scale),b[1:]/scale]
    e=y-np.column_stack([np.ones(len(y)),features])@coef
    return {'coef':coef.tolist(),'n':len(y),'train_rmse':float(np.sqrt(np.mean(e**2))),
            'design_condition_standardized':float(np.linalg.cond(X))}


def fit(train, name):
    y=train.real;x=train.ks_avg3
    if name.startswith('static_'):
        if name=='static_huber_matched':y=y.where(y.shift(1).notna())
        m=fit_model(y,x,'ols' if name=='static_ols' else 'huber')
        return {'coef':m['coef'].tolist(),'n':int(m['valid'].sum())}
    if name=='mean12':return {'coef':[float(y.iloc[-12:].mean())],'n':int(y.iloc[-12:].notna().sum())}
    if name=='last_value':return {'coef':[float(y.iloc[-1])],'n':1}
    if name=='ar1_huber':return regress(y,y.shift(1).to_numpy()[:,None])
    if name in ['arx1_huber','arx1_ols']:
        return regress(y,np.column_stack([y.shift(1),x]),robust=name.endswith('huber'))
    if name=='arx3_huber':
        return regress(y,np.column_stack([y.shift(1).rolling(3,min_periods=3).mean(),x]))
    if name=='delta_arx1_huber':
        return regress(y.diff(),np.column_stack([y.diff().shift(1),x.diff()]))
    raise ValueError(name)


def recursive(train, future_x, name, model):
    coef=model['coef'];last=float(train.real.iloc[-1]);last_x=float(train.ks_avg3.iloc[-1])
    last_change=float(train.real.diff().iloc[-1]);out=[]
    recent=train.real.iloc[-3:].to_list()
    for x in np.asarray(future_x,float):
        if name.startswith('static_'):pred=coef[0]+coef[1]*x
        elif name in ['mean12','last_value']:pred=coef[0]
        elif name=='ar1_huber':pred=coef[0]+coef[1]*last
        elif name.startswith('arx1_'):pred=coef[0]+coef[1]*last+coef[2]*x
        elif name=='arx3_huber':pred=coef[0]+coef[1]*np.mean(recent)+coef[2]*x
        else:
            change=coef[0]+coef[1]*last_change+coef[2]*(x-last_x)
            pred=last+change
        out.append(float(pred));last_change=pred-last;last=pred;last_x=x
        recent=(recent+[float(pred)])[-3:]
    return np.asarray(out)


def evaluate(data, origins, start=None):
    rows=[];paths=[]
    for end in origins:
        train=data.iloc[:end] if start is None else data.iloc[:end].loc[start:]
        future=data.iloc[end:end+12]
        if len(future)!=12 or future.real.isna().any():raise ValueError('Incomplete validation horizon')
        for name in MODELS:
            m=fit(train,name);pred=recursive(train,future.ks_avg3,name,m)
            for h in [1,3,6,12]:
                rows.append({'origin':str(train.index[-1].date()),'model':name,'horizon':h,
                    'actual_real':float(future.real.iloc[:h].sum()),'predicted_real':float(pred[:h].sum()),
                    'n_train':m['n'],'coef':json.dumps(m['coef'])})
            for date,actual,forecast in zip(future.index,future.real,pred):
                paths.append({'origin':str(train.index[-1].date()),'date':str(date.date()),'model':name,
                              'actual_real':float(actual),'predicted_real':float(forecast)})
    return pd.DataFrame(rows),pd.DataFrame(paths)


def summarize(rows):
    out={}
    for (name,h),g in rows.groupby(['model','horizon']):
        q=metrics(g.actual_real,g.predicted_real)
        q['rmse_area_mln_m2']=q['rmse']/282.7
        q['underprediction_folds']=int((g.predicted_real<g.actual_real).sum())
        out.setdefault(name,{})[str(h)]=q
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reference_results/inertia')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data,rates=read_sources(ROOT/'data/cash_sales.csv',ROOT/'data/macro_scenario.xlsx',ROOT/'data/price_index.xlsx')
    origins=list(range(24,len(data)-11))
    rows,paths=evaluate(data,origins)
    # Sensitivity 1: exclude the isolated Jan-2023 point for every model.
    continuous,_=evaluate(data,origins,start='2023-03-01')
    # Sensitivity 2: same four dates with only >=12 post-July calendar observations.
    post_origins=[end for end in origins if len(data.iloc[:end].loc['2024-07-01':])>=12]
    post,_=evaluate(data,post_origins,start='2024-07-01')
    # Leakage check: replacing validation targets must not affect a forecast.
    train=data.iloc[:origins[0]];future=data.iloc[origins[0]:origins[0]+12].copy()
    mutated=future.copy();mutated['real']=999999.
    for name in MODELS:
        m=fit(train,name)
        np.testing.assert_allclose(recursive(train,future.ks_avg3,name,m),recursive(train,mutated.ks_avg3,name,m))
        if name=='arx1_huber':
            b=m['coef'];x=future.ks_avg3.to_numpy()
            first=b[0]+b[1]*train.real.iloc[-1]+b[2]*x[0]
            second=b[0]+b[1]*first+b[2]*x[1]
            np.testing.assert_allclose(recursive(train,x,name,m)[:2],[first,second])
    # Verify the simple multi-regressor solver reproduces the established estimator.
    check=regress(data.real,data.ks_avg3.to_numpy()[:,None])
    np.testing.assert_allclose(check['coef'],fit_model(data.real,data.ks_avg3)['coef'],atol=1e-6)
    # Main-model backtest parity with saved calculations.
    old=pd.read_csv(ROOT/'reference_results/backtest.csv')
    old=old[(old.method=='huber')&(old.scope=='expanding')].sort_values('origin')
    baseline=rows[(rows.model=='static_huber')&(rows.horizon==12)].sort_values('origin')
    np.testing.assert_allclose(baseline.predicted_real,old.predicted_real,atol=1e-7)
    np.testing.assert_allclose(baseline.actual_real,old.actual_real,atol=1e-7)
    future_dates=pd.date_range(data.index[-1]+pd.offsets.MonthBegin(1),'2030-12-01',freq='MS')
    future_x=rates.shift(1).rolling(3,min_periods=3).mean().reindex(future_dates)
    if future_x.isna().any():raise ValueError('Missing scenario KS')
    fits={};forecasts=[];monthly=[]
    for name in MODELS:
        m=fit(data,name);fits[name]=m
        pred=recursive(data,future_x,name,m)
        if name.startswith('ar') or name.startswith('delta_'):
            phi=m['coef'][1];m['phi']=phi
            # For the 3-month average, companion roots determine dynamic stability.
            roots=np.roots([1,-phi/3,-phi/3,-phi/3]) if name=='arx3_huber' else np.array([phi])
            m['max_companion_root_modulus']=float(max(abs(roots)))
            m['stable_ar_component']=bool(max(abs(roots))<1)
            if name.startswith('arx') and abs(1-phi)>1e-6:
                m['long_run_intercept']=m['coef'][0]/(1-phi)
                m['long_run_ks_sensitivity']=m['coef'][2]/(1-phi)
        if name.startswith('delta_') and abs(1-m['coef'][1])>1e-6:
            m['steady_monthly_drift_at_constant_ks']=m['coef'][0]/(1-m['coef'][1])
        for date,x,value in zip(future_dates,future_x,pred):
            monthly.append({'date':str(date.date()),'model':name,'ks_avg3':float(x),'real_bln_rub':float(value),
                            'area_mln_m2':float(value)/282.7,'negative_forecast':bool(value<0)})
        fs=pd.Series(pred,index=future_dates)
        for year in range(2026,2031):
            actual=data.loc[data.index.year==year,'real'];forecast=fs.loc[fs.index.year==year]
            total=float(actual.sum()+forecast.sum())
            forecasts.append({'model':name,'year':year,'real_bln_rub':total,'area_mln_m2':total/282.7,
                              'actual_months':len(actual),'forecast_months':len(forecast),
                              'negative_forecast_months':int((forecast<0).sum())})
    summary={'primary':summarize(rows),'continuous_history_sensitivity':summarize(continuous),
             'post_july_sensitivity':summarize(post),'post_july_origins':post.origin.unique().tolist(),
             'full_history_same_post_july_origins':summarize(rows[rows.origin.isin(post.origin.unique())]),
             'fits':fits,'origins':rows.origin.unique().tolist(),
             'checks':['Recursive forecasts do not use validation sales',
                'Huber matrix implementation matches established single-factor estimator',
                'All ten baseline annual folds match saved main-model backtest'],
             'notes':['Future KS is supplied/known; historical vintages unavailable.',
                'Twelve-month horizons overlap; ten folds are not ten independent annual observations.',
                'No automatic clipping, stationarity constraints, model selection or replacement.',
                'All models use expanding history. Missing sales and their lags stay missing.',
                'ARX(1) uses two fewer usable sales observations than the original static model; matched static control is supplied.',
                'Stable AR component does not establish stationary sales when KS is nonstationary; delta model sales levels retain an integration step.',
                'Differenced model has an unrestricted intercept; inspect cumulative drift.',
                'Area is real RUB billion / 282.7; base price RUB282700/m2.'],
             'sources_sha256':{f:hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest()
                               for f in ['cash_sales.csv','price_index.xlsx','macro_scenario.xlsx']}}
    for name,frame in [('backtest.csv',rows),('backtest_monthly.csv',paths),
        ('continuous_sensitivity.csv',continuous),('post_july_sensitivity.csv',post),
        ('forecast_annual.csv',pd.DataFrame(forecasts)),('forecast_monthly.csv',pd.DataFrame(monthly))]:
        frame.to_csv(args.output/name,index=False,encoding='utf-8-sig')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Descriptive paired comparisons; overlapping folds do not justify independent-fold p-values.
    q=rows[rows.horizon.eq(12)]
    actual=q.groupby('origin').actual_real.first()
    errors=q.pivot(index='origin',columns='model',values='predicted_real').sub(actual,axis=0)
    paired={name:{'lower_absolute_error_folds':int((abs(errors[name])<abs(errors.static_huber)).sum()),
                  'n':len(errors),'max_abs_real_error':float(abs(errors[name]).max())} for name in MODELS}
    (args.output/'paired_comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['static_huber','ar1_huber','arx1_huber','arx3_huber','delta_arx1_huber']
    labels=['Текущая модель','AR(1)','ARX: лаг продаж + КС','ARX: средние продажи за 3 мес. + КС','Модель изменений']
    colors=['#193b61','#16877e','#d57b27','#8863a1','#b65252']
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
    values=[summary['primary'][name]['12']['wape_pct'] for name in names]
    axes[0].barh(labels,values,color=colors)
    for i,v in enumerate(values):axes[0].text(v+.4,i,f'{v:.2f}%',va='center',fontsize=9)
    axes[0].invert_yaxis();axes[0].set_xlim(0,max(values)*1.2)
    axes[0].set_title('Ошибка суммы следующих 12 месяцев')
    axes[0].set_xlabel('WAPE, % · 10 пересекающихся горизонтов')
    af=pd.DataFrame(forecasts)
    for name,label,color in zip(names,labels,colors):
        q=af[af.model.eq(name)];axes[1].plot(q.year,q.area_mln_m2,marker='o',color=color,label=label)
    axes[1].axhline(0,color='gray',linewidth=.7);axes[1].set_xticks(range(2026,2031))
    axes[1].set_ylabel('Годовая расчётная площадь, млн м²')
    axes[1].set_title('Условные прогнозы по текущему сценарию КС')
    axes[1].legend(fontsize=8,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.supxlabel('2026: 9 месяцев истории + 3 месяца прогноза. Отрицательные оценки не обрезаны.\nСравнение исследовательское; основная модель сохранена.',fontsize=9)
    fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    for name in MODELS:
        s=summary['primary'][name]['12']
        print(f'{name:20} WAPE={s["wape_pct"]:.2f}% MAPE={s["mape_pct"]:.2f}% RMSE={s["rmse"]:.2f} bias={s["bias_pct"]:+.2f}%')
    print('FITS',json.dumps(fits,ensure_ascii=False))


if __name__=='__main__':main()
