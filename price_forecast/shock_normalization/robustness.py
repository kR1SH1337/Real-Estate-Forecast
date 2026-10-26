"""Huber and ridge-type sensitivity of the current two-dummy model.
No parameter or penalty is selected by closeness to a desired price forecast.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import full_history as base
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'robustness_results';OUT.mkdir(exist_ok=True)
P,C,R,DATES=base.P,base.C,base.R,base.DATES
F=pd.DataFrame({'pre_2025':P.index.isin(base.EARLY).astype(float),'from_2025':P.index.isin(base.WAVE+base.LATEST).astype(float)},index=P.index)


def fit(dates,kind='ols',penalty=0.):
    y=R.loc[dates].to_numpy();lag=R.shift().loc[dates].to_numpy()
    f=F.loc[dates].to_numpy();fl=F.shift().loc[dates].to_numpy()
    def raw(z): return y-f@z[2:]-z[0]-z[1]*(lag-fl@z[2:])
    bounds=([-np.inf,-.95,-np.inf,-np.inf],[np.inf,.95,np.inf,np.inf])
    starts=[[.3,p,1.,1.] for p in [-.2,.3,.7]]
    ols=min([least_squares(raw,s,bounds=bounds,max_nfev=2000) for s in starts],key=lambda o:o.fun@o.fun)
    scale=max(.1,1.4826*np.median(abs(ols.fun-np.median(ols.fun))))
    if kind=='huber':
        opt=least_squares(raw,ols.x,bounds=bounds,loss='huber',f_scale=1.345*scale,max_nfev=2000)
    elif kind=='ridge_type':
        # Structural latent AR model is nonlinear in phi and beta jointly.
        # Standardized L2 on slopes, not ordinary linear Ridge; intercept unpenalized.
        scales=np.r_[np.std(lag),np.std(f,axis=0)]
        def residual(z):return np.r_[raw(z)/np.sqrt(len(y)),np.sqrt(penalty)*z[1:]*scales]
        opt=least_squares(residual,ols.x,bounds=bounds,max_nfev=2000)
    else:opt=ols
    assert opt.success
    c,phi=opt.x[:2];betas=opt.x[2:];effect=pd.Series(F.to_numpy()@betas,index=P.index)
    pars=dict(c=float(c),phi=float(phi),beta=dict(zip(F.columns,betas)),mu=float(c/(1-phi)),
              rmse=float(np.sqrt(np.mean(raw(opt.x)**2))),huber_scale=scale,penalty=penalty,n=len(y))
    residual=raw(opt.x);cutoff=1.345*scale
    weights=np.minimum(1.,cutoff/np.maximum(abs(residual),1e-15)) if kind=='huber' else np.ones(len(y))
    return pars,effect,pd.DataFrame({'date':dates,'residual':residual,'weight':weights})


def main():
    specs=[('ols','ols',0.),('huber','huber',0.)]+[(f'ridge_{p:g}','ridge_type',p) for p in [.01,.1,1.,10.]]
    rows=[];parameters={}
    for name,kind,penalty in specs:
        pars,effect,audit=fit(DATES,kind,penalty);parameters[name]=pars
        if kind=='huber':
            audit.to_csv(OUT/'huber_weights.csv',index=False)
            audit['year']=audit.date.dt.year
            audit.groupby('year').agg(n=('weight','size'),mean_weight=('weight','mean'),downweighted=('weight',lambda s:(s<1).sum())).to_csv(OUT/'huber_weights_by_year.csv')
        forecasts=base.forecast(pars,effect)
        for year,growth in forecasts.items():
            pi=100*(C.loc[f'{year}-12-01']/C.loc[f'{year-1}-12-01']-1)
            rows.append(dict(model=name,year=year,nominal_growth_pct=growth,real_growth_pct=100*((1+growth/100)/(1+pi/100)-1)))
    influences=[]
    for omit in sorted(set(DATES.year)):
        pars,effect,_=fit(DATES[DATES.year!=omit])
        fc=base.forecast(pars,effect)
        influences.append(dict(omitted_year=int(omit),n=pars['n'],longrun_real_log_pp_month=pars['mu'],growth2027=fc[2027],growth2030=fc[2030]))
    pd.DataFrame(influences).to_csv(OUT/'year_influence.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'annual_forecast.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    ref,_=base.fit(base.SCENARIOS['provisional_full_history'],'by_era')
    assert np.isclose(parameters['ols']['mu'],ref['mu'],atol=1e-5)
    assert np.isfinite(pd.DataFrame(rows).select_dtypes('number')).all().all()
    (OUT/'diagnostics.json').write_text(json.dumps(dict(checks_passed=True,targets_used=False,
        ridge='Standardized L2 penalty on phi and the two shock coefficients; intercept unpenalized. Nonlinear structural AR model, not ordinary linear Ridge.',
        selection='Penalty grid is sensitivity only. No best penalty is selected, and future predictive accuracy is not established.',
        omission='Leaving a calendar year out is an influence diagnostic, not a valid cleaning rule. Observed lags are retained; all forecasts start at actual September 2026 price.',
        calendar='Early event months remain provisional.'),indent=2))
    print(pd.DataFrame(rows).pivot(index='model',columns='year',values='nominal_growth_pct').round(3).to_string())
    print(pd.DataFrame(influences).round(3).to_string(index=False))
    print(pd.read_csv(OUT/'huber_weights_by_year.csv').round(3).to_string(index=False))
    print(json.dumps(parameters,indent=2))


if __name__=='__main__':main()
