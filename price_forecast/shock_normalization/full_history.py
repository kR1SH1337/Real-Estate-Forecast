"""Repeated policy anticipation episodes, expert-confirmed and provisional.
No price-growth targets. Historical windows require expert review.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import expert_wave as data

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'full_history_results';OUT.mkdir(exist_ok=True)
P,C,R,DATES=data.P,data.C,data.R,data.DATES
WAVE=list(pd.date_range('2025-09-01','2026-01-01',freq='MS'))
LATEST=list(pd.to_datetime(['2026-06-01','2026-09-01']))
DATA_ANOMALIES=list(pd.to_datetime(['2025-02-01','2025-04-01']))
# Historical candidate months follow a policy calendar, not a price threshold.
EARLY=list(pd.to_datetime(['2021-06-01','2022-12-01','2023-08-01','2023-09-01','2023-12-01','2024-05-01','2024-06-01']))
SCENARIOS={'unadjusted':[], 'confirmed_2025_wave':WAVE,
           'all_user_confirmed':WAVE+LATEST,
           'provisional_full_history':WAVE+LATEST+EARLY}


def fit(months,grouped=False):
    if grouped in ('by_era','by_era_plus_data'):
        early=[d for d in months if d.year<2025]
        recent=[d for d in months if d.year>=2025 and (grouped!='by_era_plus_data' or d not in DATA_ANOMALIES)]
        f=pd.DataFrame({'pre_2025':P.index.isin(early).astype(float),
                        'from_2025':P.index.isin(recent).astype(float)},index=P.index)
        if grouped=='by_era_plus_data':
            f['sales_data_2025']=P.index.isin(DATA_ANOMALIES).astype(float)
            assert (f.sum(axis=1)<=1).all()
        assert not (f.pre_2025*f.from_2025).any()
    elif grouped:
        f=pd.DataFrame({'long_wave':P.index.isin(WAVE).astype(float),
                        'other_episodes':P.index.isin([d for d in months if d not in WAVE]).astype(float)},index=P.index)
    elif months:
        f=pd.DataFrame({'shock':P.index.isin(months).astype(float)},index=P.index)
    else:
        f=pd.DataFrame(index=P.index)
    y=R.loc[DATES].to_numpy();lag=R.shift().loc[DATES].to_numpy()
    x=f.loc[DATES].to_numpy();xl=f.shift().loc[DATES].to_numpy();k=len(f.columns)
    def error(z):return y-x@z[2:]-z[0]-z[1]*(lag-xl@z[2:])
    starts=[np.r_[.3,p,np.ones(k)] for p in [-.2,.2,.7]]
    fits=[least_squares(error,s,bounds=(np.r_[-np.inf,-.95,np.full(k,-np.inf)],np.r_[np.inf,.95,np.full(k,np.inf)]),max_nfev=2000) for s in starts]
    opt=min([o for o in fits if o.success],key=lambda o:o.fun@o.fun)
    c,phi=opt.x[:2];effect=pd.Series(f.to_numpy()@opt.x[2:],index=P.index)
    pars=dict(c=float(c),phi=float(phi),beta=dict(zip(f.columns,opt.x[2:])),mu=float(c/(1-phi)),
              rmse=float(np.sqrt(np.mean(opt.fun**2))),n=int(len(y)),n_shock_months=int((f.sum(axis=1)>0).sum()),
              dummy_counts={col:int(f[col].sum()) for col in f})
    return pars,effect


def forecast(pars,effect):
    # Remove current shock from the initial growth state, preserving observed price level.
    state=float(R.iloc[-1]-effect.iloc[-1]);level=float(P.iloc[-1]);out=[]
    dates=pd.date_range('2026-10-01','2030-12-01',freq='MS')
    for date in dates:
        state=pars['c']+pars['phi']*state
        level*=np.exp(state/100)*C.loc[date]/C.loc[date-pd.offsets.MonthBegin(1)]
        out.append(level)
    full=pd.concat([P,pd.Series(out,index=dates)])
    return {y:100*(full.loc[f'{y}-12-01']/full.loc[f'{y-1}-12-01']-1) for y in range(2026,2031)}


def main():
    annual=[];parameters={};audits=[];history=[]
    specs=[(name,months,False) for name,months in SCENARIOS.items()]
    specs+=[('provisional_full_history_two_coefficients',SCENARIOS['provisional_full_history'],True)]
    specs+=[('provisional_full_history_by_era',SCENARIOS['provisional_full_history'],'by_era')]
    specs+=[('provisional_full_history_by_era_plus_data',SCENARIOS['provisional_full_history']+DATA_ANOMALIES,'by_era_plus_data')]
    for name,months,grouped in specs:
        pars,effect=fit(months,grouped);parameters[name]=pars
        for year,growth in forecast(pars,effect).items():
            inflation=100*(C.loc[f'{year}-12-01']/C.loc[f'{year-1}-12-01']-1)
            real=100*((1+growth/100)/(1+inflation/100)-1)
            annual.append(dict(scenario=name,year=year,growth_pct=growth,inflation_pct=inflation,real_growth_pct=real))
        nominal=100*np.log(P).diff();adjusted=nominal-effect
        assert np.allclose(nominal.dropna(),(adjusted+effect).dropna())
        for date in R.dropna().index:
            audits.append(dict(scenario=name,date=str(date.date()),shock_dummy=int(date in months),
                               early_policy_component_log_pp=pars['beta'].get('pre_2025',0.)*int(date in EARLY),
                               recent_policy_component_log_pp=pars['beta'].get('from_2025',0.)*int(date in WAVE+LATEST),
                               sales_data_component_log_pp=pars['beta'].get('sales_data_2025',0.)*int(date in DATA_ANOMALIES),
                               nominal_log_pp=nominal.loc[date],effect_log_pp=effect.loc[date],adjusted_nominal_log_pp=adjusted.loc[date],
                               observed_real_log_pp=R.loc[date],adjusted_real_log_pp=R.loc[date]-effect.loc[date],
                               observed_real_mom_pct=100*np.expm1(R.loc[date]/100),
                               adjusted_real_mom_pct=100*np.expm1((R.loc[date]-effect.loc[date])/100)))
        for year in range(2022,2027):
            history.append(dict(scenario=name,year=year,months=len(P.loc[str(year)]),
                                observed_growth_pct=100*np.expm1(nominal.loc[str(year)].sum()/100),
                                adjusted_growth_pct=100*np.expm1(adjusted.loc[str(year)].sum()/100),
                                inflation_pct=100*np.expm1((100*np.log(C).diff()).reindex(P.index).loc[str(year)].sum()/100),
                                observed_real_growth_pct=100*np.expm1(R.loc[str(year)].sum()/100),
                                adjusted_real_growth_pct=100*np.expm1((R-effect).loc[str(year)].sum()/100)))
    calendar=[]
    for date in P.index:
        calendar.append(dict(date=str(date.date()),user_confirmed=int(date in WAVE+LATEST),
                             historical_candidate=int(date in EARLY),provisional_all=int(date in WAVE+LATEST+EARLY),
                             dummy_pre_2025=int(date in EARLY),dummy_from_2025=int(date in WAVE+LATEST),
                             dummy_sales_data_2025=int(date in DATA_ANOMALIES)))
    pd.DataFrame(calendar).to_csv(OUT/'calendar.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_forecast.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'cleaning_audit.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'historical_year_cleaning.csv',index=False)
    forecast_real=pd.DataFrame(annual)
    assert np.allclose(1+forecast_real.growth_pct/100,(1+forecast_real.inflation_pct/100)*(1+forecast_real.real_growth_pct/100))
    history_real=pd.DataFrame(history)
    assert np.allclose(1+history_real.adjusted_growth_pct/100,(1+history_real.inflation_pct/100)*(1+history_real.adjusted_real_growth_pct/100))
    (OUT/'parameters.json').write_text(json.dumps(parameters,indent=2))
    assert parameters['all_user_confirmed']['n_shock_months']==7
    assert parameters['provisional_full_history']['n_shock_months']==14
    assert parameters['provisional_full_history_by_era']['dummy_counts']=={'pre_2025':7,'from_2025':7}
    assert parameters['provisional_full_history_by_era_plus_data']['dummy_counts']=={'pre_2025':7,'from_2025':7,'sales_data_2025':2}
    assert all(abs(v['phi'])<1 for v in parameters.values())
    (OUT/'diagnostics.json').write_text(json.dumps(dict(targets_used=False,checks_passed=True,
        distinction='7 user-confirmed months vs 7 additional historical candidate months; full history not yet expert-confirmed.',
        limitations='Retrospective event selection; shared coefficient assumes identical monthly effect; two-coefficient variant is sensitivity, not a selected winner; no causal identification or independent backtest. No post-event payback effect is included.'),indent=2))
    print(pd.DataFrame(annual).pivot(index='year',columns='scenario',values='growth_pct').round(3).to_string())
    print(json.dumps(parameters,indent=2))


if __name__=='__main__':main()
