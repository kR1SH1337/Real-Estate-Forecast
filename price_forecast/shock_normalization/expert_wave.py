"""Expert-identified family-mortgage anticipation episode, no price targets.
Confirmed expert window: September 2025 through January 2026 inclusive.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expert_wave_results';OUT.mkdir(exist_ok=True)
D=pd.read_csv(ROOT.parent/'inputs/monthly_data.csv',index_col=0,parse_dates=True)
P=D['index'].dropna();C=D.cpi_index_dec2020
R=(100*(np.log(P).diff()-np.log(C).diff())).reindex(P.index)
DATES=R.dropna().index[1:]
START=pd.Timestamp('2025-09-01');END=pd.Timestamp('2026-01-01')


def estimate(start=START,indicator=True):
    dummy=pd.Series(((P.index>=start)&(P.index<=END)).astype(float),index=P.index)
    y=R.loc[DATES].to_numpy();lag=R.shift().loc[DATES].to_numpy()
    d=dummy.loc[DATES].to_numpy() if indicator else np.zeros(len(DATES))
    dl=dummy.shift().loc[DATES].to_numpy() if indicator else np.zeros(len(DATES))
    def resid(z):
        c,phi,beta=z
        return y-beta*d-c-phi*(lag-beta*dl)
    if indicator:
        fit=least_squares(resid,[.3,.3,1.],bounds=([-np.inf,-.95,-np.inf],[np.inf,.95,np.inf]),max_nfev=2000)
        c,phi,beta=fit.x
    else:
        fit=least_squares(lambda z:resid([*z,0]),[.3,.3],bounds=([-np.inf,-.95],[np.inf,.95]),max_nfev=2000)
        c,phi=fit.x;beta=0.
    assert fit.success
    return dict(c=float(c),phi=float(phi),beta=float(beta),mu=float(c/(1-phi)),
                rmse=float(np.sqrt(np.mean(fit.fun**2))),start=str(start.date()),end=str(END.date()),
                n=int(len(y)),n_wave=int(d.sum())),dummy


def forecast(m):
    dates=pd.date_range('2026-10-01','2030-12-01',freq='MS')
    real=float(R.iloc[-1]);level=float(P.iloc[-1]);out=[]
    for date in dates:
        real=m['c']+m['phi']*real
        level*=np.exp(real/100)*C.loc[date]/C.loc[date-pd.offsets.MonthBegin(1)]
        out.append(level)
    full=pd.concat([P,pd.Series(out,index=dates)])
    return {year:100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1) for year in range(2026,2031)}


def main():
    base,_=estimate(indicator=False);main,dummy=estimate()
    pars=dict(baseline=base,expert_wave=main)
    (OUT/'parameters.json').write_text(json.dumps(pars,indent=2))
    audit=pd.DataFrame(dict(price=P,nominal_log_pp=100*np.log(P).diff(),real_log_pp=R,wave_dummy=dummy))
    audit['estimated_wave_log_pp']=main['beta']*dummy
    audit['adjusted_nominal_log_pp']=audit.nominal_log_pp-audit.estimated_wave_log_pp
    audit['adjusted_real_log_pp']=R-audit.estimated_wave_log_pp
    audit.to_csv(OUT/'cleaning_audit.csv',index_label='month')
    rows=[]
    for year in range(2022,2027):
        a=audit.loc[str(year)]
        rows.append(dict(year=year,months=len(a),observed_growth_pct=100*np.expm1(a.nominal_log_pp.sum()/100),
                         adjusted_growth_pct=100*np.expm1(a.adjusted_nominal_log_pp.sum()/100),
                         removed_log_pp=a.estimated_wave_log_pp.sum()))
    pd.DataFrame(rows).to_csv(OUT/'historical_year_cleaning.csv',index=False)
    annual=pd.DataFrame({'year':range(2026,2031),'baseline_pct':list(forecast(base).values()),'wave_adjusted_pct':list(forecast(main).values())})
    annual.to_csv(OUT/'annual_forecast.csv',index=False)
    sensitivity=[]
    for start in pd.date_range('2025-07-01','2025-10-01',freq='MS'):
        m,_=estimate(start)
        sensitivity.append(dict(start=str(start.date()),beta_log_pp_month=m['beta'],longrun_real_log_pp_month=m['mu'],**{f'growth{y}':v for y,v in forecast(m).items()}))
    pd.DataFrame(sensitivity).to_csv(OUT/'start_date_sensitivity.csv',index=False)
    assert np.allclose(audit.real_log_pp.dropna(),(audit.adjusted_real_log_pp+audit.estimated_wave_log_pp).dropna())
    assert len(audit.loc[audit.wave_dummy==1])==5
    diagnostics=dict(targets_used=False,forecast_anchored_to_actual_september=True,
                     expert_period='User-confirmed window: September 2025 through January 2026. Not automatically detected.',
                     interpretation='Conditional association of a selected five-month episode; omitted factors may enter its coefficient. Not a causal subsidy estimate.',
                     historical_backtest='No honest backtest of a new event type from a single retrospectively identified wave. Future 2026-2030 forecast guidance not used.',checks_passed=True)
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(audit.index,100*np.expm1(audit.nominal_log_pp/100),color='gray',label='Observed nominal growth')
    ax.plot(audit.index,100*np.expm1(audit.adjusted_nominal_log_pp/100),color='#007c91',label='After expert wave adjustment')
    ax.axvspan(START,END,color='orange',alpha=.12,label='Expert episode')
    ax.axhline(0,color='black',lw=.5);ax.set_ylabel('Monthly price growth (%)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'cleaning.svg');plt.close(fig)
    print(json.dumps(pars,indent=2));print(annual.round(3).to_string(index=False));print(pd.DataFrame(rows).round(3).to_string(index=False));print(pd.DataFrame(sensitivity).round(3).to_string(index=False))


if __name__=='__main__':main()
