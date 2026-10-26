"""Calibrated Moscow price scenario, not an econometrically identified forecast.
Run python model.py. Dependencies: numpy pandas scipy matplotlib.
All rate components below are monthly log percentage points.
"""
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
D=pd.read_csv(ROOT/'inputs/monthly_data.csv',index_col=0,parse_dates=True)
ACTUAL=D['index'].dropna()
assert ACTUAL.index[-1]==pd.Timestamp('2026-09-01')
FUTURE=pd.date_range('2026-10-01','2030-12-01',freq='MS')
H=np.arange(len(FUTURE),dtype=float)
YEARS=list(range(2026,2031))
CPI=D.cpi_index_dec2020
ORIGINAL_PI_MONTH=100*np.log(CPI).diff().loc[FUTURE].to_numpy()
# Uniform monthly log inflation within each forecast year; annual CPI is preserved.
PI_MONTH=ORIGINAL_PI_MONTH.copy()
for year in YEARS:
    mask=FUTURE.year==year
    PI_MONTH[mask]=ORIGINAL_PI_MONTH[mask].mean()
PI_YEAR=np.array([100*(CPI.loc[f'{y}-12-01']/CPI.loc[f'{y-1}-12-01']-1) for y in YEARS])
TARGET=np.r_[10.,PI_YEAR[1:]+np.array([1.5,2.,1.8,1.8])]
SCALES=np.array([.25,.4,.4,.4,.4])
NAMES=['longrun_premium_pp_at_4pct_cpi','initial_adjustment_log_pp_month',
       'recovery_amplitude_log_pp_month','decay_time_months']
LOW=np.array([0.,-2.,0.,2.]);HIGH=np.array([2.,0.,2.,24.])

def evaluate(z,pi_month=PI_MONTH):
    delta,a,b,tau=z
    mu=100/12*np.log((104+delta)/104)
    correction=a*np.exp(-H/tau)
    recovery=b*(H/tau)*np.exp(-H/tau)
    real=mu+correction+recovery
    growth=pi_month+real
    price=pd.Series(ACTUAL.iloc[-1]*np.exp(np.cumsum(growth)/100),index=FUTURE)
    whole=pd.concat([ACTUAL,price])
    annual=np.array([100*(whole.loc[f'{y}-12-01']/whole.loc[f'{y-1}-12-01']-1) for y in YEARS])
    return dict(price=price,annual=annual,real=real,correction=correction,recovery=recovery,mu=mu,
                nominal_month=100*np.expm1(growth/100))

def residual(z,target=TARGET,weight=1.,mask=None):
    if mask is None:mask=np.ones(5,dtype=bool)
    errors=(evaluate(z)['annual']-target)/SCALES
    # These are explicit expert regularizers, not estimated probability distributions.
    return np.r_[np.sqrt(weight)*errors[mask],(z[0]-1.8)/.2,(z[3]-9.)/6.,z[2]/1.]

def calibrate(target=TARGET,weight=1.,mask=None,multi=False):
    starts=[[1.8,-.6,.6,9.]]
    if multi:
        starts += [[d,a,b,t] for d in [1.5,1.9] for a in [-.3,-1.] for b in [.2,1.] for t in [4.,14.]]
    fits=[least_squares(residual,x,args=(target,weight,mask),bounds=(LOW,HIGH),
                        xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=3000) for x in starts]
    good=[r for r in fits if r.success]
    assert good
    result=min(good,key=lambda r:np.sum(r.fun**2))
    return result, fits

def cpi_scenario(change_pp):
    # Preserve within-year monthly profile while shifting December/December inflation.
    # In 2026 only the three future months are changed; observed CPI stays untouched.
    out=PI_MONTH.copy();pis=PI_YEAR+change_pp
    for i,y in enumerate(YEARS):
        mask=FUTURE.year==y
        out[mask]+=100*np.log((1+pis[i]/100)/(1+PI_YEAR[i]/100))/mask.sum()
    return out,pis

def main():
    opt,fits=calibrate(multi=True);z=opt.x;v=evaluate(z)
    month=pd.DataFrame({'price_index':v['price'],'cpi_log_pp_month':PI_MONTH,
         'source_cpi_log_pp_month':ORIGINAL_PI_MONTH,
         'longrun_real_log_pp_month':v['mu'],'temporary_adjustment_log_pp_month':v['correction'],
         'recovery_log_pp_month':v['recovery'],'real_log_pp_month':v['real'],
         'price_mom_pct':v['nominal_month']},index=FUTURE)
    month.to_csv(OUT/'monthly_forecast.csv',index_label='month')
    annual=pd.DataFrame({'year':YEARS,'inflation_pct':PI_YEAR,'guidance_price_growth_pct':TARGET,
          'model_price_growth_pct':v['annual'],'model_premium_pp':v['annual']-PI_YEAR,
          'deviation_from_guidance_pp':v['annual']-TARGET,'calibration_scale_pp':SCALES})
    annual.to_csv(OUT/'annual_forecast.csv',index=False)
    pd.DataFrame({'parameter':NAMES,'value':z,'lower_bound':LOW,'upper_bound':HIGH}).to_csv(OUT/'parameters.csv',index=False)
    sensitivity=[]
    for weight in [.25,1.,4.]:
        r,_=calibrate(weight=weight);p=evaluate(r.x)
        for i,y in enumerate(YEARS):sensitivity.append({'test':'guidance_weight','variant':str(weight),
              'year':y,'growth_pct':p['annual'][i],'premium_pp':p['annual'][i]-PI_YEAR[i]})
    for leave in range(5):
        mask=np.ones(5,dtype=bool);mask[leave]=False
        r,_=calibrate(mask=mask);p=evaluate(r.x)
        for i,y in enumerate(YEARS):sensitivity.append({'test':'leave_one_guidance_out','variant':str(YEARS[leave]),
              'year':y,'growth_pct':p['annual'][i],'premium_pp':p['annual'][i]-PI_YEAR[i]})
    # Same 2026 guidance, no individual 2027–2030 anchors: exposes dependence on their use.
    r,_=calibrate(mask=np.array([True,False,False,False,False]));p=evaluate(r.x)
    for i,y in enumerate(YEARS):sensitivity.append({'test':'2026_only','variant':'no_2027_2030_guidance',
              'year':y,'growth_pct':p['annual'][i],'premium_pp':p['annual'][i]-PI_YEAR[i]})
    pd.DataFrame(sensitivity).to_csv(OUT/'calibration_sensitivity.csv',index=False)
    scenarios=[]
    for change in [-1.,0.,1.]:
        pim,piy=cpi_scenario(change);p=evaluate(z,pim)
        for i,y in enumerate(YEARS):scenarios.append({'scenario':f'CPI_{change:+.0f}pp_fixed_parameters',
           'year':y,'inflation_pct':piy[i],'growth_pct':p['annual'][i]})
    for factor in [.75,1.25]:
        zz=z.copy();zz[1]*=factor;p=evaluate(zz)
        for i,y in enumerate(YEARS):scenarios.append({'scenario':f'temporary_adjustment_x{factor}',
           'year':y,'inflation_pct':PI_YEAR[i],'growth_pct':p['annual'][i]})
    pd.DataFrame(scenarios).to_csv(OUT/'fixed_parameter_scenarios.csv',index=False)
    # Verify identities and parameter stability, not historical predictive accuracy.
    assert np.isfinite(month.to_numpy()).all() and (month.price_index>0).all()
    assert np.allclose(100*np.log(month.price_index/pd.concat([ACTUAL.iloc[[-1]],month.price_index]).shift(1).loc[FUTURE]),
                       PI_MONTH+month.real_log_pp_month)
    assert np.allclose(month.real_log_pp_month,month.longrun_real_log_pp_month+
                       month.temporary_adjustment_log_pp_month+month.recovery_log_pp_month)
    costs=[float(np.sum(r.fun**2)) for r in fits if r.success]
    assert max(costs)-min(costs)<1e-5
    # Long horizon limit at constant 4% CPI must reproduce the calibrated asymptote.
    assert abs(104*np.expm1(12*v['mu']/100)-z[0])<1e-10
    last_cpi=float(CPI.loc['2026-09-01']);dec_cpi=float(CPI.loc['2026-12-01'])
    q4=100*(month.price_index.loc['2026-12-01']/ACTUAL.iloc[-1]-1)
    checks={'calibration_converged':bool(opt.success),'multistart_successes':len(costs),
       'objective':float(np.sum(opt.fun**2)),'multistart_objective_spread':max(costs)-min(costs),
       'nine_month_growth_pct':100*(ACTUAL.iloc[-1]/ACTUAL.loc['2025-12-01']-1),
       'q4_growth_pct':q4,'q4_cpi_growth_pct':100*(dec_cpi/last_cpi-1),
       'max_annual_guidance_deviation_pp':float(np.max(abs(v['annual']-TARGET))),
       'parameter_on_bound':bool(np.any(np.isclose(z,LOW,atol=1e-5)|np.isclose(z,HIGH,atol=1e-5))),
       'historical_predictive_validation':'not performed; future guidance is used in calibration',
       'sales_supply_link':'not included; no identified elasticity from previous diagnostics'}
    (OUT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    config={'formula':'r_h=mu+(a+b*h/tau)*exp(-h/tau); mu=100/12*log((104+delta)/104)',
      'origin':'2026-09-01','h_zero':'2026-10-01','rate_units':'monthly logarithmic percentage points',
      'annual_guidance':dict(zip(YEARS,TARGET.tolist())), 'annual_scales_pp':SCALES.tolist(),
      'regularization':{'longrun_delta_center':1.8,'longrun_delta_scale':.2,'tau_center_months':9.,
                        'tau_scale_months':6.,'b_center':0.,'b_scale':1.},'parameters':dict(zip(NAMES,z.tolist()))}
    (OUT/'calibration_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    axes[0].plot(YEARS,PI_YEAR,ls=':',color='#747b83',label='Инфляция')
    axes[0].plot(YEARS,v['annual'],marker='o',color='#245b83',label='Расчёт модели')
    axes[0].scatter(YEARS,TARGET,color='#b46a27',marker='x',s=65,label='Ориентиры калибровки')
    axes[0].set_title('Москва: рост цен, декабрь к декабрю',loc='left');axes[0].set_ylabel('%');axes[0].legend(ncol=3)
    axes[1].plot(FUTURE,month.price_mom_pct,color='#245b83',label='Рост цен')
    axes[1].plot(FUTURE,100*np.expm1(PI_MONTH/100),ls=':',color='#747b83',label='Инфляция')
    axes[1].set_title('Месячная траектория',loc='left');axes[1].set_ylabel('% за месяц');axes[1].legend()
    axes[2].plot(FUTURE,month.real_log_pp_month,color='#245b83',label='Реальный рост, всего')
    axes[2].plot(FUTURE,month.temporary_adjustment_log_pp_month,color='#b46a27',label='Временная корректировка')
    axes[2].plot(FUTURE,month.recovery_log_pp_month,color='#56816a',label='Восстановление')
    axes[2].axhline(v['mu'],ls=':',color='#747b83',label='Долгосрочный компонент')
    axes[2].set_title('Компоненты месячного реального роста',loc='left');axes[2].set_ylabel('лог. п.п.');axes[2].legend(ncol=2)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'forecast.png',dpi=150);plt.close(fig)
    print(annual.round(5).to_string(index=False));print(dict(zip(NAMES,z)));print(checks)
    print('LEAVE OUT',pd.DataFrame(sensitivity).query("test=='leave_one_guidance_out'").pivot(index='year',columns='variant',values='growth_pct').round(3).to_string())

if __name__=='__main__':main()
