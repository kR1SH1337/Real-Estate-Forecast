"""Plug the agreed annual sales/unsold-stock forecast into existing price coefficients.
Monthly disaggregation is an explicit bridge, not a new market forecast.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import model as price
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['svg.fonttype']='none'

ROOT=Path(__file__).resolve().parent
DEFAULT_CSV=ROOT/'inputs/portfolio_forecast_2026-10-27.csv'
DEFAULT_SNAPSHOT=ROOT/'inputs/portfolio_snapshot_2026-10-27.json'


def build_market(annual,snapshot,sales_profile='uniform',stock_profile='linear'):
    history=price.Q.copy()
    assert history.index.max()==price.P.index.max()
    last=history.index[-1]
    forecast_dates=pd.date_range(last+pd.offsets.MonthBegin(1),'2030-12-01',freq='MS')
    q=history.reindex(history.index.union(forecast_dates))
    # Optional historical seasonality is sensitivity only, never selected by price results.
    season=np.ones(12)
    if sales_profile=='seasonal':
        shares=[]
        for year in [2023,2024,2025]:
            values=history.loc[str(year)]
            assert len(values)==12
            shares.append(values.to_numpy()/values.sum())
        season=np.mean(shares,axis=0)
    for year,row in annual.iterrows():
        dates=forecast_dates[forecast_dates.year==year]
        known=float(history.loc[history.index.year==year].sum())
        remaining=float(row.sales_annual)*1000-known
        assert remaining>0
        if year==2026:
            # Source assumes uniform Q4 sales per calendar day (remaining 69/92 days).
            weights=np.asarray([d.days_in_month for d in dates],float)
        elif sales_profile=='seasonal':weights=np.asarray([season[d.month-1] for d in dates])
        else:weights=np.ones(len(dates))
        q.loc[dates]=remaining*weights/weights.sum()
    stocks=pd.read_csv(ROOT/'inputs/stock_snapshots.csv',index_col=0,parse_dates=True).unsold_m2
    stocks=stocks.loc[stocks.index<=last+pd.offsets.MonthBegin(1)]
    as_of=pd.Timestamp(snapshot['as_of'])
    current_unsold=1000*sum(float(area)-float(sold) for _,_,area,sold in snapshot['cohorts'])
    anchors=stocks.to_dict();anchors[as_of]=current_unsold
    for year,row in annual.iterrows():anchors[pd.Timestamp(year+1,1,1)]=float(row.unsold_closing)*1000
    anchor_series=pd.Series(anchors,dtype=float).sort_index()
    # Year-end stock is represented as next January 1 opening stock.
    all_dates=q.index.union(anchor_series.index)
    raw=anchor_series.reindex(all_dates).sort_index()
    if stock_profile=='loglinear':s=np.exp(np.log(raw).interpolate(method='time'))
    else:s=raw.interpolate(method='time')
    s=s.reindex(q.index)
    driver=pd.DataFrame(index=q.index)
    driver['mos_lag1']=np.log(s.shift(1)/q.rolling(6).mean().shift(1))
    driver['mos_lag3']=driver.mos_lag1.shift(2)
    driver['sales_yoy_lag1']=np.log(q.rolling(3).mean()).diff(12).shift(1)
    driver['sales_momentum_lag1']=np.log(q.rolling(3).mean()).diff(3).shift(1)
    # Reconcile both flows and point-in-time stocks to every annual input.
    for year,row in annual.iterrows():
        assert np.isclose(q.loc[str(year)].sum(),float(row.sales_annual)*1000,atol=1e-5)
        assert np.isclose(anchor_series.loc[pd.Timestamp(year+1,1,1)],float(row.unsold_closing)*1000)
    assert np.allclose(q.loc[history.index],history)
    assert np.allclose(s.loc[price.P.index],price.S)
    assert np.isclose(q.loc['2026-10-01':'2026-12-01'].sum()*.75,annual.loc[2026,'sales_period']*1000)
    assert driver.loc[forecast_dates].notna().all().all()
    return q,s,driver,anchor_series,as_of,current_unsold


def project(parameters,drivers):
    origin=price.P.index[-1];dates=pd.date_range('2026-10-01','2030-12-01',freq='MS')
    state=float(price.R.loc[origin]-price.effect(parameters,origin))
    level=float(price.P.loc[origin]);rows=[]
    for date in dates:
        x=float(drivers.loc[date,parameters['feature']]) if parameters['feature'] else 0.
        market=parameters['gamma']*(x-parameters['x_mean'])
        seasonal=0.
        if parameters['season']:
            seasonal=parameters['season_beta'][0]*np.sin(2*np.pi*date.month/12)+parameters['season_beta'][1]*np.cos(2*np.pi*date.month/12)
        state=parameters['c']+parameters['phi']*state+market+seasonal
        inflation=100*np.log(price.C.loc[date]/price.C.loc[date-pd.offsets.MonthBegin(1)])
        level*=np.exp((inflation+state)/100)
        rows.append(dict(month=date,price_index=level,real_log_pp=state,inflation_log_pp=inflation,
                         market_driver=x,market_direct_component_log_pp=market))
    return pd.DataFrame(rows).set_index('month')


def yearly(month):
    full=pd.concat([price.P,month.price_index]);rows=[]
    for year in range(2026,2031):
        g=100*(full.loc[f'{year}-12-01']/full.loc[f'{year-1}-12-01']-1)
        pi=100*(price.C.loc[f'{year}-12-01']/price.C.loc[f'{year-1}-12-01']-1)
        rows.append(dict(year=year,inflation_pct=pi,nominal_growth_pct=g,real_growth_pct=100*((1+g/100)/(1+pi/100)-1)))
    return pd.DataFrame(rows).set_index('year')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--portfolio-csv',type=Path,default=DEFAULT_CSV)
    parser.add_argument('--portfolio-snapshot',type=Path,default=DEFAULT_SNAPSHOT)
    args=parser.parse_args()
    out=ROOT/'linked_results';out.mkdir(exist_ok=True)
    raw=pd.read_csv(args.portfolio_csv,encoding='utf-8-sig')
    annual=raw[raw.housing_class=='Всего'].set_index('year').sort_index()
    assert list(annual.index)==list(range(2026,2031))
    # Check total rows against the sum of the three classes.
    by_class=raw[raw.housing_class!='Всего'].groupby('year')[['sales_annual','unsold_closing']].sum()
    assert np.allclose(by_class,annual[['sales_annual','unsold_closing']])
    snapshot=json.loads(args.portfolio_snapshot.read_text())
    assert snapshot.get('forecast_version')
    assert np.isclose(price.Q.loc['2026'].sum(),1000*sum(snapshot['actual9']))
    params=json.loads((ROOT/'results/parameters.json').read_text())
    original_params=json.dumps(params,sort_keys=True)
    q,s,drivers,anchors,as_of,current_unsold=build_market(annual,snapshot)
    monthly_inputs=pd.DataFrame({'sales_m2':q,'unsold_begin_m2':s}).join(drivers)
    monthly_inputs.to_csv(out/'monthly_market_inputs.csv',index_label='month')
    anchors.rename('unsold_m2').to_csv(out/'stock_anchors.csv',index_label='snapshot_date')
    rows=[];paths={}
    for name,m in params.items():
        updated=project(m,drivers);previous=price.annual(m)
        new=yearly(updated)
        for oldrow in previous:
            year=oldrow['year'];new.loc[year,'previous_nominal_growth_pct']=oldrow['nominal_growth_pct'];new.loc[year,'previous_real_growth_pct']=oldrow['real_growth_pct']
        new['nominal_change_pp']=new.nominal_growth_pct-new.previous_nominal_growth_pct
        new['real_change_pp']=new.real_growth_pct-new.previous_real_growth_pct
        for year,row in new.iterrows():rows.append(dict(model=name,year=int(year),**row.to_dict()))
        paths[name]=updated
    selected='mos_lag1'
    paths[selected].to_csv(out/'monthly_price_forecast.csv',index_label='month')
    comparison=pd.DataFrame(rows);comparison.to_csv(out/'model_comparison.csv',index=False)
    result=comparison[comparison.model==selected].copy().set_index('year')
    for year,row in annual.iterrows():
        result.loc[year,'sales_thousand_m2']=row.sales_annual
        result.loc[year,'unsold_closing_thousand_m2']=row.unsold_closing
        result.loc[year,'end_stock_to_annual_monthly_sales']=row.unsold_closing/(row.sales_annual/12)
        result.loc[year,'price_driver_months_december']=np.exp(drivers.loc[f'{year}-12-01','mos_lag1'])
    result.to_csv(out/'annual_forecast.csv',index_label='year')
    sensitivity=[]
    for sales_profile,stock_profile in [('uniform','linear'),('seasonal','linear'),('uniform','loglinear'),('seasonal','loglinear')]:
        _,_,x,_,_,_=build_market(annual,snapshot,sales_profile,stock_profile)
        a=yearly(project(params[selected],x))
        for year,row in a.iterrows():sensitivity.append(dict(sales_profile=sales_profile,stock_profile=stock_profile,year=int(year),**row.to_dict()))
    pd.DataFrame(sensitivity).to_csv(out/'monthly_bridge_sensitivity.csv',index=False)
    assert json.dumps(params,sort_keys=True)==original_params
    assert np.isclose(paths[selected].iloc[0].price_index,price.predict(params[selected],price.P.index[-1],pd.date_range('2026-10-01','2026-10-01',freq='MS'))[0].iloc[0])
    assert np.isfinite(comparison.select_dtypes('number')).all().all()
    assert np.allclose(1+result.nominal_growth_pct/100,(1+result.inflation_pct/100)*(1+result.real_growth_pct/100))
    # With zero market coefficient, a new market path cannot affect prices.
    zero=dict(params[selected]);zero['gamma']=0.
    assert np.allclose(project(zero,drivers).price_index,project(zero,drivers*2).price_index)
    diagnostics=dict(as_of=snapshot['forecast_version'],price_actual_through=str(price.P.index[-1].date()),
        portfolio_snapshot_date=str(as_of.date()),initial_unsold_m2=current_unsold,selected_model=selected,
        coefficients_refitted=False,parameter_file_sha256=hashlib.sha256((ROOT/'results/parameters.json').read_bytes()).hexdigest(),
        checks_passed=True,source_csv_sha256=hashlib.sha256(args.portfolio_csv.read_bytes()).hexdigest(),
        source_snapshot_sha256=hashlib.sha256(args.portfolio_snapshot.read_bytes()).hexdigest(),
        monthly_bridge='Q4 2026 uniform per calendar day, subsequent years uniform per month. Stocks linearly interpolated in time through actual monthly snapshots, October 23 snapshot and next January 1 year-end forecast anchors.',
        scope='One-way conditional update: sales, launches, completions and portfolio model are not re-estimated; price-to-sales feedback is not solved. Historical price/market coefficient instability and validation limitations are unchanged.')
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'parameters_used.json').write_text(json.dumps(params[selected],indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].plot(result.index,result.previous_real_growth_pct,'o--',label='Прежний: постоянное состояние')
    axs[0].plot(result.index,result.real_growth_pct,'o-',label='С прогнозом продаж и остатка')
    axs[0].set(ylabel='Реальный рост цен, декабрь к декабрю, %',xticks=result.index);axs[0].legend(fontsize=8)
    future=monthly_inputs.loc['2026-10-01':]
    axs[1].plot(future.index,np.exp(future.mos_lag1),label='Показатель для модели цен')
    axs[1].set(ylabel='Обеспеченность, месяцев продаж');axs[1].legend(fontsize=8)
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'linked_forecast.svg');fig.savefig(out/'linked_forecast.png',dpi=130);plt.close(fig)
    print(result.round(3).to_string())
    print(pd.DataFrame(sensitivity).pivot_table(index='year',columns=['sales_profile','stock_profile'],values='real_growth_pct').round(3).to_string())


if __name__=='__main__':main()
