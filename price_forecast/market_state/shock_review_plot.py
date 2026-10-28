"""Dual-axis factual real monthly price changes and unsmoothed sales.

Shading is the existing event calendar, not a revised or price-selected calendar.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import model as source

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'shock_review'


def main():
    OUT.mkdir(exist_ok=True)
    dates=source.P.loc['2022-01-01':].index
    real_mom=100*np.expm1(source.R.loc[dates]/100)
    sales=source.Q.loc[dates]/1000
    assert len(dates)==57 and real_mom.notna().all() and sales.notna().all()
    exact=100*((source.P/source.P.shift())/(source.C/source.C.shift())-1).loc[dates]
    assert np.allclose(real_mom,exact)
    fig,ax=plt.subplots(figsize=(16,7.4),facecolor='white')
    fig.subplots_adjust(left=.075,right=.925,bottom=.27,top=.82)
    right=ax.twinx()
    colors={'early':'#e9b95b','recent':'#76b8a0','data':'#b99bd7'}
    for date in dates:
        for group,color in colors.items():
            if source.F.loc[date,group]:ax.axvspan(date,date+pd.offsets.MonthBegin(1),color=color,alpha=.25,lw=0,zorder=0)
    x=dates+pd.Timedelta(days=14)
    lp,=ax.plot(x,real_mom,color='#174a7e',lw=2.2,marker='o',ms=3.4,label='Реальный прирост цен, % за месяц',zorder=4)
    ls,=right.plot(x,sales,color='#c45e21',lw=2.0,marker='o',ms=3.2,label='Продажи ДДУ, тыс. м² за месяц',zorder=3)
    ax.set_ylim(-5,5);right.set_ylim(0,600)
    ax.set_ylabel('Реальный прирост цен, % к предыдущему месяцу',color='#174a7e',fontsize=11,labelpad=12)
    right.set_ylabel('Продажи ДДУ, тыс. м² за месяц',color='#c45e21',fontsize=11,labelpad=12)
    ax.tick_params(axis='y',colors='#174a7e');right.tick_params(axis='y',colors='#c45e21')
    ax.set_yticks(np.arange(-5,6,1));right.set_yticks(np.arange(0,601,100))
    ax.axhline(0,color='#657382',lw=.9,alpha=.7)
    ax.grid(axis='y',color='#dce2e8',lw=.65,alpha=.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))
    ax.tick_params(axis='x',labelsize=9,rotation=45)
    ax.set_xlim(dates[0],dates[-1]+pd.offsets.MonthBegin(1))
    for a in [ax,right]:
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_color('#b8c1ca')
    ax.spines['left'].set_color('#174a7e');right.spines['right'].set_color('#c45e21')
    fig.text(.075,.94,'Цены и продажи: проверка разметки шоков',fontsize=18,fontweight='bold',color='#182a3c')
    fig.text(.075,.895,'Москва · январь 2022 — сентябрь 2026 · фактические месячные данные',fontsize=11,color='#536477')
    fig.legend(handles=[lp,ls],loc='upper left',bbox_to_anchor=(.07,.865),frameon=False,ncol=2,fontsize=10)
    patches=[Patch(facecolor=colors['early'],alpha=.55,label='Ранние ипотечные эпизоды — предварительная разметка'),
             Patch(facecolor=colors['recent'],alpha=.55,label='Ипотечные эпизоды с 2025 года'),
             Patch(facecolor=colors['data'],alpha=.55,label='Аномалии данных: февраль и апрель 2025')]
    fig.legend(handles=patches,loc='upper left',bbox_to_anchor=(.07,.155),frameon=False,ncol=2,fontsize=9.5)
    fig.text(.075,.02,'Цены дефлированы на ИПЦ, шоки ещё не вычтены. Продажи не сглажены и не сдвинуты: обе линии относятся к одному месяцу.\nФон показывает прежнюю разметку; шкалы независимы, положение пересечений линий не имеет самостоятельного смысла.',fontsize=9,color='#536477',linespacing=1.5)
    plt.rcParams['svg.fonttype']='none'
    fig.savefig(OUT/'prices_sales_dual_axis.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'prices_sales_dual_axis.svg',facecolor='white')
    plt.close(fig)
    rows=[dict(month=str(d.date()),real_mom_pct=float(real_mom.loc[d]),sales_thousand_m2=float(sales.loc[d]),
               early=int(source.F.loc[d,'early']),recent=int(source.F.loc[d,'recent']),data=int(source.F.loc[d,'data'])) for d in dates]
    (OUT/'monthly_data.json').write_text(json.dumps(dict(period='2022-01 through 2026-09',price='100*((P_t/P_previous)/(CPI_t/CPI_previous)-1)',sales='Unsmoothed contemporaneous DDU sales area, thousand m2; same source series used in market model',event_calendar='Existing three-group calendar, unchanged. Data-anomaly effects and early events remain hypotheses.',rows=rows),ensure_ascii=False,indent=2))
    print('Saved dual-axis chart; observations:',len(rows))


if __name__=='__main__':main()
