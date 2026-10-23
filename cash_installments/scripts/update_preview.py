"""Build the README chart from a saved model run (default: reference_results)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=ROOT / 'reference_results')
args = parser.parse_args()
history = pd.read_csv(args.results / 'input_series.csv', parse_dates=['date'])
forecast = pd.read_csv(args.results / 'forecast_monthly.csv', parse_dates=['date'])
metadata = json.loads((args.results / 'diagnostics.json').read_text(encoding='utf-8'))
fitted = history.set_index('date')[['ks_avg3', 'real']].copy()
fitted['area_mln_m2'] = (
    metadata['diagnostics']['alpha'] + metadata['diagnostics']['beta'] * fitted.ks_avg3
) * 1_000 / metadata['base_price_rub_m2']
# Show fitted values only where a historical sales observation is available.
fitted.loc[fitted.real.isna(), 'area_mln_m2'] = float('nan')
fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#f5f7fa')
ax.plot(history.date, history.area_mln_m2, color='#193b61', linewidth=1.9, label='Историческая расчётная площадь')
ax.plot(fitted.index, fitted.area_mln_m2, color='#16877e', linewidth=1.8,
        linestyle='--', marker='o', markersize=2.5,
        label='Оценка модели на истории')
ax.plot(forecast.date, forecast.area_mln_m2, color='#d57b27', linewidth=2.1, label='Условный прогноз')
ax.axvspan(forecast.date.iloc[0], forecast.date.iloc[-1], color='#d57b27', alpha=.07)
ax.axvline(forecast.date.iloc[0], color='#a56730', linestyle='--', linewidth=1)
ax.set_title('Москва · продажи за наличные и в рассрочку', loc='left', fontsize=16, fontweight='bold', pad=18)
ax.set_ylabel('Расчётная площадь, млн м² в месяц')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(bottom=0)
ax.grid(axis='y', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper right', frameon=False, fontsize=9)
footer = 'История дефлирована к январю 2023; базовая цена — 282 700 руб./м².\nОценки на истории — по последним коэффициентам модели, не ретропрогноз.\nПрогноз условен на сценарий КС; пропуски истории не интерполированы.'
if 'price_index_assumed' in history and history['price_index_assumed'].astype(str).str.lower().eq('true').any():
    footer += '\nПредварительно: индекс сентября 2026 принят равным августовскому.'
fig.supxlabel(footer, fontsize=9, color='#526171')
output = ROOT / 'docs/images/sales_area.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=170)
print(output)

