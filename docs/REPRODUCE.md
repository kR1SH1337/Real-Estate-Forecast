# Воспроизведение и техническое изучение

## Актуальный совместный расчет

Требуется Python 3.10+. Команды выполняются из корня репозитория:

```bash
python -m pip install -r requirements.txt
python -m pip install -r price_forecast/requirements.txt
python price_forecast/market_state/reanchored_price_equilibrium.py
```

Скрипт использует сохраненные агрегированные входы ипотечного канала и кешевых площадей. Он не обновляет автоматически исходную ипотечную Excel-модель или кеш-модель.

Результаты сценария **growth54_september_2026**:

| Файл | Назначение |
|---|---|
| [annual_equilibrium.csv](../price_forecast/market_state/reanchored_price_results/annual_equilibrium.csv) | Годовой итог прогноза |
| [portfolio_snapshot_equilibrium.json](../price_forecast/market_state/reanchored_price_results/portfolio_snapshot_equilibrium.json) | Согласованные входы для повтора портфельного баланса |
| [monthly_paths.csv](../price_forecast/market_state/reanchored_price_results/monthly_paths.csv) | Месячные траектории всех вариантов; фильтр по scenario = growth54_september_2026 |
| [convergence_checks.csv](../price_forecast/market_state/reanchored_price_results/convergence_checks.csv) | Проверки сходимости |

Описание входов и перевода индекса в рубли: [REANCHORED_PRICE_EQUILIBRIUM.md](../price_forecast/market_state/REANCHORED_PRICE_EQUILIBRIUM.md).

## Самостоятельные модели

### Кеш и рассрочка

```bash
python cash_installments/cash_model.py
```

[Инструкция](../cash_installments/README.md) · [Подробная методология](../cash_installments/docs/usage.md). Новый результат сохраняется в cash_installments/results/. Он не заменяет автоматически кешевые входы совместного расчета.

### Краткосрочные запуски

```bash
python launches/model.py --input-json /path/to/local_snapshot.json
```

Реальные входы отдельного квартального блока хранятся локально. [Синтетический пример](../launches/examples/synthetic_input.json) проверяет установку, а не прогноз рынка:

```bash
python launches/model.py --input-json launches/examples/synthetic_input.json
```

### Портфель

Для воспроизведения согласованного портфельного баланса используйте движок совместной системы:

```bash
python price_forecast/market_state/portfolio_engine.py \
  --input-json price_forecast/market_state/reanchored_price_results/portfolio_snapshot_equilibrium.json \
  --output-dir price_forecast/market_state/replay_results
```

Движок поддерживает отдельное исполнение ввода проданной и непроданной площади и текущий вес будущих продаж.

## Диагностика

```bash
python -m unittest discover -s cash_installments/tests -v
python -m unittest discover -s launches/tests -v
python cash_installments/scripts/compare_inertia.py
```

Дополнительная диагностика кеш-модели:

```bash
python -m pip install -r requirements-diagnostics.txt
python cash_installments/scripts/stationarity_diagnostics.py
```

- [Историческая проверка цен](../price_forecast/market_state/results/backtest_metrics.csv).
- [Календарь исторических эпизодов](../price_forecast/shock_normalization/full_history_results/calendar.csv).
- [Параметры исходной N54](../price_forecast/market_state/coupled_results/parameters_used.json).

## Горизонт расчета

Основные расчетные формы работают до 2030 года. Продолжение по реальному ВВП на 2031–2035 годы описано в методологии и пока не представлено отдельным блоком кода.

[Вернуться на главную](../README.md)
