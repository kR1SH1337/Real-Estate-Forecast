# Воспроизведение и техническое изучение

## Актуальный совместный расчет

Требуется Python 3.10+. Команды выполняются из корня репозитория:

```bash
python -m pip install -r requirements.txt
python -m pip install -r price_forecast/requirements.txt
python price_forecast/market_state/reanchored_price_equilibrium.py
```

Скрипт использует сохраненные агрегированные входы ипотечного канала и кешевых площадей. Он не обновляет автоматически исходную ипотечную Excel-модель или кеш-модель.

Итог выбранной функции **growth54_september_2026**:

| Файл | Назначение |
|---|---|
| [annual_equilibrium.csv](../price_forecast/market_state/reanchored_price_results/annual_equilibrium.csv) | Годовой итог выбранной модели |
| [portfolio_snapshot_equilibrium.json](../price_forecast/market_state/reanchored_price_results/portfolio_snapshot_equilibrium.json) | Согласованные входы для повтора портфельного баланса |
| [monthly_paths.csv](../price_forecast/market_state/reanchored_price_results/monthly_paths.csv) | Месячные траектории всех вариантов; фильтр по scenario = growth54_september_2026 |
| [convergence_checks.csv](../price_forecast/market_state/reanchored_price_results/convergence_checks.csv) | Проверки сходимости |
| [annual_comparison.csv](../price_forecast/market_state/reanchored_price_results/annual_comparison.csv) | Сравнение функций и ценовых привязок, включая исследовательские |

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

Самостоятельный [portfolio_forecast](../portfolio_forecast/README.md) содержит базовый снимок предыдущего этапа. Для воспроизведения **окончательного согласованного баланса** используйте движок совместной системы:

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
- [Сравнение моделей кеша](../cash_installments/docs/inertia_comparison_2026-10-23.md).
- [Диагностика переоценок ценовых функций](../price_forecast/market_state/statistical_comparison_results/README.md). Ее N57-оценки не следует приписывать коэффициентам N54.

## Статус архивов

price_forecast/model.py, сценарии с поправкой κ и таблицы прежнего самостоятельного портфеля сохранены для истории. Команда запуска такого этапа не воспроизводит текущий согласованный прогноз.

[История ценового раздела](../price_forecast/HISTORY.md) · [История исследований состояния рынка](../price_forecast/market_state/HISTORY.md).

Основные расчетные формы работают до 2030 года. Продолжение по реальному ВВП на 2031–2035 годы описано в методологии и пока не представлено отдельным блоком кода.

[Вернуться на главную](../README.md)
