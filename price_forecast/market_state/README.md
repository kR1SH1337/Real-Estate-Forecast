# Совместное решение цен, продаж и портфеля

## Актуальный расчет

**Окончательная функция:** N54 / growth54. **Ценовая привязка:** факт сентября 2026. **Сценарий в сравнительных файлах:** growth54_september_2026.

- [Методика и итог](REANCHORED_PRICE_EQUILIBRIUM.md).
- [Годовая таблица](reanchored_price_results/annual_equilibrium.csv).
- [Месячные траектории](reanchored_price_results/monthly_paths.csv).
- [Согласованный портфельный снимок](reanchored_price_results/portfolio_snapshot_equilibrium.json).
- [Проверки сходимости](reanchored_price_results/convergence_checks.csv).
- [Инструкция запуска](../../docs/REPRODUCE.md).

## Расчетные файлы

| Файл | Роль |
|---|---|
| [reanchored_price_equilibrium.py](reanchored_price_equilibrium.py) | Запуск выбранного совместного расчета и сохранение сравнительных вариантов |
| [coupled_forecast.py](coupled_forecast.py) | Базовая логика согласования |
| [linked_forecast.py](linked_forecast.py) | Перевод годовых продаж и остатков в месячные факторы и ценовую траекторию |
| [portfolio_engine.py](portfolio_engine.py) | Баланс классов, запуски, вводы и проданные площади |
| [model.py](model.py) | Оценивание исторических ценовых зависимостей и условные проверки |

Параметры выбранной N54 сохранены в [coupled_results/parameters_used.json](coupled_results/parameters_used.json). Месячный фактор — логарифм непроданного остатка относительно средних продаж шести завершенных месяцев, с лагом.

## Проверки и ограничения

[Исторические ошибки](results/backtest_metrics.csv) · [Календарь эпизодов](../shock_normalization/full_history_results/calendar.csv) · [Диагностика переоценок](statistical_comparison_results/README.md).

Статистика переоценок N57 и модель уровня относятся к другим оценкам. Их коэффициенты и метрики нельзя автоматически переносить на N54. Сходимость совместной системы показывает согласованность формул, а не подтвержденную прогнозную точность.

## История исследования

[Хронология, альтернативные факторы и эксперименты](HISTORY.md). Сохраненные этапы доступны для проверки принятых решений. Текущий прогноз начинается со ссылок первого раздела.

[На главную](../../README.md) · [Краткая методология](../../docs/METHODOLOGY.md)
