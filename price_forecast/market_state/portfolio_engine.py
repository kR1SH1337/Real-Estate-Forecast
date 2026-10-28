"""Reproduce Moscow class portfolio forecast using only the Python standard library."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def forecast(data):
    classes = data['classes']
    a = data['assumptions']
    sales = a['sales']
    years = list(range(2026, 2031))
    profiles = data['profiles']
    if len(classes) != 3 or len(sales) != 5:
        raise ValueError('Expected three classes and five forecast years')
    if any(abs(sum(p) - 1) > 1e-9 or p[0] != 0 for p in profiles):
        raise ValueError('Completion profiles must sum to one, with zero same-year input')
    if abs(sum(a['sales_by_completion_lag']) - 1) > 1e-9:
        raise ValueError('Sales allocation weights must sum to one')
    intensity = []
    prior_shares = []
    for k in classes:
        hist = [h for h in data['history'] if h[0] == k]
        intensity.append(sum(h[2] / (h[3] * (1 - h[4])) for h in hist) / len(hist))
        prior_shares.append(hist[-1][2])
    prior_shares = [x / sum(prior_shares) for x in prior_shares]
    targets = a['launch_targets_by_class_2026']
    weights = [x / sum(targets) for x in targets]
    launches = [a['launches_2026']]
    future_sales_weight = float(a.get('launch_future_sales_weight', 0.5))
    if not 0 <= future_sales_weight <= 1:
        raise ValueError('Future-sales weight must be between zero and one')
    for t in range(1, 5):
        next_sales = sales[t + 1] if t < 4 else a['sales_2031']
        speed = a['launch_adjustment_speed']
        desired_sales = (1 - future_sales_weight) * sales[t] + future_sales_weight * next_sales
        launches.append((1 - speed) * launches[-1] + speed * a['launch_sales_ratio'] * desired_sales)
    remaining_total = launches[0] - sum(data['accounted'])
    gaps = [max(0, targets[i] - data['accounted'][i]) for i in range(3)]
    if remaining_total < -1e-8 or (remaining_total > 0 and sum(gaps) == 0):
        raise ValueError('Annual launch target conflicts with launches already in the matrix')
    remaining = [remaining_total * g / sum(gaps) if sum(gaps) else 0 for g in gaps]
    states = [{y: [0., 0.] for y in range(2026, 2036)} for _ in classes]
    for k, y, area, sold in data['cohorts']:
        if sold < -1e-8 or sold > area + 1e-8:
            raise ValueError('Sold area is outside portfolio capacity')
        states[classes.index(k)][y] = [area, sold]
    adjustment = [0.] * 3
    adjustment_sold = [0.] * 3
    result = []
    for t, year in enumerate(years):
        unsold = [sum(v[0] - v[1] for v in states[i].values()) + adjustment[i] - adjustment_sold[i]
                  if t else data['jan2026'][i][0] * (1 - data['jan2026'][i][1]) for i in range(3)]
        scores = [unsold[i] * intensity[i] for i in range(3)]
        if sum(scores) <= 0:
            raise ValueError('No initial unsold inventory for sales decomposition')
        inertia = a['sales_share_inertia']
        shares = [inertia * prior_shares[i] + (1 - inertia) * scores[i] / sum(scores) for i in range(3)]
        for i, k in enumerate(classes):
            opening = sum(v[0] for v in states[i].values()) + adjustment[i]
            new_launches = remaining[i] if not t else launches[t] * weights[i]
            for yy in range(year, min(year + 6, 2036)):
                states[i][yy][0] += new_launches * profiles[i][yy - year]
            annual_sales = sales[t] * shares[i]
            period_sales = ((annual_sales - data['actual9'][i]) * a['q4_remaining_fraction']
                            if not t else annual_sales)
            if period_sales < -1e-8:
                raise ValueError('Annual class sales below observed nine-month actual')
            capacity = [max(0, states[i][yy][0] - states[i][yy][1]) for yy in range(year, year + 5)]
            allocated = [0.] * 5
            for _ in range(5):
                eligible = [a['sales_by_completion_lag'][j] if capacity[j] - allocated[j] > 1e-9 else 0 for j in range(5)]
                denominator = sum(eligible)
                leftover = max(0, period_sales - sum(allocated))
                for j in range(5):
                    allocated[j] += min(max(0, capacity[j] - allocated[j]), leftover * eligible[j] / denominator) if denominator else 0
            for j in range(5):
                states[i][year + j][1] += allocated[j]
            due = states[i][year][:]
            delay = a['completion_delays'][t]
            completions = due[0] * (1 - delay)
            completed_sold = due[1] * (1 - delay)
            states[i][year] = [0., 0.]
            states[i][year + 1][0] += due[0] * delay
            states[i][year + 1][1] += due[1] * delay
            correction = opening * a['stock_correction_rate'] if t else 0
            pre_area = sum(v[0] for v in states[i].values()) + adjustment[i]
            pre_sold = sum(v[1] for v in states[i].values()) + adjustment_sold[i]
            adjustment[i] += correction
            adjustment_sold[i] += correction * pre_sold / pre_area if pre_area else 0
            closing = sum(v[0] for v in states[i].values()) + adjustment[i]
            sold = sum(v[1] for v in states[i].values()) + adjustment_sold[i]
            balance_error = closing - (opening + new_launches - completions + correction)
            if abs(balance_error) > 1e-7 or sold < -1e-7 or sold > closing + 1e-7:
                raise ValueError('Portfolio balance or sold-area check failed')
            result.append(dict(year=year, housing_class=k,
                               launches_annual=new_launches + data['accounted'][i] if not t else new_launches,
                               sales_annual=annual_sales,
                               completions_annual=completions + data['fact'][i] if not t else completions,
                               portfolio_opening_period=opening, launches_period=new_launches,
                               sales_period=period_sales, completions_period=completions,
                               portfolio_correction=correction, portfolio_closing=closing,
                               sold_closing=sold, sellout=sold / closing if closing else 0,
                               unsold_closing=closing - sold,
                               unallocated_sales=max(0, period_sales - sum(allocated)),
                               sold_exiting_at_completion=completed_sold,
                               balance_error=balance_error))
        prior_shares = shares
    for t, year in enumerate(years):
        rr = [r for r in result if r['year'] == year]
        if abs(sum(r['sales_annual'] for r in rr) - sales[t]) > 1e-7:
            raise ValueError('Class sales do not reconcile with aggregate sales')
        total = {key: sum(r[key] for r in rr) for key in rr[0] if key not in ['year', 'housing_class', 'sellout']}
        total.update(year=year, housing_class='Всего', sellout=total['sold_closing'] / total['portfolio_closing'])
        result.append(total)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json', type=Path, default=ROOT / 'inputs/snapshot_2026-10-27.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(encoding='utf8'))
    rows = forecast(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'forecast_by_class.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'rows': len(rows), 'max_balance_error': max(abs(r['balance_error']) for r in rows),
                      'max_unallocated_sales': max(r['unallocated_sales'] for r in rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
