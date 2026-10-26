"""Reproduce the bulk-sales screen offline. Python 3.10+, no dependencies.

Default input contains ONLY the 37 previously selected buildings, not all Moscow.
For a new full-market screen supply the SQL export with --sales PATH.
"""
import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from xlsx_reader import records

ROOT = Path(__file__).resolve().parents[1]

def num(value):
    return float(value) if value not in (None, '') else 0.0

def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))

def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sales', type=Path, default=ROOT/'data/monthly_selected.csv')
    p.add_argument('--projects', type=Path, default=ROOT/'data/2026-10-23-project_data.xlsx')
    p.add_argument('--output', type=Path, default=ROOT/'reproduced')
    args = p.parse_args()
    meta = {str(int(num(x['Корпус']))): x for x in records(args.projects)}
    history = defaultdict(list)
    for s in load_csv(args.sales):
        s['building'] = str(int(num(s['building'])))
        history[s['building']].append(s)
    events, unmatched, duplicates = [], set(), []
    for building, rows in history.items():
        if building not in meta:
            unmatched.add(building)
            continue
        m = meta[building]
        if m.get('ИНН застройщика') == '7703434808':
            continue  # Fund as developer is NOT fund as purchaser.
        area, qty = num(m.get('Жилая площадь')), num(m.get('Кол-во квартир по проекту'))
        rows.sort(key=lambda s: s['period'])
        seen = set()
        for j, s in enumerate(rows):
            if s['period'] in seen:
                duplicates.append([building, s['period']])
            seen.add(s['period'])
            bought, units = num(s.get('res_bought_m2')), num(s.get('res_bought_qty'))
            area_share = bought/area if area else 0
            qty_share = units/qty if qty else 0
            if not ((bought >= 5000 or units >= 100) and (area_share >= .5 or qty_share >= .5)):
                continue
            prev = rows[:j]
            last6 = prev[-6:]
            cancelled = num(s.get('res_cancel_m2'))
            events.append({
                'building': building, 'project': m.get('Проект', ''),
                'property_class': s.get('property_class') or m.get('Класс недвижимости', ''),
                'period': s['period'], 'bought_m2': bought, 'bought_qty': units,
                'project_area_snapshot_m2': area, 'project_qty_snapshot': qty,
                'area_share': area_share, 'qty_share': qty_share,
                'same_month_cancel_m2': cancelled, 'same_month_net_m2': bought-cancelled,
                'cancel_from_event_m2': sum(num(v.get('res_cancel_m2')) for v in rows[j:]),
                'first_observation': j == 0,
                'prior_sold_m2': sum(num(v.get('res_bought_m2')) for v in prev),
                'previous6_observations_avg_m2': sum(num(v.get('res_bought_m2')) for v in last6)/len(last6) if last6 else '',
                'buyer_status': 'NOT_ESTABLISHED_BY_SCREEN',
            })
    peaks = {}
    for e in events:
        if e['building'] not in peaks or e['bought_m2'] > peaks[e['building']]['bought_m2']:
            peaks[e['building']] = e
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'screen_events.csv', events)
    write_csv(args.output/'screen_buildings.csv', list(peaks.values()))
    # Reproduce the separate WORKING renovation-purchase series, not a buyer proof.
    purchases = load_csv(ROOT/'data/working_purchases_21.csv')
    market = load_csv(ROOT/'data/moscow_net_annual.csv')
    annual = []
    for row in market:
        amount = sum(num(v['net_m2']) for v in purchases if v['period'][:4] == row['year'])
        annual.append({'year':row['year'], 'coverage':row['coverage'],
                       'market_net_m2':num(row['net_m2']), 'working_linked_m2':amount,
                       'working_share':amount/num(row['net_m2']) if num(row['net_m2']) else '',
                       'meaning':'WORKING_ESTIMATE_NOT_CONFIRMED_FUND_SHARE'})
    write_csv(args.output/'working_share_annual.csv', annual)
    report = {'events':len(events), 'buildings':len(peaks),
              'input_buildings':len(history), 'unmatched_buildings':sorted(unmatched),
              'duplicate_building_months':duplicates,
              'scope':'selected_buildings_only' if args.sales.resolve()==(ROOT/'data/monthly_selected.csv').resolve() else 'user_supplied_export',
              'static_project_snapshot':'2026-10-23'}
    (args.output/'run_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
