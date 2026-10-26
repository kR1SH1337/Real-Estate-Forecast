"""Review all 18 previously unmatched sites against the whole saved snapshot.

The output is a search audit, not a certificate that a project is absent from
EIS or outside 214-FZ. No third-party dependencies or network are required.
"""
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'remaining_18_review_20261026'
OUT.mkdir(exist_ok=True)

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=';'))

def norm(s):
    s = unicodedata.normalize('NFKC', s).casefold().replace('ё','е')
    return re.sub(r'[^\w/]+', ' ', s).strip()

def distance(s, r):
    try:
        a,b,c,d = map(math.radians, [float(s['latitude']),float(s['longitude']),float(r['Широта']),float(r['Долгота'])])
        return 12742000 * math.asin(min(1, math.sqrt(math.sin((a-c)/2)**2 + math.cos(a)*math.cos(c)*math.sin((b-d)/2)**2)))
    except (ValueError, TypeError):
        return None

def compact(r, distance_m=None):
    return dict(eis_id=r['Корпус'], project=r['Проект'], developer=r['Застройщик'],
        apartments=r['Кол-во квартир по проекту'], area_m2=r['Жилая площадь'],
        status=r['Статус корпуса'], rns=r['Номер РНС'], cadastral=r['Кадастровый номер ЗУ'],
        distance_m=round(distance_m,1) if distance_m is not None else None)

rows = read(ROOT/'renovation_handoff/data/projects_flat.csv')
sites = {s['site_id']: s for s in read(ROOT/'construction_inventory_20261026/construction_sites.csv')}
review = json.loads((ROOT/'matching_review_20261026/review.json').read_text())
targets = [s for s in review['sites'] if s['status']=='NOT_MATCHED']
sources = {s['address']:s for s in json.loads((ROOT/'followup_18_sources.json').read_text())}
stems = ['нижнелих','амурск','амурск','черемушкин','буденн','дежнев','дежнев','кржижанов','матросов','нахимов','одесск','оренбург','рязанск','ставропольск','ставропольск','судаков','тюменск','шеремет']
assert len(targets)==len(stems)==len(sources)==18
assert len({r['Корпус'] for r in rows})==len(rows)
results=[]
for target,stem in zip(targets,stems):
    site=sites[target['site_id']]
    ranked=sorted([(distance(site,r),r) for r in rows if distance(site,r) is not None], key=lambda x:x[0])
    expected=target['fund_apartments']
    exact=[compact(r,m) for m,r in ranked if r['Кол-во квартир по проекту']==str(expected)]
    named=[compact(r,distance(site,r)) for r in rows if stem in norm(r['Проект'])]
    groups=defaultdict(list)
    for m,r in ranked:
        if m<=2000 and r['Номер РНС']:
            groups[r['Номер РНС']].append((m,r))
    sums=[]
    for permit,members in groups.items():
        if len(members)<2: continue
        total=sum(int(float(r['Кол-во квартир по проекту'] or 0)) for _,r in members)
        sums.append(dict(rns=permit,eis_ids=[r['Корпус'] for _,r in members],apartments_sum=total,
            equals_site_apartments=total==expected, caveat='Sum of saved IDs within 2 km by RNS; not a mapped construction site. Zero and missing counts can make sums incomplete.'))
    result=dict(site_id=target['site_id'],address=target['address'],fund_apartments=expected,
        matched_eis_ids=[],status='NOT_LOCATED_AFTER_EXTENDED_SEARCH',
        evidence=sources[target['address']],fund_url=target['fund_url'],
        nearest_candidates=[compact(r,m) for m,r in ranked[:10]],
        same_apartment_count_all_snapshot=exact,street_stem_candidates_all_snapshot=named,
        nearby_multiple_id_groups=sums,
        rns_from_fund_document=None,cadastral_from_fund_document=None,
        limitation='No verified permit/cadastral document was obtained for the fund site; absence from EIS and legal regime remain unproven.')
    results.append(result)

stats=dict(snapshot_unique_eis_ids=len(rows),reviewed_sites=len(results),new_confirmed_eis_ids=0,
    working_renovation_eis_ids_including_user_assumption=24,
    working_apartment_area_m2=379085.09,
    unresolved_sites=len(results),apartments_in_18_site_cards=sum(r['fund_apartments'] for r in results),
    sites_with_same_apartment_count_candidate_within_3km=sum(any(x['distance_m']<=3000 for x in r['same_apartment_count_all_snapshot']) for r in results))
payload=dict(as_of='2026-10-26',eis_snapshot='2026-10-23',statistics=stats,sites=results,
    web_search_note='Official fund/city pages were searched for every target; a separate EIS-domain query was run for every address. No new valid identity was established. Search-engine absence is not proof of registry absence. Direct article fetches often timed out.',
    related_but_different_project=dict(eis_id=66285,address='Оренбургская, 30',apartments=326,
        interpretation='The fund confirms renovation use of a 326-apartment house at Orenburgskaya 30. This contextual link does not map the 148-apartment site Orenburgskaya 26 and does not add an under-construction site to the working list.',
        source=sources['Оренбургская ул., з/у 26']['additional_url']))
(OUT/'review.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

md=['# Расширенная проверка 18 площадок', '',
    'Дата проверки — 26.10.2026. Выгрузка ЕИСЖС — 23.10.2026.', '',
    '**Новых подтверждённых сопоставлений: 0.** Рабочий список сохраняет 24 корпуса / 379,1 тыс. м² с учётом ранее принятого допущения по Петра Алексеева.', '',
    'Проверены все 18 площадок, для которых первоначальные соседи не подошли. По каждой просмотрены вся сохранённая выгрузка на 2680 уникальных ID, названия по основе улицы, одинаковая квартирность по всей Москве и расширенная пространственная выборка. '
    'Для многокорпусных проектов дополнительно проверены суммы квартир по общему РНС у соседних объектов (в пределах 2 км). Радиус 2 км служит дополнительной диагностикой, не заменяет поиск по всей выгрузке.', '',
    'У всех 18 площадок в радиусе 3 км нет единственного корпуса с точно совпадающей полной квартирностью. Это только диагностический результат: состав проекта может меняться, а площадка — делиться между ID.', '',
    'Дополнительно для каждого адреса выполнен поиск официальных карточек фонда, городских публикаций и индексированных страниц ЕИСЖС. Это не полная выгрузка живого реестра.', '',
    '## Что выяснилось', '',
    '- Амурская, 52: 24.09.2026 согласован проект на 158 квартир. Подтверждения конкретного ID, РНС и физического начала строительства не получено.',
    '- Дежнёва, 20 и 24: согласование проектов по 139 квартир опубликовано 13.09.2026; 18.10.2026 город анонсировал дома проекта «Московские кварталы». Ранее в сообщении было преждевременно сказано о подтверждённых разрешениях: их номера и даты этим этапом не установлены. Возможный лаг раскрытия ПД — гипотеза, не установленная причина отсутствия.',
    '- Будённого, 49/49А: город описывает стройку на 253 квартиры для проекта «Московские кварталы». Это полезный адресный вариант и связь с проектом фонда, но ID в нашей базе не найден.',
    '- Амурская, 1/12: проект из трёх корпусов; Матросова, 29/1: два корпуса; Судакова, 5: три корпуса. Поиск одного ID с полной квартирностью площадки недостаточен.',
    '- Оренбургская, 30 (ID 66285, 326 квартир): фонд отдельно подтверждает связь этого соседнего дома с реновацией. Он не является соответствием площадке Оренбургская, 26 на 148 квартир. Текущий рабочий перечень строящихся объектов из-за этой находки не расширен.', '',
    '## Все 18 площадок', '',
    '| Площадка | Квартир по карточке | Новые принятые ID | Основание / найденные сведения |',
    '|---|---:|---|---|']
for r in results:
    md.append(f"| {r['address']} | {r['fund_apartments']} | Не установлены | {r['evidence']['stage']} |")
md+=['', '## Источники и ближайшие альтернативы', '']
for r in results:
    md+=['### '+r['address'],'','[Карточка фонда]('+r['fund_url']+') · [Первичный источник]('+r['evidence']['url']+')']
    if r['evidence'].get('additional_url'):md+=['[Дополнительный первичный источник]('+r['evidence']['additional_url']+')']
    md+=['','| ID | Название в базе | Квартир | Расстояние, м | РНС |','|---|---|---:|---:|---|']
    for c in r['nearest_candidates'][:3]:
        md.append(f"| {c['eis_id']} | {c['project']} | {c['apartments']} | {c['distance_m']} | {c['rns']} |")
    nearest=r['same_apartment_count_all_snapshot'][:1]
    md+=['', ('Ближайший корпус с полной квартирностью площадки: ID '+nearest[0]['eis_id']+', '+nearest[0]['project']+', '+str(nearest[0]['distance_m'])+' м; как совпадение не принят.' if nearest else 'Корпус с полной квартирностью площадки по всей сохранённой выгрузке не найден.'),'']
md+=['## Границы результата и дальнейшая проверка','',
     '**Все 18 остаются со статусом «не найдены после расширенного поиска», а не «доказанно вне ЕИСЖС/214-ФЗ».** '
     'Полные РНС/кадастровые документы самих площадок не получены; соответствующие поля JSON оставлены null. Это недостающая часть документальной проверки, а не отсутствие документов у объектов.', '',
     'Нельзя рассчитывать площадь этого сегмента из числа квартир без отдельной оценки средней площади. Нельзя объявлять эти 18 площадок полным внешним сегментом: 128 других площадок пока остаются за рамками этого этапа.', '',
     'Для окончательного решения нужны разрешительные документы с номером РНС, кадастровым участком и составом корпусов, затем сопоставление этих ключей с выгрузкой и живым реестром. Для трёх свежих согласований отдельно проверить дату появления ПД. '
     'Реестр результатов и альтернатив: [review.json](review.json).', '',
     'Воспроизведение локальных проверок и отчёта:', '', '```bash','python renovation_analysis/check_remaining_18.py','```','']
(OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
