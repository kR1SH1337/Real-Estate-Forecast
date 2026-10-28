"""Read cached annual channel inputs from the user's saved XLSX; never edit it."""
from pathlib import Path
import argparse,copy,hashlib,json,math,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def extract(path):
 old=json.loads((ROOT/'inputs/channel_forecast_2026-10-27.json').read_text())
 with zipfile.ZipFile(path) as z:
  shared=[''.join(t.text or '' for t in item.iter('{'+NS['m']+'}t')) for item in ET.fromstring(z.read('xl/sharedStrings.xml'))]
  workbook=ET.fromstring(z.read('xl/workbook.xml'))
  rid=next(x for x in workbook.find('m:sheets',NS) if x.get('name')=='ДДУ-Продажи').get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
  rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(x.get('Target') for x in rels if x.get('Id')==rid)
  sheet=target.lstrip('/') if target.startswith('/') else 'xl/'+target
  cells={c.get('r'):c for c in ET.fromstring(z.read(sheet)).findall('m:sheetData/m:row/m:c',NS)}
 def value(ref,blank=False):
  cell=cells.get(ref)
  v=cell.find('m:v',NS) if cell is not None else None
  if v is None or v.text is None:
   if blank and (cell is None or cell.find('m:f',NS) is None):return 0.
   raise ValueError('Missing cached value: '+ref)
  if cell.get('t')=='e':raise ValueError('Excel error: '+ref+' '+v.text)
  if cell.get('t')=='s':return shared[int(v.text)]
  x=float(v.text);assert math.isfinite(x);return x
 new=copy.deepcopy(old);new.update(source_file=Path(path).name,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),source_value_mode='Saved Excel cached values; no workbook recalculation or external-link refresh performed')
 mapping={'mortgage_transactions_bln_rub':43,'cash_model_real_bln_rub':50,'oneoff_real_bln_rub':51,'cash_total_thousand_m2':63,'reference_annual_price_thousand_rub_m2':55,'reference_sales_thousand_m2':61,'reference_mortgage_thousand_m2':62}
 for item,col in zip(new['annual'],'EFGHI'):
  assert value(col+'42')==item['year'] and value(col+'60')==item['year']
  for field,row in mapping.items():item[field]=value(f'{col}{row}',blank=(row==51))
  assert math.isclose(item['cash_total_thousand_m2'],1000*(item['cash_model_real_bln_rub']+item['oneoff_real_bln_rub'])/new['base_cash_price_thousand_rub_m2'],rel_tol=1e-12)
  assert math.isclose(item['reference_sales_thousand_m2'],item['cash_total_thousand_m2']+item['reference_mortgage_thousand_m2'],rel_tol=1e-12)
  assert math.isclose(item['reference_mortgage_thousand_m2'],1000*item['mortgage_transactions_bln_rub']/item['reference_annual_price_thousand_rub_m2'],rel_tol=1e-12)
 for field,row in [('mortgage_thousand_m2',4),('cash_thousand_m2',5),('mortgage_transactions_bln_rub',24)]:
  observed=sum(value(col+str(row)) for col in ['AL','AM','AN','AO','AP','AQ','AR','AS','AT'])
  assert math.isclose(observed,new['actual_2026_9m'][field],rel_tol=1e-12)
 assert math.isclose(value('D55'),new['price_anchor_2025_mean_thousand_rub_m2'],rel_tol=1e-12)
 assert math.isclose(1000*value('AT11')/value('AT3'),new['price_anchor_september_2026_thousand_rub_m2'],rel_tol=1e-12)
 for item in new['annual'][:3]:
  columns=[ref[:-1] for ref in cells if ref.endswith('1') and ref[:-1].isalpha() and value(ref)==item['year']]
  assert len(columns)==12
  assert math.isclose(sum(value(col+'24') for col in columns),item['mortgage_transactions_bln_rub'],rel_tol=1e-12)
 new['source_cells'].update(actual_mortgage_area='AL4:AT4',actual_cash_area='AL5:AT5',actual_mortgage_nominal='AL24:AT24',anchor_2025='D55',anchor_september_2026='1000*AT11/AT3')
 return new
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('xlsx',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'inputs/channel_forecast_2026-10-28.json');args=parser.parse_args()
 data=extract(args.xlsx);args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(data['annual'],ensure_ascii=False,indent=2))
