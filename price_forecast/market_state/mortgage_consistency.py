"""Read-only audit of saved mortgage workbook inputs and their price-model bridge."""
from pathlib import Path
import argparse,hashlib,json,re,zipfile,xml.etree.ElementTree as E
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mortgage_consistency_results'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def colnum(c):
 n=0
 for k in c:n=26*n+ord(k)-64
 return n
def colname(n):
 s=''
 while n:n,k=divmod(n-1,26);s=chr(k+65)+s
 return s
def translate(formula,origin,target):
 oc,orr=re.fullmatch(r'([A-Z]+)(\d+)',origin).groups();tc,tr=re.fullmatch(r'([A-Z]+)(\d+)',target).groups()
 dc=colnum(tc)-colnum(oc);dr=int(tr)-int(orr)
 def repl(m):
  a,c,b,r=m.groups();return a+(c if a else colname(colnum(c)+dc))+b+(r if b else str(int(r)+dr))
 return re.sub(r'(?<![A-Za-z0-9_])([$]?)([A-Z]{1,3})([$]?)(\d+)',repl,formula)
def read(path):
 with zipfile.ZipFile(path) as z:
  shared=[''.join(t.text or '' for t in item.iter('{'+NS['m']+'}t')) for item in E.fromstring(z.read('xl/sharedStrings.xml'))]
  rels={r.get('Id'):r.get('Target') for r in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
  sheets={}
  for item in E.fromstring(z.read('xl/workbook.xml')).find('m:sheets',NS):
   name=item.get('name')
   if name not in ['ДДУ-Продажи','Ипотека и ДДУ']:continue
   target=rels[item.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
   cells={};masters={}
   for c in E.fromstring(z.read(target)).findall('m:sheetData/m:row/m:c',NS):
    ref=c.get('r');v=c.find('m:v',NS);f=c.find('m:f',NS);val=v.text if v is not None else None
    if c.get('t')=='s' and val is not None:val=shared[int(val)]
    cells[ref]=dict(value=val,formula=f.text if f is not None else None,shared=f.get('si') if f is not None else None,type=c.get('t'))
    if f is not None and f.text and f.get('si') is not None:masters[f.get('si')]=(ref,f.text)
   for ref,c in cells.items():
    if c['shared'] is not None and c['formula'] is None:
     origin,formula=masters[c['shared']];c['formula']=translate(formula,origin,ref)
   sheets[name]=cells
 return sheets

def main(path):
 OUT.mkdir(exist_ok=True);before=hashlib.sha256(path.read_bytes()).hexdigest();sheets=read(path);x=sheets['Ипотека и ДДУ'];d=sheets['ДДУ-Продажи']
 def v(col,row):
  cell=x.get(f'{col}{row}',{});assert cell.get('type')!='e'
  return float(cell.get('value') or 0)
 rows=[];checks=[]
 for year,col in zip(range(2025,2031),['AN','BA','BN','CA','CB','CC']):
  assert float(x[col+'1']['value'])==year
  rows.append(dict(year=year,source_column=col,total_loan_bln=v(col,23),ddu_loan_bln=v(col,25),family_ddu_bln=v(col,28),market_ddu_bln=v(col,31),ddu_count_thousand=v(col,4),family_ddu_count_thousand=v(col,7),market_ddu_count_thousand=v(col,10),family_avg_loan_mln=v(col,47),market_avg_loan_mln=v(col,50),ddu_avg_loan_mln=v(col,45),market_rate_pct=100*v(col,66),family_rate_pct=100*v(col,63),family_secondary_bln=v(col,35),preferential_secondary_bln=v(col,34),secondary_loan_bln=v(col,32)))
  for label,lhs,rhs in [('DDU count components',v(col,4),v(col,6)+v(col,10)),('DDU loan components',v(col,25),v(col,27)+v(col,31)),('DDU family identity',v(col,28),v(col,7)*v(col,47)),('DDU market identity',v(col,31),v(col,10)*v(col,50)),('DDU average loan identity',v(col,25),v(col,4)*v(col,45)),('secondary preferential sum',v(col,34),sum(v(col,r) for r in [35,36,37])),('all-market rollup',v(col,23),v(col,25)+v(col,32))]:
   checks.append(dict(year=year,check=label,lhs=lhs,rhs=rhs,difference=lhs-rhs,passed=bool(np.isclose(lhs,rhs,atol=1e-7,rtol=1e-10))))
 annual=pd.DataFrame(rows).set_index('year');annual.to_csv(OUT/'mortgage_annual.csv')
 pd.DataFrame(checks).to_csv(OUT/'identity_checks.csv',index=False)
 eq=pd.read_csv(ROOT/'updated_channels_results/annual_equilibrium.csv').set_index('year')
 bridge=[]
 for year,col in zip(range(2027,2031),['BN','CA','CB','CC']):
  ycol=colname(year-2026+5);nominal=float(d[ycol+'43']['value']);loans=annual.loc[year,'ddu_loan_bln'];pv=1-loans/nominal
  assert np.isclose(nominal,eq.loc[year,'mortgage_transactions_bln_rub'])
  bridge.append(dict(year=year,ddu_loan_bln=loans,transaction_bln_including_downpayment=nominal,implied_downpayment_share=pv,model_mean_price_thousand_rub=eq.loc[year,'price_denominator_thousand_rub_m2'],mortgage_area_thousand_m2=eq.loc[year,'mortgage_thousand_m2'],ddu_count_thousand=annual.loc[year,'ddu_count_thousand'],implied_area_per_transaction_m2=eq.loc[year,'mortgage_thousand_m2']/annual.loc[year,'ddu_count_thousand'],excel_mean_price_thousand_rub=float(d[ycol+'55']['value']),excel_assumed_mean_price_growth_pct=100*float(d[ycol+'54']['value'])))
 bridge=pd.DataFrame(bridge).set_index('year');bridge['mortgage_area_growth_pct']=bridge.mortgage_area_thousand_m2.pct_change()*100
 bridge['model_mean_price_growth_pct']=bridge.model_mean_price_thousand_rub.pct_change()*100
 bridge.to_csv(OUT/'price_bridge.csv')
 important=['CC34','CC35','CC33','CC52','CA103','CB103','CC103','CA88','CB88','CC88','CA56','CB17','CB10','CC10','CM6']
 (OUT/'formula_evidence.json').write_text(json.dumps({ref:x[ref] for ref in important},ensure_ascii=False,indent=2))
 findings=dict(source_file=path.name,source_sha256=before,source_value_mode='Saved values and expanded shared formulas; workbook not recalculated or edited.',confirmed_secondary_2030_excess_bln=v('CC',34)-sum(v('CC',r) for r in [35,36,37]),reported_total_2030_bln=v('CC',23),total_2030_if_only_secondary_rollup_corrected_bln=v('CC',23)-v('CC',34)+sum(v('CC',r) for r in [35,36,37]),direct_price_input_affected=False,price_conditional_mortgage='Family loan budget growth in 2028-2030 references assumed mean-price growth in Excel; frozen in coupled price model, not fed back.',market_count_response='CM6=15 implies +15% market DDU loan count per 1pp rate decrease in annual recurrence; expert parameter, not estimated by this audit.',sample_average_note='CA56 is volume-weighted average monthly loan size, not aggregate amount/count. It does not affect primary mortgage input directly.',family_average_year_references='CB103=BO103*1.05 and CC103=BP103*1.05 refer to Jan/Feb2028, not prior annual columns; intent requires review.',no_external_feasibility_validation=True)
 (OUT/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2))
 assert hashlib.sha256(path.read_bytes()).hexdigest()==before
 print(annual.round(4).to_string());print('\nFAILED IDENTITIES\n'+pd.DataFrame(checks).query('not passed').to_string(index=False));print('\nBRIDGE\n'+bridge.round(4).to_string());print(json.dumps(findings,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('xlsx',type=Path);main(p.parse_args().xlsx)
