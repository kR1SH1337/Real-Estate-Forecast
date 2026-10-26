"""Read values from XLSX with Python's standard library; does not modify files."""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def read_sheet(path, sheet_number=1):
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS):
                shared.append(''.join(t.text or '' for t in si.findall('.//s:t', NS)))
        root = ET.fromstring(z.read(f'xl/worksheets/sheet{sheet_number}.xml'))
        rows = []
        for row in root.findall('s:sheetData/s:row', NS):
            cells = {}
            for c in row.findall('s:c', NS):
                col = re.match(r'[A-Z]+', c.attrib['r']).group()
                pos = 0
                for ch in col:
                    pos = pos * 26 + ord(ch) - 64
                kind = c.get('t')
                v = c.find('s:v', NS)
                value = v.text if v is not None and v.text is not None else ''
                if kind == 's':
                    value = shared[int(value)] if value else ''
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in c.findall('.//s:t', NS))
                cells[pos - 1] = value
            rows.append((int(row.attrib['r']), cells))
        return rows

def records(path, sheet_number=1, header_row=1):
    rows = read_sheet(path, sheet_number)
    header = next(c for r, c in rows if r == header_row)
    result = []
    for r, cells in rows:
        if r <= header_row or not cells.get(0):
            continue
        result.append({name: cells.get(i, '') for i, name in header.items() if name})
    return result
