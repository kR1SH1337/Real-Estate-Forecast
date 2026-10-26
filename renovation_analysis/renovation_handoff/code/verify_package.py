"""Check integrity of the unpacked handoff. No network or third-party packages."""
import csv
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
with (root/'MANIFEST_SHA256.csv').open(encoding='utf-8-sig',newline='') as f:
    manifest=list(csv.DictReader(f,delimiter=';'))
bad=[]
for row in manifest:
    p=root/row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        bad.append(row['path'])
if bad:
    raise SystemExit('Changed or missing files: '+', '.join(bad))
print(f'OK: {len(manifest)} files match SHA-256 manifest.')
