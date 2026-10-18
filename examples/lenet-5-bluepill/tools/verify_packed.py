#!/usr/bin/env python3
"""Verify the packed Case3 SD directory from its SHA256 manifest."""
import sys, json, hashlib
from pathlib import Path
folder=Path(sys.argv[1] if len(sys.argv)>1 else 'sd_int8_packed')
man=json.loads((folder/'pack_manifest.json').read_text())
assert len(man['parameter_files'])==14
assert sum(item['size_bytes'] for item in man['parameter_files'].values())==64302
assert {p.name for p in folder.glob('*.bin')}==set(man['parameter_files'])
for name,record in man['parameter_files'].items():
    data=(folder/name).read_bytes()
    assert len(data)==record['size_bytes']
    assert hashlib.sha256(data).hexdigest()==record['sha256'],name
print('PASS: 14 INT8 SD files, 64,302 bytes, matching SHA256')
