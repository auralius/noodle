#!/usr/bin/env python3
"""Compare Mega 2560 channel means for B5/B6 with Keras from Colab.
Usage: python3 tools/compare_b5_b6_probe.py --hardware case2_fp32_b5b6_probe.log --keras case2_fp32_keras_b5_b6_reference.csv
"""
import argparse
import csv
import re
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--hardware',required=True,help='Output of targeted sender with [rx] lines')
p.add_argument('--keras',required=True)
a=p.parse_args()
with open(a.keras, newline='') as f:
    ref={int(r['idx']):r for r in csv.DictReader(f)}
with open(a.hardware,encoding='utf-8') as f:
    lines=f.read().splitlines()
measured={}
idx=None
for line in lines:
    m=re.search(r'Sending idx=(\d+)\s+gt=(\d+)\s+pixel_crc32=([0-9A-Fa-f]+)',line)
    if m:
        idx=int(m.group(1))
        assert idx not in measured,('repeated index',idx)
        measured[idx]={'gt':int(m.group(2)),'pixel_crc32':m.group(3).upper()}
        continue
    if idx is None:continue
    for tag,key in [('FP32_B5_MEAN','b5'),('FP32_B6_MEAN','b6')]:
        if f'{tag} ' in line:
            values=line.split(f'{tag} ',1)[1].strip().split()
            assert len(values)==24,(idx,key,len(values))
            measured[idx][key]=np.array([float(v) for v in values],dtype=np.float64)
    if 'FP32_GAP_RECOMPUTE_MAXERR ' in line:
        measured[idx]['gap_err']=float(line.split('FP32_GAP_RECOMPUTE_MAXERR ',1)[1].strip())
if not measured:raise SystemExit('No hardware records: ensure CASE2_FP32_B5B6_TRACE=1 and use targeted sender.')
for idx,row in measured.items():
    if idx not in ref:raise SystemExit(f'No Keras reference for index {idx}')
    t=ref[idx]
    assert int(t['gt'])==row['gt']
    assert t['pixel_crc32'].upper()==row['pixel_crc32'],f'Input CRC mismatch at {idx}'
    print(f'Image {idx} (gt={row["gt"]}):')
    for key in ('b5','b6'):
        assert key in row,f'Missing {key} at index {idx}'
        expected=np.array([float(t[f'{key}_mean{c}']) for c in range(24)])
        delta=row[key]-expected
        top=np.argsort(np.abs(delta))[-3:][::-1]
        print(f' {key.upper()} max_abs={np.max(np.abs(delta)):.8g} MAE={np.mean(np.abs(delta)):.8g} ' +
              'top_channels='+str([(int(c),round(float(delta[c]),6)) for c in top]))
    assert 'gap_err' in row,f'Missing GAP check at index {idx}'
    print(f' B6 mean vs device GAP max_abs={row["gap_err"]:.8g}')
