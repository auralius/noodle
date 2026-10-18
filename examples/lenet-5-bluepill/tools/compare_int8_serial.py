#!/usr/bin/env python3
"""Compare Blue Pill INT8 serial log with the Colab TFLite first100 reference.

Example:
python tools/compare_int8_serial.py case3_int8.log case3_lenet5_fp32_vs_int8_first100.csv

Matches frames by raw-pixel CRC32; recognizes PRED, CRC32, SOFTMAX_Q lines.
"""
import argparse
import csv
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=Path)
    ap.add_argument('reference', type=Path)
    args = ap.parse_args()
    refs = {}
    with args.reference.open(newline='') as f:
        for r in csv.DictReader(f):
            crc = r['pixel_crc32'].upper().zfill(8)
            refs.setdefault(crc, []).append(r)

    frames = []
    cur = None
    for line in args.log.read_text(errors='replace').splitlines():
        # Ignore ANSI coloring and optional monitor timestamps before record.
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        s = re.search(r'\bPRED\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', line)
        if s:
            cur = dict(pred=int(s.group(1)), seconds=float(s.group(2)),
                       confidence=float(s.group(3)))
            frames.append(cur)
            continue
        if cur is None:
            continue
        s = re.search(r'\bCRC32\s+([0-9a-fA-F]{8})', line)
        if s:
            cur['crc'] = s.group(1).upper()
        s = re.search(r'\bSOFTMAX_Q\s+([-\d ]+)', line)
        if s:
            q = [int(x) for x in s.group(1).split()]
            if len(q) == 10:
                cur['q'] = q
    checked = passed = predictions = 0
    durations = []
    for j, f in enumerate(frames):
        crc = f.get('crc')
        if not crc or crc not in refs:
            print(f'frame {j}: no matching first100 input CRC ({crc})')
            continue
        ref = refs[crc][0]
        checked += 1
        idx = int(ref['index'])
        expected_class = int(ref['int8_pred'])
        same_pred = f['pred'] == expected_class
        predictions += same_pred
        expected_q = [int(ref[f'int8_raw{k}']) for k in range(10)]
        q = f.get('q')
        equal_q = q == expected_q
        passed += equal_q and same_pred
        durations.append(f['seconds'])
        detail = '' if q is not None else ' SOFTMAX_Q missing'
        if q and not equal_q:
            detail = f' max_q_diff={max(abs(a-b) for a,b in zip(q,expected_q))}'
        print(f"ref_idx={idx:3} predicted={f['pred']} expected={expected_class} "
              f"pred_ok={same_pred} exact_q={equal_q} t={f['seconds']:.4f}s{detail}")
    print(f'Frames {len(frames)}; reference matched {checked}; '
          f'predictions {predictions}/{checked}; exact TFLite output {passed}/{checked}')
    if durations:
        print(f'Mean inference time: {sum(durations)/len(durations):.4f}s')
    if not frames or not checked:
        raise SystemExit('No reference-matched frames found; inspect IMG source and CRC32 lines')
    if passed != checked:
        raise SystemExit('Numerical differences detected: inspect LOGITS_Q, stage quantization and pooling')


if __name__ == '__main__':
    main()
