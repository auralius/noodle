#!/usr/bin/env python3
"""Compare two Case 3 FP32 sender logs by image index (100-image benchmark)."""
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def parse(path):
    out = {}
    record = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith('HOST idx='):
            if record is not None:
                out[record['idx']] = record
            parts = line.split()
            record = {'idx': int(parts[1].split('=')[1]),
                      'gt': int(parts[2].split('=')[1]),
                      'pixel_crc32': parts[3].split('=')[1],
                      'stages': {}}
        elif record is not None:
            bits = line.split()
            if not bits:
                continue
            if bits[0] == 'PRED':
                record['pred'], record['time'] = int(bits[1]), float(bits[2])
            elif bits[0] == 'CRC32':
                record['crc32'] = bits[1]
            elif bits[0] in ('LOGITS_F', 'SOFTMAX_F'):
                record[bits[0]] = [float(x) for x in bits[1:]]
            elif bits[0] == 'STAGE':
                record['stages'][bits[1]] = int(bits[2])
    if record is not None:
        out[record['idx']] = record
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('trial', type=Path)
    args = p.parse_args()
    a, b = parse(args.baseline), parse(args.trial)
    keys = sorted(set(a) & set(b))
    print(f'Images: baseline={len(a)} trial={len(b)} common={len(keys)}')
    if len(keys) != len(a) or len(keys) != len(b):
        print('WARNING: image index sets differ')
    for field in ['gt', 'pixel_crc32', 'crc32', 'pred']:
        mismatch = [k for k in keys if a[k].get(field) != b[k].get(field)]
        print(f'{field}: {len(mismatch)} mismatches' + (f' (first {mismatch[:5]})' if mismatch else ''))
    mean_a = statistics.mean(a[k]['time'] for k in keys)
    mean_b = statistics.mean(b[k]['time'] for k in keys)
    print(f'Mean: baseline {mean_a:.6f} s; trial {mean_b:.6f} s; delta {(mean_b-mean_a)*1000:+.3f} ms ({(mean_b/mean_a-1)*100:+.3f}%)')
    print('Stages:')
    names = sorted(set().union(*(a[k]['stages'] for k in keys), *(b[k]['stages'] for k in keys)))
    for name in names:
        if all(name in a[k]['stages'] and name in b[k]['stages'] for k in keys):
            am = statistics.mean(a[k]['stages'][name] for k in keys)/1000
            bm = statistics.mean(b[k]['stages'][name] for k in keys)/1000
            print(f'  {name:15s} {am:10.3f} -> {bm:10.3f} ms ({bm-am:+.3f} ms)')
    for field in ['LOGITS_F', 'SOFTMAX_F']:
        diffs=[]
        missing=[]
        for k in keys:
            lhs,rhs=a[k].get(field),b[k].get(field)
            if lhs is None or rhs is None or len(lhs)!=len(rhs):
                missing.append(k)
            else:
                diffs.extend(abs(x-y) for x,y in zip(lhs,rhs))
        if diffs:
            print(f'{field}: max abs diff={max(diffs):.9g}; values compared={len(diffs)}; missing/unequal vectors={len(missing)}')
        else:
            print(f'{field}: unavailable')

if __name__ == '__main__':
    main()
