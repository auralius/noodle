#!/usr/bin/env python3
"""Compare two serial benchmark captures using identical sample indices.

Works with the sender's lines:
  [3/100] idx=2 gt=4 pred=4 OK time=12.3456s ...
Does not equate FP32 logits with quantized INT8 logits.
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r'\bidx=(\d+)\b.*?\bgt=(\d+)\b.*?\bpred=(\d+)\b.*?\btime=([0-9.]+)s')


def parse(path):
    rows = {}
    for line in path.read_text(errors='replace').splitlines():
        match = LINE.search(line)
        if not match:
            continue
        index, gt, pred, seconds = match.groups()
        index = int(index)
        if index in rows:
            raise ValueError(f'{path}: duplicate sample index {index}')
        rows[index] = (int(gt), int(pred), float(seconds))
    if not rows:
        raise ValueError(f'{path}: no sender summary records found')
    return rows


def summary(label, rows):
    times = [v[2] for v in rows.values()]
    correct = sum(int(v[0] == v[1]) for v in rows.values())
    print(f'{label}: n={len(rows)}; correct={correct}; accuracy={correct/len(rows):.2%}; '
          f'mean={statistics.mean(times):.4f}s; min={min(times):.4f}s; max={max(times):.4f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp32-log', type=Path, required=True)
    ap.add_argument('--int8-log', type=Path, required=True)
    ap.add_argument('--expected-count', type=int, default=100)
    args = ap.parse_args()
    fp32 = parse(args.fp32_log)
    int8 = parse(args.int8_log)
    if set(fp32) != set(int8):
        only_fp32 = sorted(set(fp32) - set(int8))
        only_int8 = sorted(set(int8) - set(fp32))
        raise ValueError(f'NOT EQUIVALENT DATASETS: only FP32={only_fp32[:20]}; '
                         f'only INT8={only_int8[:20]}')
    if len(fp32) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} matching indices, found {len(fp32)}')
    if any(fp32[i][0] != int8[i][0] for i in fp32):
        raise ValueError('GT label differs for at least one index')
    summary('FP32', fp32)
    summary('INT8', int8)
    differences = [i for i in sorted(fp32) if fp32[i][1] != int8[i][1]]
    print('Prediction disagreements:', len(differences), 'indices:', differences[:40])
    mean_fp32 = statistics.mean(v[2] for v in fp32.values())
    mean_int8 = statistics.mean(v[2] for v in int8.values())
    print('INT8/FP32 mean-time ratio:', f'{mean_int8/mean_fp32:.4f}x')
    print('Matched data indices:', min(fp32), 'through', max(fp32))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
