#!/usr/bin/env python3
"""Send selected MNIST examples to the *instrumented FP32* Mega firmware.

Reuses the exact 28->16 Pillow preprocessing from sender_uno_fcn16.py.
Requires the existing sender script to remain beside this script.
Writes a self-describing JSON record for subsequent Colab/Keras comparison.
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import serial
from sender_uno_fcn16 import load_mnist_csv, resize_to_16


def args_parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, default=Path('tools/mnist_train.csv'))
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--indices', default='30,42,80', help='CSV image indices, comma-separated')
    ap.add_argument('--timeout', type=float, default=120.)
    ap.add_argument('--boot-wait', type=float, default=3.)
    ap.add_argument('--out', type=Path, default=Path('case2_fp32_numeric_probe.json'))
    return ap.parse_args()


def softmax(v):
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def capture_one(ser, timeout):
    result = {'stages_us': {}}
    vectors = {'FP32_GAP':24, 'FP32_LOGITS':10, 'FP32_PROBS':10}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = ser.readline().decode('utf-8',errors='replace').strip()
        if not line:
            continue
        print('[rx]',line)
        if line.startswith('ERR'):
            raise RuntimeError('Device error: '+line)
        p = line.split()
        if p[0] == 'PRED':
            if len(p) != 4: raise ValueError('Invalid PRED: '+line)
            result.update(pred=int(p[1]), latency_s=float(p[2]), confidence=float(p[3]))
        elif p[0] in vectors:
            need = vectors[p[0]]
            if len(p) != 1+need: raise ValueError(f'Expected {need} floats in {p[0]}, got {len(p)-1}')
            if p[0] in result: raise ValueError('Duplicate '+p[0])
            vals = [float(x) for x in p[1:]]
            if not np.isfinite(vals).all(): raise ValueError('Non-finite '+p[0])
            result[p[0]] = vals
        elif p[0] == 'STAGE' and len(p)==3:
            result['stages_us'][p[1]]=int(p[2])
        elif p[0]=='MEM' and len(p)>=2:
            result['mem']=' '.join(p[1:])
        elif line=='READY':
            missing=[k for k in ('pred','latency_s',*vectors) if k not in result]
            if missing: raise RuntimeError('Missing expected FP32 diagnostics: '+str(missing))
            computed=softmax(result['FP32_LOGITS'])
            result['host_softmax_vs_mcu_probs_max_abs']=float(np.max(np.abs(computed-result['FP32_PROBS'])))
            result['host_softmax_pred']=int(np.argmax(computed))
            result['fp32_probs_pred']=int(np.argmax(result['FP32_PROBS']))
            if result['pred'] != result['fp32_probs_pred']:
                raise RuntimeError('PRED disagrees with FP32_PROBS argmax')
            return result
    raise TimeoutError('No READY+FP32 diagnostics within timeout')


def main():
    args = args_parse()
    inds = [int(x) for x in args.indices.split(',') if x.strip()]
    if not inds or len(set(inds))!=len(inds):
        raise ValueError('--indices must be distinct comma-separated integers')
    labels, imgs = load_mnist_csv(args.csv)
    if min(inds)<0 or max(inds)>=len(labels):
        raise ValueError('One or more indices out of range')
    recs=[]
    print('Input:',args.csv,'images:',len(labels),'indices:',inds)
    with serial.Serial(args.port,args.baud,timeout=0.25,write_timeout=3.0) as ser:
        time.sleep(args.boot_wait)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print('Skipping initial READY (matching current Mega firmware)')
        for idx in inds:
            raw=resize_to_16(imgs[idx],method='bilinear',invert=False,threshold=None).reshape(-1).tobytes()
            assert len(raw)==256
            crc=f'{zlib.crc32(raw):08X}'
            print(f'\nSending idx={idx} gt={int(labels[idx])} pixel_crc32={crc}')
            ser.write(raw)
            ser.flush()
            r=capture_one(ser,args.timeout)
            r.update(index=idx,gt=int(labels[idx]),pixel_crc32=crc)
            recs.append(r)
            print('Result pred=',r['pred'],'time=',r['latency_s'],
                  'host-softmax vs device max abs=',r['host_softmax_vs_mcu_probs_max_abs'])
    args.out.write_text(json.dumps({'format':'case2_fp32_probe_v1','samples':recs},indent=2)+'\n')
    print('\nSaved',args.out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
