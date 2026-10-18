#!/usr/bin/env python3
"""Noodle Case 1 reproducible serial benchmark (inference only; no SD upload).

Examples:
  python benchmark_case1.py --port /dev/ttyACM0 --csv mnist_train.csv \
      --count 100 --split train --precision int8 --out results/int8_train100 \
      --sd-dir /mnt/sdcard --quant-header ../include/model_quant.h

  python benchmark_case1.py --port /dev/ttyACM0 --csv mnist_test.csv \
      --count 1000 --split test --precision int8 --out results/int8_test1000 \
      --sd-dir /mnt/sdcard --quant-header ../include/model_quant.h

Raw serial evidence, per-sample CSV and summary JSON are written to --out.
The script NEVER writes to the SD card; --sd-dir is a *host-side* mirror
of the mounted/manual-copy SD contents for bit-exact INT8 host comparison.
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import zlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from uno_int8_host_reference import layer_info, load_layer, fcn, pixels_from_csv


def load_reference(sd_dir: Path, quant_header: Path):
    hdr = quant_header.read_text()
    c1, c2 = layer_info(hdr, 1), layer_info(hdr, 2)
    if c1['output_zero_point'] != c2['input_zero_point'] or abs(c1['output_scale'] - c2['input_scale']) > 1e-5:
        raise ValueError('Layer quantization continuity mismatch')
    w1,p1 = load_layer(sd_dir,1,256,64)
    w2,p2 = load_layer(sd_dir,2,64,10)
    return (c1,c2,w1,p1,w2,p2)


def reference_logits(img, ref):
    c1,c2,w1,p1,w2,p2 = ref
    qin = np.clip(np.rint(img.astype(np.float64)/c1['input_scale']) + c1['input_zero_point'], -128,127).astype(np.int8).reshape(-1)
    return [int(a) for a in fcn(fcn(qin,w1,p1,c1), w2,p2,c2).tolist()]


def parse_serial_result(lines):
    out = {'pred': None, 'elapsed_s': None, 'confidence': None, 'logits_q': None,
           'guard_min_gap': None, 'guard_collision': None, 'hash_ok': None,
           'free_ram': None, 'arena_used': None, 'error_lines': []}
    for s in lines:
        if s.startswith('ERR '): out['error_lines'].append(s)
        if s.startswith('PRED '):
            p = s.split()
            if len(p) >= 4:
                out.update(pred=int(p[1]), elapsed_s=float(p[2]), confidence=float(p[3]))
        elif s.startswith('DBG LOGITS_Q '):
            out['logits_q'] = [int(x) for x in s.split()[2:]]
        elif s.startswith('DBG V17_HASH ') or s.startswith('DBG V16_HASH '):
            p = s.split()
            # DBG V17_HASH I a b c H d e
            if len(p) >= 9 and p[2] == 'I' and p[6] == 'H':
                out['hash_ok'] = (p[3] == p[4] == p[5] and p[7] == p[8])
        elif s.startswith('GUARD '):
            vals=dict(re.findall(r'(\w+)=(-?\d+)',s))
            if 'min_gap' in vals: out['guard_min_gap'] = int(vals['min_gap'])
            if 'collision' in vals: out['guard_collision'] = int(vals['collision'])
        elif s.startswith('MEM '):
            vals=dict(re.findall(r'(\w+)=(-?\d+)',s))
            if 'free_ram' in vals: out['free_ram']=int(vals['free_ram'])
            if 'arena_used' in vals: out['arena_used']=int(vals['arena_used'])
    return out


def collect_until(ser, marker: str, timeout_s: float, log):
    lines=[]
    deadline=time.monotonic()+timeout_s
    while time.monotonic() < deadline:
        line=ser.readline().decode('ascii',errors='replace').strip()
        if not line:
            continue
        log.write(f'{time.monotonic():.6f} {line}\n'); log.flush()
        lines.append(line)
        if line == marker: return lines
    raise TimeoutError(f'Timed out waiting for {marker!r}. Recent lines: {lines[-12:]}')


def run(args):
    try: import serial
    except ImportError as e: raise RuntimeError('Install pyserial: python -m pip install pyserial') from e
    if args.count <= 0 or args.start < 0: raise ValueError('count must be positive and start nonnegative')
    if bool(args.sd_dir) != bool(args.quant_header):
        raise ValueError('--sd-dir and --quant-header must be supplied together')
    if args.precision != 'int8' and args.sd_dir:
        raise ValueError('SD-backed bit-exact reference is currently INT8 only; omit --sd-dir for FP32')
    ref = load_reference(args.sd_dir,args.quant_header) if args.sd_dir else None
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'Output folder is not empty: {out}')
    params={}
    if args.sd_dir:
        for name in ['w01.bin','p01.bin','w02.bin','p02.bin']:
            p=args.sd_dir/name
            buf=p.read_bytes(); params[name]={'bytes':len(buf), 'crc32':f'{zlib.crc32(buf):08X}'}
    rows=[]
    firmware='unknown'
    started=datetime.now(timezone.utc).isoformat()
    with (out/'serial_raw.log').open('w', encoding='utf-8') as raw:
        with serial.Serial(args.port,args.baud,timeout=0.25,write_timeout=5) as ser:
            incoming=collect_until(ser,'READY',args.boot_timeout,raw)
            firmware=next((l[5:] for l in incoming if l.startswith('BOOT ')), 'unknown')
            print('Firmware:',firmware)
            for idx, gt, img in pixels_from_csv(args.csv,args.start,args.count):
                data=img.astype('uint8').reshape(-1).tobytes()
                assert len(data)==256
                ser.write(data); ser.flush()
                lines=collect_until(ser,'READY',args.timeout,raw)
                result=parse_serial_result(lines)
                exp=reference_logits(img,ref) if ref else None
                eq=(result['logits_q']==exp) if exp is not None else None
                expected_pred=int(np.argmax(exp)) if exp is not None else None
                correct=(result['pred']==gt) if result['pred'] is not None else False
                ok=(result['pred'] is not None and not result['error_lines']
                    and result['hash_ok'] is not False and result['guard_collision'] != 1
                    and eq is not False)
                row={'idx':idx,'split':args.split,'gt':gt,'pred':result['pred'],
                     'correct':int(correct),'valid':int(ok), 'elapsed_s':result['elapsed_s'],
                     'confidence':result['confidence'], 'logits_q':json.dumps(result['logits_q']),
                     'reference_logits_q':json.dumps(exp),'host_pred':expected_pred,
                     'logits_match':None if eq is None else int(eq),
                     'hash_ok':None if result['hash_ok'] is None else int(result['hash_ok']),
                     'guard_min_gap':result['guard_min_gap'],'guard_collision':result['guard_collision'],
                     'free_ram':result['free_ram'],'arena_used':result['arena_used'],
                     'error_lines':' | '.join(result['error_lines'])}
                rows.append(row)
                label='OK' if correct else 'WRONG'
                eqs='' if eq is None else f' logits_match={eq}'
                print(f'[{len(rows)}/{args.count}] idx={idx} gt={gt} pred={result["pred"]} {label} '
                      f't={result["elapsed_s"]}s valid={ok}{eqs}')
                if not ok and not args.continue_on_error:
                    print('Stopped at first integrity/output failure. Raw serial log preserved.')
                    break
                if args.inter_sample_ms: time.sleep(args.inter_sample_ms/1000)
    fields=['idx','split','gt','pred','correct','valid','elapsed_s','confidence','logits_q',
            'reference_logits_q','host_pred','logits_match','hash_ok',
            'guard_min_gap','guard_collision','free_ram','arena_used','error_lines']
    with (out/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    timings=[r['elapsed_s'] for r in rows if r['valid'] and r['elapsed_s'] is not None]
    eq_coverage=sum(r['logits_match'] is not None for r in rows)
    data={'timestamp_utc':started,'firmware':firmware, 'board':'Arduino Uno R3',
          'precision':args.precision,'serial_baud':args.baud,'dataset':str(args.csv),
          'dataset_split':args.split,'start_index':args.start,'requested_count':args.count,
          'completed_count':len(rows), 'valid_count':sum(r['valid'] for r in rows),
          'correct_count':sum(r['correct'] for r in rows),
          'accuracy_over_completed_pct':(100*sum(r['correct'] for r in rows)/len(rows)) if rows else None,
          'timing_definition':'device-reported preprocessing+FCN1+FCN2+softmax, before serial results; excludes UART transfer/SD preflight',
          'mean_s':statistics.mean(timings) if timings else None,
          'sd_s':statistics.stdev(timings) if len(timings)>1 else 0 if timings else None,
          'min_s':min(timings) if timings else None,
          'max_s':max(timings) if timings else None,
          'all_logits_match_reference':all(r['logits_match']==1 for r in rows) if eq_coverage == len(rows) and rows else None,
          'reference_compared':eq_coverage,
          'all_integrity_valid':all(r['valid'] for r in rows),
          'parameter_files':params,'activation_arena_bytes':330 if args.precision=='int8' else None,
          'note':'Accuracy on train is a training-subset pilot, not test-set generalization; benchmark vs FP32 only if timing definitions match.'}
    (out/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ('completed_count','valid_count','correct_count','accuracy_over_completed_pct','mean_s','sd_s','all_logits_match_reference')},indent=2))
    print('Saved:',out)
    if len(rows) < args.count or not data['all_integrity_valid']:
        return 2
    return 0


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port',required=True)
    p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--split',choices=['train','test','other'],required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--precision',choices=['int8','fp32'],default='int8')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sd-dir',type=Path,help='Directory mirroring manually copied INT8 SD files, for exact logit comparison')
    p.add_argument('--quant-header',type=Path,help='The same model_quant.h as the board firmware')
    p.add_argument('--boot-timeout',type=float,default=15)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--inter-sample-ms',type=int,default=0)
    p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args(argv)
    try: return run(a)
    except (OSError,ValueError,TimeoutError,FileExistsError,RuntimeError) as e:
        p.exit(1,f'ERROR: {e}\n')

if __name__=='__main__': sys.exit(cli())
