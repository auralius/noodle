#!/usr/bin/env python3
"""Independent Case-1 INT8 FCN reference, using the exact SD binaries.

Dependencies: numpy, pillow (no TensorFlow required).
Example:
  python uno_int8_host_reference.py --csv mnist_train.csv \
      --sd-dir ./sd --quant-header ./include/model_quant.h --count 5

This uses Noodle's current integer multiplier/shift arithmetic; it does NOT
retrain, export, or modify the model. Inputs use the Uno sender's Pillow
28x28 -> 16x16 bilinear preprocessing.
"""
from __future__ import annotations

import argparse
import csv
import re
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

I32_MIN, I32_MAX = -(2**31), 2**31 - 1


def sat32(x: int) -> int:
    return min(I32_MAX, max(I32_MIN, x))


def trunc_div(a: int, divisor: int) -> int:
    return abs(a) // divisor * (-1 if a < 0 else 1)


def multiply_by_quantized_multiplier(x: int, m: int, shift: int) -> int:
    """Mirror noodle_i8_multiply_by_quantized_multiplier(), not floats."""
    left = max(0, shift)
    right = max(0, -shift)
    shifted = sat32(x * (1 << left)) if left < 31 else (I32_MAX if x >= 0 else I32_MIN)
    if shifted == I32_MIN and m == I32_MIN:
        high = I32_MAX
    else:
        prod = shifted * m
        nudge = (1 << 30) if prod >= 0 else 1 - (1 << 30)
        high = trunc_div(prod + nudge, 1 << 31)
    if right <= 0:
        return high
    if right > 31:
        return -1 if high < 0 else 0
    mask = (1 << right) - 1
    remainder = high & mask
    threshold = (mask >> 1) + (1 if high < 0 else 0)
    return (high >> right) + (1 if remainder > threshold else 0)


def const(header: str, layer: int, key: str, as_int: bool = False):
    name = f'layer{layer:02d}_{key}'
    m = re.search(r'\b' + re.escape(name) + r'\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*[fF]?\s*;', header)
    if not m:
        raise ValueError(f'Missing {name} in model_quant.h')
    return int(m.group(1)) if as_int else float(m.group(1))


def layer_info(header: str, number: int) -> dict:
    return {
        'input_scale': const(header, number, 'input_scale'),
        'input_zero_point': const(header, number, 'input_zero_point', True),
        'output_scale': const(header, number, 'output_scale'),
        'output_zero_point': const(header, number, 'output_zero_point', True),
        'activation_min': const(header, number, 'activation_min', True),
        'activation_max': const(header, number, 'activation_max', True),
    }


def load_layer(folder: Path, number: int, din: int, dout: int):
    wp = folder / f'w{number:02d}.bin'
    pp = folder / f'p{number:02d}.bin'
    wbytes, pbytes = wp.read_bytes(), pp.read_bytes()
    if len(wbytes) != din * dout or len(pbytes) != dout * 12:
        raise ValueError(f'Layer {number}: expected {din*dout} weight bytes and {dout*12} metadata bytes; found {len(wbytes)} and {len(pbytes)}')
    w = np.frombuffer(wbytes, dtype='i1').astype(np.int64).reshape(dout, din)
    metadata = np.frombuffer(pbytes, dtype='<i4').astype(np.int64).reshape(dout, 3)
    print(f'L{number} w{number:02d}.bin={len(wbytes)} B CRC32={zlib.crc32(wbytes):08X} '
          f'p{number:02d}.bin={len(pbytes)} B CRC32={zlib.crc32(pbytes):08X}')
    return w, metadata


def fcn(x: np.ndarray, w: np.ndarray, p: np.ndarray, cfg: dict) -> np.ndarray:
    x32 = x.astype(np.int64) - cfg['input_zero_point']

    if w.shape == (64, 256):
        for stop in (64, 128, 192, 256):
            partial = int(w[0, :stop] @ x32[:stop])
            print("HOST N0_PART", stop, partial)

    accum = w @ x32

    if w.shape == (64, 256):
        print("HOST N0",
            int(accum[0]) + int(p[0, 0]),
            *map(int, p[0]))
    
    y = np.empty(w.shape[0], dtype=np.int8)
    for i in range(len(y)):
        bias, mult, shift = map(int, p[i])
        q = multiply_by_quantized_multiplier(sat32(int(accum[i]) + bias), mult, shift)
        q += cfg['output_zero_point']
        q = max(-128, min(127, cfg['activation_max'], max(cfg['activation_min'], q)))
        y[i] = q
    return y


def pixels_from_csv(path: Path, start: int, count: int):
    with path.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        label_pos = header.index('label') if 'label' in header else 0
        pix_cols = [j for j in range(len(header)) if j != label_pos][:784]
        if len(pix_cols) != 784:
            raise ValueError(f'CSV requires 784 pixel columns, found {len(pix_cols)}')
        sent = 0
        for row_idx, row in enumerate(reader):
            if row_idx < start:
                continue
            if sent >= count:
                break
            img28 = np.array([int(row[j]) for j in pix_cols], dtype=np.uint8).reshape(28, 28)
            img16 = np.asarray(Image.fromarray(img28).resize((16, 16), Image.Resampling.BILINEAR), dtype=np.uint8)
            yield row_idx, int(row[label_pos]), img16
            sent += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, required=True, help='Exactly the CSV used by sender_uno_fcn16.py')
    ap.add_argument('--sd-dir', type=Path, required=True, help='Folder with V7 w01.bin,p01.bin,w02.bin,p02.bin')
    ap.add_argument('--quant-header', type=Path, required=True, help='Same model_quant.h compiled into Uno')
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--start', type=int, default=0)
    args = ap.parse_args()
    h = args.quant_header.read_text()
    c1, c2 = layer_info(h, 1), layer_info(h, 2)
    if c1['output_zero_point'] != c2['input_zero_point'] or abs(c1['output_scale']-c2['input_scale']) > 1e-5:
        raise ValueError('Quantization parameters between FCN1 and FCN2 do not match')
    w1, p1 = load_layer(args.sd_dir, 1, 256, 64)
    w2, p2 = load_layer(args.sd_dir, 2, 64, 10)
    print('L1:', c1)
    print('L2:', c2)
    print('---')
    matches = 0
    for idx, gt, img in pixels_from_csv(args.csv, args.start, args.count):
        qin = np.clip(np.rint(img.astype(np.float64) / c1['input_scale']) + c1['input_zero_point'], -128, 127).astype(np.int8).reshape(-1)
        print("HOST X128", qin[128:144].tolist())
        print("HOST X192", qin[192:208].tolist())
        hidden = fcn(qin, w1, p1, c1)
        logits = fcn(hidden, w2, p2, c2)
        pred = int(np.argmax(logits))
        matches += (pred == gt)
        print(f'idx={idx} gt={gt} pred={pred} input_sum={int(img.sum())} nonzero={int(np.count_nonzero(img))}')
        print(f'  hidden: zero={int(np.sum(hidden == c1["output_zero_point"]))} saturated={int(np.sum(hidden == 127))} '
              f'min={int(hidden.min())} max={int(hidden.max())}')
        print('  FCN2_LOGITS_Q', ' '.join(map(str, logits.astype(int).tolist())))
        if idx == args.start:
            print('  FCN1_HIDDEN_Q', ' '.join(map(str, hidden.astype(int).tolist())))
    print(f'Host inference correct: {matches}/{args.count}')
    print('Compare FCN1_HIDDEN_Q and FCN2_LOGITS_Q against matching Uno samples, not TFLite softmax outputs.')

if __name__ == '__main__':
    main()
