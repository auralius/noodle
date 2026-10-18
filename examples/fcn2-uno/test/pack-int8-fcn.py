#!/usr/bin/env python3
"""Pack FCN bias/multiplier/shift into 12-byte records, preserving bit patterns.

Usage: python pack-int8-fcn.py path/to/export/int8 path/to/export/int8/packed
No extra dependencies. Copies weights and interleaves the original LE int32
values byte-for-byte; does not requantize or change weight layout.
"""
import argparse
import shutil
import sys
import zlib
from pathlib import Path

EXPECTED = {1: (64, 256), 2: (10, 64)}  # outputs, inputs for Case 1


def pack_layer(src: Path, dest: Path, index: int, outputs: int, inputs: int) -> None:
    name = f'{index:02d}'
    weight = src / f'w{name}.bin'
    parts = [(src / f'{p}{name}.bin').read_bytes() for p in 'bms']
    weight_size = weight.stat().st_size
    if weight_size != outputs * inputs:
        raise ValueError(f'{weight}: expected {outputs * inputs} bytes, got {weight_size}')
    for p, data in zip('bms', parts):
        if len(data) != 4 * outputs:
            raise ValueError(f'{p}{name}.bin: expected {4 * outputs} bytes, got {len(data)}')
    packed = b''.join(b''.join(part[4*i:4*i+4] for part in parts)
                      for i in range(outputs))
    assert len(packed) == 12 * outputs
    for i in range(outputs):
        for k, part in enumerate(parts):
            assert packed[12*i+4*k:12*i+4*k+4] == part[i*4:i*4+4]
    target = dest / f'p{name}.bin'
    target.write_bytes(packed)
    shutil.copyfile(weight, dest / weight.name)
    print(f'{weight.name} {weight_size:,} bytes; {target.name} {len(packed):,} bytes; '
          f'packed CRC32={zlib.crc32(packed):08X}')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', type=Path, help='Directory holding w/b/m/s*.bin')
    p.add_argument('output_dir', type=Path, help='Directory to write w01,p01,w02,p02')
    args = p.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        p.error('Use a DIFFERENT output directory to preserve the original 8 files')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (outputs, inputs) in EXPECTED.items():
        pack_layer(args.input_dir, args.output_dir, index, outputs, inputs)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(1)
