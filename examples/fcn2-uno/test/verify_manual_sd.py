#!/usr/bin/env python3
"""Verify manual-copy of validated Noodle Case 1 INT8 files (never writes)."""
import argparse
import zlib
from pathlib import Path
EXPECTED={
    'w01.bin':(16384,'F3BD4266'),
    'p01.bin':(768,'D4D0FEFD'),
    'w02.bin':(640,'C2A3E4E4'),
    'p02.bin':(120,'C95850BA'),
}

def verify(root, expected=EXPECTED):
    ok=True
    for fn,(size,crc) in expected.items():
        file=root/fn
        if not file.is_file():
            print(fn, 'MISSING'); ok=False; continue
        raw=file.read_bytes(); got=f'{zlib.crc32(raw):08X}'
        valid=len(raw)==size and got==crc
        print(f'{fn:10} {len(raw):6} bytes CRC32={got} {"OK" if valid else "MISMATCH"}')
        ok &= valid
    return ok

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sd-dir',type=Path,required=True,help='Mounted SD card root or exact host copy')
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.sd_dir) else 1)
