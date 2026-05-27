#!/usr/bin/env python3
"""
Convert Noodle ASCII scalar text files to raw binary files.

Default float output is little-endian float32 (<f4), matching ESP32/STM32/RP2040.
Use this for weight/bias/tensor files that Noodle reads with NOODLE_FILE_FORMAT_BIN.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def read_text_scalars(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: cannot parse scalar {s!r}") from exc
    return values


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for v in values:
            f.write(struct.pack("<f", float(v)))


def convert_one(src: Path, dst: Path) -> tuple[int, int]:
    values = read_text_scalars(src)
    write_f32(dst, values)
    return len(values), dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input .txt file(s)")
    ap.add_argument("-o", "--output", help="Output .bin file for single input")
    ap.add_argument("--out-dir", help="Output directory for batch conversion")
    args = ap.parse_args()

    inputs = [Path(x) for x in args.inputs]

    if args.output and len(inputs) != 1:
        raise SystemExit("--output can only be used with one input file")

    for src in inputs:
        if args.output:
            dst = Path(args.output)
        elif args.out_dir:
            dst = Path(args.out_dir) / (src.stem + ".bin")
        else:
            dst = src.with_suffix(".bin")

        n, size = convert_one(src, dst)
        print(f"{src} -> {dst} : {n} float32 values, {size} bytes")


if __name__ == "__main__":
    main()
