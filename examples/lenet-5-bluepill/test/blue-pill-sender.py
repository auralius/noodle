#!/usr/bin/env python3
"""Noodle Case Study 3: 28x28 raw-MNIST sender for STM32 Blue Pill FP32/INT8.

Protocol:
  host:   b'IMG'
  board:  RDYIMG\n
  host:   784 raw uint8 pixel bytes in 64-byte chunks
  board:  ACK\n after each chunk
  board:  PRED <class> <seconds> <confidence>, CRC32/SOFTMAX_Q (INT8), READY

No pixel normalization, rotation, inversion, or augmentation.
Example:
  python blue-pill-sender.py --csv mnist_train.csv --port /dev/ttyUSB0 --count 100 \\
      --reference case3_lenet5_fp32_vs_int8_first100.csv --log case3_int8.log

Install: python -m pip install pyserial
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import zlib
from pathlib import Path

IMG_BYTES = 28 * 28
CHUNK_BYTES = 64
PRED_PATTERN = re.compile(r"^PRED\s+(\d+)\s+([0-9.+\-eE]+)\s+([0-9.+\-eE]+)")


def rows_from_mnist(csv_path: Path, start: int, count: int):
    """Yield (zero-based dataset index, label, 784 raw bytes), ignoring CSV header."""
    end = start + count
    produced = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        dataset_index = 0
        for lineno, row in enumerate(reader, 1):
            if not row:
                continue
            if lineno == 1:
                try:
                    int(row[0])
                except ValueError:
                    continue  # Conventional MNIST label,pixel0,... header
            if dataset_index >= end:
                break
            if dataset_index >= start:
                if len(row) < IMG_BYTES + 1:
                    raise ValueError(f"CSV line {lineno}: expected label + 784 pixel fields, got {len(row)}")
                gt = int(row[0])
                vals = [int(s) for s in row[1:785]]
                if not 0 <= gt <= 9 or any(p < 0 or p > 255 for p in vals):
                    raise ValueError(f"CSV line {lineno}: invalid digit label or uint8 pixels")
                yield dataset_index, gt, bytes(vals)
                produced += 1
            dataset_index += 1
    if produced != count:
        raise ValueError(f"CSV only had {produced} requested samples from index {start}; needed {count}")


def load_reference(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = {int(r["index"]): r for r in csv.DictReader(f)}
    for row in rows.values():
        if "pixel_crc32" not in row:
            raise ValueError("Reference must include pixel_crc32")
    return rows


def read_line(ser, deadline: float):
    while time.monotonic() < deadline:
        raw = ser.readline()
        if raw:
            s = raw.decode("utf-8", errors="replace").strip()
            if s:
                return s
    raise TimeoutError("Serial line timeout")


def until_token(ser, token: str, seconds: float, writer=None):
    deadline = time.monotonic() + seconds
    while True:
        line = read_line(ser, deadline)
        if writer:
            writer(line)
        if line == token:
            return
        if line.startswith("ERR") or line.startswith("ERROR"):
            raise RuntimeError(f"Firmware reported: {line}")


def send_image(ser, payload: bytes, handshake_timeout: float):
    assert len(payload) == IMG_BYTES
    ser.write(b"IMG")
    ser.flush()
    until_token(ser, "RDYIMG", handshake_timeout)
    for offset in range(0, IMG_BYTES, CHUNK_BYTES):
        ser.write(payload[offset:offset + CHUNK_BYTES])
        ser.flush()
        until_token(ser, "ACK", handshake_timeout)


def receive_result(ser, deadline_s: float, log_line):
    deadline = time.monotonic() + deadline_s
    pred = None
    lines = []
    while True:
        line = read_line(ser, deadline)
        lines.append(line)
        log_line(line)
        m = PRED_PATTERN.match(line)
        if m:
            pred = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        if line == "READY":
            if pred is None:
                raise RuntimeError("Firmware signalled READY without PRED. Check ERR/STAGE lines in log")
            return pred, lines
        if line.startswith("ERR") or line.startswith("ERROR"):
            raise RuntimeError(f"Firmware reported: {line}")


def run(args):
    try:
        import serial
    except ImportError as e:
        raise SystemExit("Missing pyserial. Install with: python -m pip install pyserial") from e

    if not args.csv.is_file():
        raise SystemExit(f"MNIST CSV not found: {args.csv}")
    refs = load_reference(args.reference) if args.reference else {}
    rows = rows_from_mnist(args.csv, args.start, args.count)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    correct = matched_predictions = exact_q = reference_count = successful = 0
    inference_times = []
    print(f"Opening {args.port} at {args.baud} baud; source={args.csv}; no rotation/normalization")
    with args.log.open("w", buffering=1, encoding="utf-8") as logfile:
        def log_line(s):
            logfile.write(s + "\n")
            if args.verbose:
                print("  [mcu]", s, flush=True)

        with serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=10,
                           rtscts=False, dsrdtr=False) as ser:
            ser.dtr = False
            ser.rts = False
            # Wait for device's periodic READY while idle (every ~1s).
            until_token(ser, "READY", args.ready_timeout, log_line)
            for k, (idx, gt, pixels) in enumerate(rows, 1):
                crc = f"{zlib.crc32(pixels):08X}"
                ref = refs.get(idx)
                if ref and ref["pixel_crc32"].upper() != crc:
                    raise RuntimeError(
                        f"Input mismatch at index {idx}: source CRC={crc}, "
                        f"reference={ref['pixel_crc32']}. Ensure same MNIST training CSV."
                    )
                logfile.write(f"HOST idx={idx} gt={gt} pixel_crc32={crc}\n")
                try:
                    send_image(ser, pixels, args.handshake_timeout)
                    (pred, seconds, confidence), lines = receive_result(
                        ser, args.inference_timeout, log_line
                    )
                except (TimeoutError, RuntimeError, OSError) as e:
                    print(f"[{k}/{args.count}] idx={idx} ERROR {e}. See {args.log}", file=sys.stderr)
                    raise
                successful += 1
                correct += (pred == gt)
                inference_times.append(seconds)
                status = "OK" if pred == gt else "WRONG"
                match_text = ""
                if ref:
                    reference_count += 1
                    expected_pred = int(ref.get("int8_pred", pred)) if args.mode == "int8" else int(ref.get("fp32_pred", pred))
                    same_pred = (pred == expected_pred)
                    matched_predictions += same_pred
                    match_text = f" ref_pred={expected_pred} match={same_pred}"
                    if args.mode == "int8":
                        raw_line = next((s for s in lines if s.startswith("SOFTMAX_Q")), None)
                        if raw_line is not None:
                            actual_q = [int(v) for v in raw_line.split()[1:]]
                            expected_q = [int(ref[f"int8_raw{i}"]) for i in range(10)]
                            equal_q = actual_q == expected_q
                            exact_q += equal_q
                            match_text += f" exact_q={equal_q}"
                            if not equal_q and len(actual_q) == 10:
                                match_text += f" max_q_diff={max(abs(a-b) for a,b in zip(actual_q, expected_q))}"
                print(f"[{k:3}/{args.count}] idx={idx:3} gt={gt} pred={pred} {status} "
                      f"t={seconds:.4f}s conf={confidence:.4f} crc={crc}{match_text}", flush=True)
                if args.pause > 0:
                    time.sleep(args.pause)
    print(f"Completed {successful}/{args.count}; accuracy={correct}/{successful}; "
          f"mean inference={sum(inference_times)/len(inference_times):.4f}s")
    if reference_count:
        print(f"Reference matches={matched_predictions}/{reference_count}; "
              f"exact INT8 softmax={exact_q}/{reference_count} (when SOFTMAX_Q available)")
    print(f"Raw firmware output saved in {args.log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="MNIST training CSV: label followed by 784 pixels")
    p.add_argument("--port", default="/dev/ttyUSB0", help="Blue Pill USB-UART port")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mode", choices=["int8", "fp32"], default="int8")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--reference", type=Path, help="Optional case3_lenet5_fp32_vs_int8_first100.csv")
    p.add_argument("--log", type=Path, default=Path("case3_int8.log"))
    p.add_argument("--ready-timeout", type=float, default=20.0)
    p.add_argument("--handshake-timeout", type=float, default=10.0)
    p.add_argument("--inference-timeout", type=float, default=240.0,
                   help="Long timeout for SD-backed FP32 and INT8 inference")
    p.add_argument("--pause", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true", help="Print every firmware diagnostic line")
    args = p.parse_args()
    if args.start < 0 or args.count < 1:
        p.error("--start must be nonnegative and --count positive")
    run(args)


if __name__ == "__main__":
    main()
