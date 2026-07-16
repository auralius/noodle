#!/usr/bin/env python3
"""
Sender for the Arduino Uno FCN-16 serial benchmark.

Input CSV format:
    label,1x1,1x2,...,28x28
    5,0,0,...

The script:
1. Loads 28x28 MNIST rows from CSV.
2. Resizes each image to 16x16.
3. Sends exactly 256 uint8 pixels to the Uno.
4. Reads:
       PRED <label> <seconds> <confidence>
       MEM <fields...>
       READY
5. Reports per-sample and aggregate accuracy/timing.

Dependencies:
    pip install numpy pandas pyserial pillow
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import serial
from PIL import Image


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_CSV = "mnist_train.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send resized MNIST images to the Arduino Uno FCN-16 demo."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_CSV),
        help=f"MNIST CSV file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Serial port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=100,
        help="Number of images to test (default: 100)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting row index when not using --random (default: 0)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Choose random rows instead of consecutive rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used with --random (default: 1234)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Serial read timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--boot-wait",
        type=float,
        default=2.0,
        help="Delay after opening serial so the Uno can reset (default: 2)",
    )
    parser.add_argument(
        "--resize",
        choices=("bilinear", "nearest", "box"),
        default="bilinear",
        help="28x28 to 16x16 resize method (default: bilinear)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert pixels before transmission",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Optional binary threshold in the range 0..255",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Optional folder for saving transmitted 16x16 images",
    )
    return parser.parse_args()


def load_mnist_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if df.shape[1] < 785:
        raise ValueError(
            f"Expected at least 785 columns (label + 784 pixels), got {df.shape[1]}"
        )

    # Prefer a named label column, otherwise use the first column.
    label_col = "label" if "label" in df.columns else df.columns[0]
    labels = df[label_col].to_numpy(dtype=np.uint8)

    pixel_df = df.drop(columns=[label_col])
    pixels = pixel_df.iloc[:, :784].to_numpy(dtype=np.uint8)

    if pixels.shape[1] != 784:
        raise ValueError(f"Expected 784 pixel values per row, got {pixels.shape[1]}")

    images = pixels.reshape((-1, 28, 28))
    return labels, images


def resize_to_16(
    image_28: np.ndarray,
    method: str,
    invert: bool,
    threshold: Optional[int],
) -> np.ndarray:
    pil = Image.fromarray(image_28.astype(np.uint8), mode="L")

    resampling = {
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
    }[method]

    image_16 = np.asarray(
        pil.resize((16, 16), resample=resampling),
        dtype=np.uint8,
    )

    if invert:
        image_16 = 255 - image_16

    if threshold is not None:
        if not 0 <= threshold <= 255:
            raise ValueError("--threshold must be between 0 and 255")
        image_16 = np.where(image_16 >= threshold, 255, 0).astype(np.uint8)

    return image_16


def read_text_line(ser: serial.Serial) -> Optional[str]:
    raw = ser.readline()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def wait_for_ready(ser: serial.Serial, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if line is None:
            continue

        print(f"[rx] {line!r}")

        if line == "READY":
            return

    raise TimeoutError("Timed out waiting for READY")


def parse_pred(line: str) -> Tuple[int, float, float]:
    parts = line.split()
    if len(parts) != 4 or parts[0] != "PRED":
        raise ValueError(f"Malformed PRED line: {line!r}")

    return int(parts[1]), float(parts[2]), float(parts[3])


def parse_mem(line: str) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for token in line.split()[2:]:
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value

    return result


def receive_result(
    ser: serial.Serial,
    timeout_s: float,
) -> Tuple[int, float, float, Dict[str, str]]:
    deadline = time.monotonic() + timeout_s
    pred: Optional[Tuple[int, float, float]] = None
    mem: Dict[str, str] = {}

    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if line is None:
            continue

        print(f"[rx] {line!r}")

        if line.startswith("ERR"):
            raise RuntimeError(f"Uno returned: {line}")

        if line.startswith("PRED "):
            pred = parse_pred(line)
            continue

        if line.startswith("MEM "):
            mem = parse_mem(line)
            continue

        if line == "READY":
            if pred is None:
                raise RuntimeError("READY received before a PRED result")
            return pred[0], pred[1], pred[2], mem

    raise TimeoutError("Timed out waiting for PRED/MEM/READY")


def choose_indices(
    total: int,
    count: int,
    start: int,
    random_mode: bool,
    seed: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("--count must be positive")

    if random_mode:
        rng = random.Random(seed)
        count = min(count, total)
        return rng.sample(range(total), count)

    end = min(start + count, total)
    if start < 0 or start >= total:
        raise ValueError(f"--start must be between 0 and {total - 1}")
    return list(range(start, end))


def main() -> int:
    args = parse_args()

    labels, images = load_mnist_csv(args.csv)
    indices = choose_indices(
        total=len(labels),
        count=args.count,
        start=args.start,
        random_mode=args.random,
        seed=args.seed,
    )

    if args.save_preview is not None:
        args.save_preview.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(labels)} rows from {args.csv}")
    print(f"Opening serial {args.port} @ {args.baud} ...")

    correct = 0
    times: list[float] = []

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.25,
        write_timeout=2.0,
    ) as ser:
        time.sleep(args.boot_wait)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        wait_for_ready(ser, timeout_s=max(args.timeout, 8.0))

        for position, idx in enumerate(indices, start=1):
            gt = int(labels[idx])
            image_16 = resize_to_16(
                images[idx],
                method=args.resize,
                invert=args.invert,
                threshold=args.threshold,
            )

            payload = image_16.reshape(-1).tobytes()
            if len(payload) != 256:
                raise RuntimeError(f"Payload size is {len(payload)}, expected 256")

            if args.save_preview is not None:
                Image.fromarray(image_16, mode="L").resize(
                    (256, 256),
                    Image.Resampling.NEAREST,
                ).save(args.save_preview / f"{position:04d}_idx{idx}_gt{gt}.png")

            ser.write(payload)
            ser.flush()

            pred, elapsed_s, confidence, mem = receive_result(
                ser,
                timeout_s=args.timeout,
            )

            ok = pred == gt
            correct += int(ok)
            times.append(elapsed_s)

            mem_text = ""
            if mem:
                mem_text = (
                    f" arena={mem.get('arena_used', '?')}/"
                    f"{mem.get('arena_capacity', '?')} B"
                    f" free_ram={mem.get('free_ram', '?')} B"
                )

            print(
                f"[{position}/{len(indices)}] "
                f"idx={idx} gt={gt} pred={pred} "
                f"{'OK' if ok else 'WRONG'} "
                f"time={elapsed_s:.4f}s conf={confidence:.6f}"
                f"{mem_text}"
            )

    accuracy = correct / len(indices)
    times_np = np.asarray(times, dtype=np.float64)

    print()
    print("Summary")
    print(f"  Samples:       {len(indices)}")
    print(f"  Correct:       {correct}")
    print(f"  Accuracy:      {accuracy * 100.0:.2f}%")
    print(f"  Mean time:     {times_np.mean():.4f} s")
    print(f"  Std. dev.:     {times_np.std(ddof=1) if len(times_np) > 1 else 0.0:.4f} s")
    print(f"  Minimum time:  {times_np.min():.4f} s")
    print(f"  Maximum time:  {times_np.max():.4f} s")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
