#!/usr/bin/env python3
"""
CIFAR-10 RGB sender for the RP2350 TinyResNet Noodle demo.

Protocol:

  PC  -> MCU: IMG
  MCU -> PC:  RDYIMG
  PC  -> MCU: 32*32*3 RGB uint8 bytes in 64-byte chunks
  MCU -> PC:  ACK after each chunk
  MCU -> PC:  PRED <class_id> <seconds> <confidence> <class_name>
  MCU -> PC:  READY

The RP2350 firmware performs:
  - uint8 [0,255] -> float [0,1]
  - per-channel CIFAR-10 normalization
  - TinyResNet inference

Therefore, this sender must transmit the original RGB uint8 image without
normalizing, transposing, or changing channel order.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from PIL import Image


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMG_W = 32
IMG_H = 32
IMG_C = 3
IMG_SIZE = IMG_W * IMG_H * IMG_C
CHUNK_SIZE = 64


def load_cifar10_local_or_download(
    cache_dir: Path,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      x_test_u8: uint8 array with shape (N, 32, 32, 3)
      y_test:    int64 array with shape (N,)

    Local cache:
      cache_dir/cifar10_test_images_u8.npy
      cache_dir/cifar10_test_labels.npy

    If the cache is absent, download the CIFAR-10 test split from
    Hugging Face dataset uoft-cs/cifar10.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_npy = cache_dir / "cifar10_test_images_u8.npy"
    lab_npy = cache_dir / "cifar10_test_labels.npy"

    if img_npy.exists() and lab_npy.exists():
        print(f"Loading local CIFAR-10 cache from {cache_dir}")

        x = np.load(img_npy)
        y = np.load(lab_npy)

        if x.dtype != np.uint8:
            raise ValueError(f"Expected uint8 images, got {x.dtype}")
        if x.ndim != 4 or x.shape[1:] != (IMG_H, IMG_W, IMG_C):
            raise ValueError(f"Unexpected image shape: {x.shape}")

        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if limit is not None:
            x = x[:limit]
            y = y[:limit]

        return x, y

    print("Local CIFAR-10 cache not found.")
    print("Downloading CIFAR-10 from Hugging Face: uoft-cs/cifar10")

    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "datasets"]
        )
        from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10")
    sample = ds["test"][0]
    image_key = "img" if "img" in sample else "image"

    images: list[np.ndarray] = []
    labels: list[int] = []

    for ex in ds["test"]:
        image = np.asarray(ex[image_key], dtype=np.uint8)

        if image.shape != (IMG_H, IMG_W, IMG_C):
            raise ValueError(f"Unexpected downloaded image shape: {image.shape}")

        images.append(image)
        labels.append(int(ex["label"]))

        if limit is not None and len(images) >= limit:
            break

    x = np.stack(images, axis=0).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int64)

    np.save(img_npy, x)
    np.save(lab_npy, y)

    png_dir = cache_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    n_png = min(len(x), 200)
    for i in range(n_png):
        label = int(y[i])
        name = CLASS_NAMES[label]
        Image.fromarray(x[i], mode="RGB").save(
            png_dir / f"test_{i:05d}_{label}_{name}.png"
        )

    print(f"Saved cache to {cache_dir}")
    print(f"Saved first {n_png} PNG images to {png_dir}")

    return x, y


def plot_init():
    plt.ion()

    fig, ax = plt.subplots(figsize=(3.2, 3.5), dpi=120)
    img0 = np.zeros((IMG_H, IMG_W, IMG_C), dtype=np.uint8)

    im = ax.imshow(img0, interpolation="nearest")
    ax.axis("off")

    title_obj = ax.set_title("Waiting for image...")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, im, title_obj


def plot_update(
    im,
    title_obj,
    img_u8: np.ndarray,
    title: str,
) -> None:
    im.set_data(img_u8)
    title_obj.set_text(title)
    im.figure.canvas.draw_idle()
    im.figure.canvas.flush_events()


def read_line(ser: serial.Serial, timeout_s: float) -> str:
    """
    Read exactly one newline-terminated serial line.

    Read one byte at a time so bytes belonging to later lines are never
    discarded. The previous bulk-read implementation could receive several
    MEM/PRED/READY lines at once, return only the first line, and silently lose
    everything after its first newline.
    """
    deadline = time.monotonic() + timeout_s
    buf = bytearray()

    while time.monotonic() < deadline:
        byte = ser.read(1)

        if not byte:
            continue

        if byte == b"\n":
            return buf.decode("utf-8", errors="replace").strip()

        if byte != b"\r":
            buf.extend(byte)

    raise TimeoutError("Timed out waiting for a serial line.")


def wait_for_ready(
    ser: serial.Serial,
    timeout_s: float = 20.0,
) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())

        try:
            line = read_line(ser, timeout_s=min(5.0, remaining))
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")

        if line == "READY" or line.endswith(" READY"):
            return

    raise TimeoutError("Device never said READY.")


def wait_for_exact_line(
    ser: serial.Serial,
    expected: str,
    timeout_s: float,
    show_expected: bool = True,
) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        line = read_line(ser, timeout_s=min(5.0, remaining))

        if line == expected:
            if show_expected:
                print(f"[rx] {line}")
            return

        if line:
            print(f"[rx] {line}")

        if line.startswith("ERR"):
            raise RuntimeError(line)

    raise TimeoutError(f"Timed out waiting for {expected!r}.")


def send_image_chunked(
    ser: serial.Serial,
    img_rgb_u8: np.ndarray,
) -> None:
    """
    Send raw HWC RGB uint8 data.

    Do not normalize here. The MCU firmware performs the same input
    normalization used by the trained Keras model.
    """
    if img_rgb_u8.shape != (IMG_H, IMG_W, IMG_C):
        raise ValueError(f"Unexpected image shape: {img_rgb_u8.shape}")
    if img_rgb_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img_rgb_u8.dtype}")

    payload = np.ascontiguousarray(img_rgb_u8).reshape(-1).tobytes()

    if len(payload) != IMG_SIZE:
        raise ValueError(
            f"Unexpected payload size: {len(payload)}, expected {IMG_SIZE}"
        )

    ser.write(b"IMG")
    ser.flush()

    wait_for_exact_line(ser, "RDYIMG", timeout_s=5.0)

    total_chunks = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_index, offset in enumerate(
        range(0, len(payload), CHUNK_SIZE),
        start=1,
    ):
        chunk = payload[offset : offset + CHUNK_SIZE]

        written = ser.write(chunk)
        ser.flush()

        if written != len(chunk):
            raise IOError(
                f"Serial write incomplete: wrote {written}/{len(chunk)} bytes"
            )

        wait_for_exact_line(
            ser,
            "ACK",
            timeout_s=5.0,
            show_expected=False,
        )

    print(
        f"[tx] image sent: {len(payload)} bytes "
        f"in {total_chunks} chunks"
    )


def read_pred_line(
    ser: serial.Serial,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())

        try:
            line = read_line(ser, timeout_s=min(5.0, remaining))
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")

        if line.startswith("PRED "):
            return line

        if line.startswith("ERR"):
            raise RuntimeError(line)

    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_line(
    line: str,
) -> tuple[int, float, float, str]:
    parts = line.strip().split(maxsplit=4)

    if len(parts) < 4 or parts[0] != "PRED":
        raise ValueError(f"Malformed prediction line: {line!r}")

    pred = int(parts[1])
    seconds = float(parts[2])
    confidence = float(parts[3])

    if pred < 0 or pred >= len(CLASS_NAMES):
        raise ValueError(f"Prediction class out of range: {pred}")

    name = parts[4] if len(parts) >= 5 else CLASS_NAMES[pred]

    return pred, seconds, confidence, name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send CIFAR-10 RGB images to RP2350 TinyResNet."
    )

    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--cache-dir", default="./cifar10_local")
    parser.add_argument("--save-dir", default="./sent_images_tinyresnet")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument("--download-limit", type=int, default=None)

    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x_test, y_test = load_cifar10_local_or_download(
        cache_dir,
        limit=args.download_limit,
    )

    n_samples = len(x_test)

    if n_samples == 0:
        raise RuntimeError("The CIFAR-10 test set is empty.")

    if not args.random and not (0 <= args.start < n_samples):
        raise ValueError(
            f"Start index out of range: {args.start}, dataset size={n_samples}"
        )

    if not args.no_plot:
        _, im, title_obj = plot_init()

    rng = np.random.default_rng(args.seed)

    print(f"Opening serial {args.port} @ {args.baud} ...")
    print("Sender format: raw HWC RGB uint8")
    print("Normalization: performed on the RP2350 firmware")

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
        write_timeout=5.0,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        ser.dtr = True
        ser.rts = False

        time.sleep(1.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)

        wait_for_ready(ser, timeout_s=20.0)

        correct = 0
        total = 0
        times: list[float] = []

        for k in range(args.n):
            if args.random:
                idx = int(rng.integers(0, n_samples))
            else:
                idx = (args.start + k) % n_samples

            img = np.ascontiguousarray(x_test[idx], dtype=np.uint8)
            gt = int(y_test[idx])
            gt_name = CLASS_NAMES[gt]

            local_png = (
                save_dir
                / f"sent_{k + 1:04d}_idx_{idx:05d}_gt_{gt}_{gt_name}.png"
            )
            Image.fromarray(img, mode="RGB").save(local_png)

            if not args.no_plot:
                plot_update(
                    im,
                    title_obj,
                    img,
                    title=(
                        f"TinyResNet {k + 1}/{args.n}\n"
                        f"idx={idx}  ground truth={gt_name}"
                    ),
                )

            try:
                send_image_chunked(ser, img)
                pred_line = read_pred_line(ser, args.timeout)
                pred, seconds, confidence, pred_name = parse_pred_line(pred_line)

            except Exception as exc:
                print(
                    f"[{k + 1}/{args.n}] idx={idx} "
                    f"gt={gt_name} -> ERROR: {exc}"
                )

                try:
                    wait_for_ready(ser, timeout_s=5.0)
                except Exception:
                    pass

                continue

            ok = pred == gt

            total += 1
            correct += int(ok)
            times.append(seconds)

            print(
                f"[{k + 1}/{args.n}] "
                f"idx={idx} "
                f"gt={gt_name} "
                f"pred={pred_name} "
                f"conf={confidence:.4f} "
                f"t={seconds:.6f}s "
                f"{'OK' if ok else 'NO'} "
                f"saved={local_png}"
            )

            if not args.no_plot:
                plot_update(
                    im,
                    title_obj,
                    img,
                    title=(
                        f"GT: {gt_name}\n"
                        f"Pred: {pred_name} ({confidence:.3f}) "
                        f"{'OK' if ok else 'NO'}"
                    ),
                )

            # The firmware prints READY after PRED.
            try:
                wait_for_ready(ser, timeout_s=5.0)
            except TimeoutError:
                print("[warn] READY not observed after prediction.")

            if args.pause > 0:
                time.sleep(args.pause)

    print(f"\nDone. Parsed {total}/{args.n} predictions.")

    if total:
        print(
            f"Accuracy: {correct}/{total} "
            f"= {100.0 * correct / total:.2f}%"
        )

    if times:
        time_array = np.asarray(times, dtype=np.float64)
        print(f"Mean inference: {time_array.mean():.6f} s")
        print(f"Min inference : {time_array.min():.6f} s")
        print(f"Max inference : {time_array.max():.6f} s")

        if len(time_array) > 1:
            print(
                f"Sample SD     : "
                f"{time_array.std(ddof=1):.6f} s"
            )

    if not args.no_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
