#!/usr/bin/env python3
"""
CIFAR-10 RGB sender for the RP2350 / Pico Tiny FireNet Noodle demo.

Protocol is the same NoodleSerial protocol used by the Blue Pill example:

  PC -> MCU:  IMG
  MCU -> PC:  RDYIMG
  PC -> MCU:  image bytes in 64-byte chunks
  MCU -> PC:  ACK after each chunk
  MCU -> PC:  PRED <class_id> <seconds> <confidence> <class_name>
  MCU -> PC:  READY

Difference from the MNIST sender:
  - image is RGB
  - payload is 32*32*3 = 3072 bytes
  - the sender saves selected images locally as PNG files
"""

import argparse
import time
from pathlib import Path

import numpy as np
import serial
import matplotlib.pyplot as plt
from PIL import Image


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMG_W = 32
IMG_H = 32
IMG_C = 3
IMG_SIZE = IMG_W * IMG_H * IMG_C
CHUNK_SIZE = 64


def load_cifar10_local_or_download(cache_dir: Path, limit: int | None = None):
    """
    Returns x_test_u8, y_test.

    Local cache:
      cache_dir/cifar10_test_images_u8.npy
      cache_dir/cifar10_test_labels.npy

    If not found, downloads from Hugging Face dataset uoft-cs/cifar10
    and saves the arrays locally.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_npy = cache_dir / "cifar10_test_images_u8.npy"
    lab_npy = cache_dir / "cifar10_test_labels.npy"

    if img_npy.exists() and lab_npy.exists():
      print(f"Loading local CIFAR-10 cache from {cache_dir}")
      x = np.load(img_npy)
      y = np.load(lab_npy)
      if limit is not None:
          x = x[:limit]
          y = y[:limit]
      return x, y

    print("Local CIFAR-10 cache not found.")
    print("Downloading CIFAR-10 from Hugging Face: uoft-cs/cifar10")

    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10")
    sample = ds["test"][0]
    image_key = "img" if "img" in sample else "image"

    images = []
    labels = []
    for ex in ds["test"]:
        images.append(np.asarray(ex[image_key], dtype=np.uint8))
        labels.append(int(ex["label"]))
        if limit is not None and len(images) >= limit:
            break

    x = np.stack(images, axis=0).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int64)

    np.save(img_npy, x)
    np.save(lab_npy, y)

    # Also save a small visual folder so the exact images are easy to inspect.
    png_dir = cache_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    n_png = min(len(x), 200)
    for i in range(n_png):
        label = int(y[i])
        name = CLASS_NAMES[label]
        Image.fromarray(x[i]).save(png_dir / f"test_{i:05d}_{label}_{name}.png")

    print(f"Saved cache to {cache_dir}")
    print(f"Saved first {n_png} PNG images to {png_dir}")

    return x, y


def plot_init():
    plt.ion()
    fig, ax = plt.subplots(figsize=(1, 1), dpi=120)
    img0 = np.zeros((IMG_H, IMG_W, IMG_C), dtype=np.uint8)
    im = ax.imshow(img0, interpolation="nearest")
    ax.axis("off")
    title_obj = ax.set_title("")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, im, title_obj


def plot_update(im, title_obj, img_u8: np.ndarray, title: str):
    im.set_data(img_u8)
    title_obj.set_text(title)
    im.figure.canvas.draw_idle()
    im.figure.canvas.flush_events()


def read_line(ser: serial.Serial, timeout_s: float) -> str:
    t0 = time.time()
    buf = bytearray()

    while (time.time() - t0) < timeout_s:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf.extend(chunk)
            if b"\n" in buf:
                line, _, _ = buf.partition(b"\n")
                return line.decode("utf-8", errors="replace").strip()

    raise TimeoutError("Timed out waiting for a line from serial.")


def wait_for_ready(ser: serial.Serial, timeout_s: float = 20.0):
    t0 = time.time()

    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=5.0)
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")

        if line and "READY" in line:
            return

    raise TimeoutError("Device never said READY.")


def send_image_chunked(ser: serial.Serial, img_rgb_u8: np.ndarray):
    assert img_rgb_u8.shape == (IMG_H, IMG_W, IMG_C)
    assert img_rgb_u8.dtype == np.uint8

    payload = img_rgb_u8.reshape(-1).tobytes()
    assert len(payload) == IMG_SIZE

    ser.write(b"IMG")
    ser.flush()

    while True:
        line = read_line(ser, timeout_s=5.0)
        print(f"[rx] {line}")

        if line == "RDYIMG":
            break

        if line.startswith("ERR"):
            raise RuntimeError(line)

    offset = 0
    while offset < len(payload):
        chunk = payload[offset:offset + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        offset += len(chunk)

        while True:
            line = read_line(ser, timeout_s=5.0)
            if line == "ACK":
                break
            if line.startswith("ERR"):
                raise RuntimeError(line)


def read_pred_line(ser: serial.Serial, timeout_s: float) -> str:
    t0 = time.time()

    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=5.0)
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")

        if line.startswith("PRED "):
            return line

        if line.startswith("ERR"):
            raise RuntimeError(line)

    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_line(line: str):
    parts = line.strip().split(maxsplit=4)
    if len(parts) < 4 or parts[0] != "PRED":
        return None, None, None, None

    pred = int(parts[1])
    seconds = float(parts[2])
    conf = float(parts[3])
    name = parts[4] if len(parts) >= 5 else CLASS_NAMES[pred]
    return pred, seconds, conf, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--cache-dir", default="./cifar10_local")
    ap.add_argument("--save-dir", default="./sent_images")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--pause", type=float, default=0.1)
    ap.add_argument("--download-limit", type=int, default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x_test, y_test = load_cifar10_local_or_download(cache_dir, limit=args.download_limit)
    n_samples = len(x_test)

    if not args.random and (args.start < 0 or args.start >= n_samples):
        raise ValueError(f"start index out of range: {args.start}, dataset size={n_samples}")

    if not args.no_plot:
        fig, ax, im, title_obj = plot_init()

    rng = np.random.default_rng()

    print(f"Opening serial {args.port} @ {args.baud} ...")

    with serial.Serial(args.port, args.baud, timeout=0.1, rtscts=False, dsrdtr=False) as ser:
        ser.dtr = True
        ser.rts = False

        time.sleep(1.5)
        ser.reset_input_buffer()
        time.sleep(0.2)

        wait_for_ready(ser, timeout_s=20.0)

        correct = 0
        total = 0
        times = []

        for k in range(args.n):
            idx = int(rng.integers(0, n_samples)) if args.random else (args.start + k) % n_samples

            img = x_test[idx].astype(np.uint8)
            gt = int(y_test[idx])
            gt_name = CLASS_NAMES[gt]

            local_png = save_dir / f"sent_{k+1:04d}_idx_{idx:05d}_gt_{gt}_{gt_name}.png"
            Image.fromarray(img).save(local_png)

            if not args.no_plot:
                plot_update(
                    im,
                    title_obj,
                    img,
                    title=f"iter={k+1}/{args.n} idx={idx} gt={gt_name}",
                )

            try:
                send_image_chunked(ser, img)
                pred_line = read_pred_line(ser, args.timeout)
            except Exception as e:
                print(f"[{k+1}/{args.n}] idx={idx} gt={gt_name} -> ERROR {e}")
                try:
                    wait_for_ready(ser, timeout_s=5.0)
                except Exception:
                    pass
                continue

            pred, seconds, conf, pred_name = parse_pred_line(pred_line)
            ok = (pred == gt)

            total += 1
            correct += 1 if ok else 0
            times.append(seconds)

            print(
                f"[{k+1}/{args.n}] idx={idx} gt={gt_name} "
                f"pred={pred_name} conf={conf:.4f} t={seconds:.6f}s "
                f"{'OK' if ok else 'NO'} saved={local_png}"
            )

            # Consume READY after prediction before next image.
            try:
                wait_for_ready(ser, timeout_s=5.0)
            except Exception:
                pass

            if args.pause > 0:
                time.sleep(args.pause)

    print(f"\nDone. Parsed {total}/{args.n} predictions.")
    if total:
        print(f"Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")
    if times:
        print(f"Mean inference: {np.mean(times):.6f} s")

    if not args.no_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
