#!/usr/bin/env python3
"""
RGB 128x128 scoliosis autoencoder sender for Noodle over serial.

Dataset layout expected:
  DATASET_ROOT/
    normal/
      image1.jpg/png/...
    scoliosis/
      image2.jpg/png/...

Protocol expected by MCU firmware:
  PC -> MCU:
      b"IMG"
      image payload in chunks, MCU replies RDYIMG then ACK per chunk

  MCU -> PC:
      OUT <microseconds>\n
      128*128*3 uint8 RGB reconstructed bytes
      optional newline
      READY\n
The image sent to the MCU is uint8 RGB HWC: R,G,B,R,G,B,...
The MCU should convert uint8 -> float [0,1] internally before inference.
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import serial
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Defaults; can be overridden by CLI
# =========================
DATASET_ROOT = "./"
NORMAL_DIR_NAME = "normal"
SCOLIOSIS_DIR_NAME = "scoliosis"
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 921600

IMG_W = 128
IMG_H = 128
IMG_C = 3
IMG_SIZE = IMG_W * IMG_H * IMG_C

N_TIMES = 20
START_INDEX = 0
RANDOM_SAMPLE = False
RNG_SEED = 123

TIMEOUT_S = 120.0
CHUNK_SIZE = 64
PAUSE_BETWEEN_S = 0.2

SHOW_PLOT_EACH = True
PLOT_EVERY = 1
SAVE_FINAL_FIG = False
FINAL_FIG_PATH = "scoliosis_noodle_autoencoder_result.png"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(dataset_root: str, normal_name: str, scoliosis_name: str) -> List[Tuple[Path, str]]:
    root = Path(dataset_root)
    items: List[Tuple[Path, str]] = []

    for label in [normal_name, scoliosis_name]:
        folder = root / label
        if not folder.exists():
            print(f"[warn] folder not found: {folder}")
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                items.append((p, label))

    if not items:
        raise FileNotFoundError(
            f"No images found under {root}/{normal_name} or {root}/{scoliosis_name}. "
            f"Supported extensions: {sorted(EXTS)}"
        )
    return items


def load_rgb_u8(path: Path, width: int = IMG_W, height: int = IMG_H) -> np.ndarray:
    """Load image, convert to RGB, resize to 128x128, return uint8 HWC."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != (height, width, 3):
        raise ValueError(f"Unexpected image shape {arr.shape} for {path}")
    return arr


def mse_u8(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    return float(np.mean((af - bf) ** 2))


def mae_u8(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    return float(np.mean(np.abs(af - bf)))


def read_line(ser: serial.Serial, timeout_s: float) -> str:
    """Read one text line. Do not use this for binary payload."""
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
    """Wait until the board announces READY."""
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=2.0)
        except TimeoutError:
            continue
        if line:
            print(f"[rx] {line!r}")
            if line == "READY" or "READY" in line:
                return
    raise TimeoutError("Device never said READY.")


def recv_exact(ser: serial.Serial, n: int, timeout_s: float) -> bytes:
    t0 = time.time()
    buf = bytearray()
    while len(buf) < n:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {n} bytes; got {len(buf)}")
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)


def send_image_chunked(ser: serial.Serial, img_rgb_u8: np.ndarray, chunk_size: int = CHUNK_SIZE):
    assert img_rgb_u8.shape == (IMG_H, IMG_W, IMG_C)
    assert img_rgb_u8.dtype == np.uint8

    payload = np.ascontiguousarray(img_rgb_u8).reshape(-1).tobytes()
    if len(payload) != IMG_SIZE:
        raise ValueError(f"Payload size mismatch: got {len(payload)}, expected {IMG_SIZE}")

    ser.write(b"IMG")
    ser.flush()

    while True:
        line = read_line(ser, timeout_s=5.0)
        print(f"[rx] {line!r}")
        if line == "RDYIMG":
            break
        if line.startswith("ERR"):
            raise RuntimeError(line)

    offset = 0
    n_chunks = (len(payload) + chunk_size - 1) // chunk_size
    chunk_id = 0
    while offset < len(payload):
        chunk = payload[offset:offset + chunk_size]
        ser.write(chunk)
        ser.flush()
        offset += len(chunk)
        chunk_id += 1

        while True:
            line = read_line(ser, timeout_s=10.0)
            # Keep this print. It is useful for finding the exact chunk where sync breaks.
            #print(f"[rx] {line!r} chunk={chunk_id}/{n_chunks}")
            if line == "ACK":
                break
            if line.startswith("ERR"):
                raise RuntimeError(line)


def send_image_get_output(ser: serial.Serial,
                          img_rgb_u8: np.ndarray,
                          timeout_s: float,
                          chunk_size: int = CHUNK_SIZE):
    assert img_rgb_u8.shape == (IMG_H, IMG_W, IMG_C)
    assert img_rgb_u8.dtype == np.uint8

    print(f"TX bytes: {IMG_SIZE} chunk_size={chunk_size}")
    send_image_chunked(ser, img_rgb_u8, chunk_size=chunk_size)

    t0 = time.time()
    us = None
    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=2.0)
        except TimeoutError:
            continue
        if not line:
            continue
        print(f"[rx] {line}")
        if line.startswith("OUT "):
            parts = line.split()
            try:
                us = int(parts[1]) if len(parts) >= 2 else None
            except ValueError:
                us = None
            break
        if line.startswith("ERR"):
            raise RuntimeError(line)
    else:
        raise TimeoutError("Timed out waiting for OUT line")

    out_bytes = recv_exact(ser, IMG_SIZE, timeout_s=timeout_s)
    out_u8 = np.frombuffer(out_bytes, dtype=np.uint8).copy().reshape(IMG_H, IMG_W, IMG_C)

    wait_for_ready(ser, timeout_s=timeout_s)
    return out_u8, us


def plot_init():
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=110)
    titles = ["Input", "Noodle reconstruction", "|Output-Input|"]
    ims = []

    ims.append(axes[0].imshow(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), interpolation="nearest"))
    axes[0].set_title(titles[0])

    ims.append(axes[1].imshow(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), interpolation="nearest"))
    axes[1].set_title(titles[1])

    ims.append(axes[2].imshow(np.zeros((IMG_H, IMG_W), dtype=np.float32), cmap="magma", vmin=0, vmax=1, interpolation="nearest"))
    axes[2].set_title(titles[2])

    for ax in axes:
        ax.axis("off")

    suptitle = fig.suptitle("")
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, axes, ims, suptitle


def plot_update(ims, suptitle, inp_u8, out_u8, title: str):
    err = np.mean(np.abs(out_u8.astype(np.float32) - inp_u8.astype(np.float32)), axis=2) / 255.0
    ims[0].set_data(inp_u8)
    ims[1].set_data(out_u8)
    ims[2].set_data(err)
    suptitle.set_text(title)
    fig = ims[0].figure
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def parse_args():
    p = argparse.ArgumentParser(description="Send normal/scoliosis RGB images to Noodle autoencoder over serial.")
    p.add_argument("--dataset-root", default=DATASET_ROOT)
    p.add_argument("--normal-dir", default=NORMAL_DIR_NAME)
    p.add_argument("--scoliosis-dir", default=SCOLIOSIS_DIR_NAME)
    p.add_argument("--port", default=SERIAL_PORT)
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--n", type=int, default=N_TIMES)
    p.add_argument("--n-per-class", type=int, default=None,
                   help="Send this many normal images and this many scoliosis images. Overrides --n when set.")
    p.add_argument("--start-index", type=int, default=START_INDEX)
    p.add_argument("--random", action="store_true", default=RANDOM_SAMPLE)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--timeout", type=float, default=TIMEOUT_S)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--pause", type=float, default=PAUSE_BETWEEN_S)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-every", type=int, default=PLOT_EVERY)
    p.add_argument("--save-final-fig", action="store_true", default=SAVE_FINAL_FIG)
    p.add_argument("--final-fig-path", default=FINAL_FIG_PATH)
    return p.parse_args()


def main():
    args = parse_args()

    items = list_images(args.dataset_root, args.normal_dir, args.scoliosis_dir)
    print(f"Found {len(items)} images.")
    print(f"normal={sum(1 for _, y in items if y == args.normal_dir)}, "
          f"scoliosis={sum(1 for _, y in items if y == args.scoliosis_dir)}")

    rng = random.Random(args.seed)

    # Build the exact test list.
    # Default behavior: old behavior, one combined list using --n and --start-index.
    # Balanced behavior: --n-per-class K sends K normal and K scoliosis images.
    if args.n_per_class is not None:
        if args.n_per_class < 0:
            raise ValueError("--n-per-class must be >= 0")

        normal_items = [(p, y) for p, y in items if y == args.normal_dir]
        scoliosis_items = [(p, y) for p, y in items if y == args.scoliosis_dir]

        if len(normal_items) < args.n_per_class:
            raise ValueError(f"Not enough normal images: requested {args.n_per_class}, found {len(normal_items)}")
        if len(scoliosis_items) < args.n_per_class:
            raise ValueError(f"Not enough scoliosis images: requested {args.n_per_class}, found {len(scoliosis_items)}")

        if args.random:
            selected = rng.sample(normal_items, args.n_per_class) + rng.sample(scoliosis_items, args.n_per_class)
            rng.shuffle(selected)
        else:
            selected = normal_items[:args.n_per_class] + scoliosis_items[:args.n_per_class]

        print(f"Balanced test: {args.n_per_class} normal + {args.n_per_class} scoliosis = {len(selected)} images")
    else:
        if not args.random and (args.start_index < 0 or args.start_index >= len(items)):
            raise ValueError(f"--start-index out of range: {args.start_index}, dataset size={len(items)}")

        selected = []
        for k in range(args.n):
            idx = rng.randrange(len(items)) if args.random else (args.start_index + k) % len(items)
            selected.append(items[idx])

    fig = axes = ims = suptitle = None
    show_plot = not args.no_plot
    if show_plot:
        fig, axes, ims, suptitle = plot_init()

    mse_list = []
    mae_list = []
    times_us = []

    print(f"Opening serial {args.port} @ {args.baud} ...")
    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        ser.dtr = False
        ser.rts = False

        time.sleep(1.0)
        wait_for_ready(ser, timeout_s=20.0)

        for k, (path, label) in enumerate(selected):
            idx = items.index((path, label))
            inp_u8 = load_rgb_u8(path)

            try:
                out_u8, us = send_image_get_output(
                    ser,
                    inp_u8,
                    timeout_s=args.timeout,
                    chunk_size=args.chunk_size,
                )
            except Exception as e:
                print(f"[{k+1}/{len(selected)}] idx={idx} label={label} file={path.name} -> ERROR: {e}")
                try:
                    wait_for_ready(ser, timeout_s=5.0)
                except Exception:
                    pass
                continue

            m = mse_u8(out_u8, inp_u8)
            a = mae_u8(out_u8, inp_u8)
            mse_list.append(m)
            mae_list.append(a)
            if us is not None:
                times_us.append(us)

            print(
                f"[{k+1}/{len(selected)}] idx={idx} label={label} file={path.name} "
                f"us={us} mse_recon={m:.7f} mae_recon={a:.7f}"
            )

            if show_plot and (k % max(1, args.plot_every) == 0):
                plot_update(
                    ims,
                    suptitle,
                    inp_u8,
                    out_u8,
                    title=f"iter={k+1}/{len(selected)} label={label} file={path.name} MSE={m:.5f} MAE={a:.5f}",
                )

            if args.pause > 0:
                time.sleep(args.pause)

    print("\nDone.")
    if mse_list:
        print(f"Mean reconstruction MSE: {float(np.mean(mse_list)):.9g}")
        print(f"Mean reconstruction MAE: {float(np.mean(mae_list)):.9g}")
    if times_us:
        print(f"Mean inference: {float(np.mean(times_us)):.1f} us")

    if show_plot:
        if args.save_final_fig and fig is not None:
            fig.savefig(args.final_fig_path, dpi=200)
            print(f"Saved final figure: {args.final_fig_path}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
