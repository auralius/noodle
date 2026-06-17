#!/usr/bin/env python3
"""
MNIST denoising sender for Noodle DeepSeq-AE-28 over serial.

Main image protocol expected by ESP32 firmware:
  PC -> ESP32:
      b"IMG"
      then 784 uint8 pixels sent in 64-byte chunks

  ESP32 -> PC during image receive:
      RDYIMG
      ACK after each chunk

  ESP32 -> PC after inference:
      OUT <microseconds>\n
      784 uint8 denoised pixels
      optional newline
      READY\n

Extra memory-query protocol:
  PC -> ESP32:
      b"MEM"

  ESP32 -> PC:
      MEM <fields...>\n
      READY\n

Why this sender is almost the same as the old AE sender:
  - Input and output are still 28x28 uint8 images.
  - The image protocol is unchanged.
  - The only addition is request_mem(), called after each completed image.
  - Memory is requested only after READY, so it does not corrupt binary output.
"""

import time
import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt

# =========================
# EDIT THESE
# =========================
CSV_PATH    = "./mnist_train.csv"
SERIAL_PORT = "/dev/ttyACM0"
BAUD        = 921600

N_TIMES     = 100
START_INDEX = 0
RANDOM      = False

# DeepSeq-AE-28 inference is about 3.4 s on the current ESP32-S3 test.
# Keep this generous while collecting data.
TIMEOUT_S   = 180.0

SHOW_PLOT_EACH = True
PAUSE_BETWEEN_S = 0.2
PLOT_EVERY = 1

# Denoising input
ADD_NOISE = True
NOISE_FACTOR = 0.25
RNG_SEED = 123

# Optional augmentation
DO_RANDOM_ROT = False
ROT_DEG_MIN = -10.0
ROT_DEG_MAX = 10.0
ROT_FILL = 0

# Memory query after each completed image.
REQUEST_MEM_EACH = True

# Save logs
SAVE_CSV = True
LOG_CSV_PATH = "deepseq_ae28_noodle_log.csv"

# Save last displayed figure
SAVE_FINAL_FIG = False
FINAL_FIG_PATH = "deepseq_ae28_noodle_denoising_result.png"
# =========================

IMG_W = 28
IMG_H = 28
IMG_SIZE = IMG_W * IMG_H
CHUNK_SIZE = 64


def load_mnist_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    if X.shape[1] != IMG_SIZE:
        raise ValueError(f"Expected {IMG_SIZE} pixel columns, got {X.shape[1]}")
    return X, y


def rotate_keep_size_u8(img28_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
    """Rotate 28x28 uint8 image by angle_deg, keeping size fixed, using NumPy bilinear sampling."""
    assert img28_u8.shape == (28, 28) and img28_u8.dtype == np.uint8

    H, W = img28_u8.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    ys, xs = np.indices((H, W), dtype=np.float32)
    x = xs - cx
    y = ys - cy

    # inverse rotate: destination -> source
    xs_src =  c * x + s * y + cx
    ys_src = -s * x + c * y + cy

    x0 = np.floor(xs_src).astype(np.int32)
    y0 = np.floor(ys_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xs_src - x0
    wy = ys_src - y0

    def sample(xx, yy):
        mask = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        out = np.full((H, W), fill, dtype=np.float32)
        out[mask] = img28_u8[yy[mask], xx[mask]].astype(np.float32)
        return out

    Ia = sample(x0, y0)
    Ib = sample(x1, y0)
    Ic = sample(x0, y1)
    Id = sample(x1, y1)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def add_gaussian_noise_u8(img28_u8: np.ndarray, rng: np.random.Generator, factor: float) -> np.ndarray:
    """MNIST uint8 -> noisy uint8, matching Keras-style x + factor*N(0,1) in [0,1]."""
    assert img28_u8.shape == (28, 28)
    x = img28_u8.astype(np.float32) / 255.0
    y = x + factor * rng.normal(size=x.shape)
    y = np.clip(y, 0.0, 1.0)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def mse_u8(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    return float(np.mean((af - bf) ** 2))


def plot_init():
    plt.ion()
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2), dpi=110)

    titles = ["Clean", "Noisy input", "Noodle output", "|Noisy-Clean|", "|Output-Clean|"]
    ims = []
    for ax, title in zip(axes, titles):
        if "|" in title:
            im = ax.imshow(
                np.zeros((IMG_H, IMG_W), dtype=np.float32),
                cmap="magma",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        else:
            im = ax.imshow(
                np.zeros((IMG_H, IMG_W), dtype=np.uint8),
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
        ax.set_title(title)
        ax.axis("off")
        ims.append(im)

    suptitle = fig.suptitle("")
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, axes, ims, suptitle


def plot_update(ims, suptitle, clean_u8, noisy_u8, out_u8, title: str):
    err_noisy = np.abs(noisy_u8.astype(np.float32) - clean_u8.astype(np.float32)) / 255.0
    err_out   = np.abs(out_u8.astype(np.float32)   - clean_u8.astype(np.float32)) / 255.0

    ims[0].set_data(clean_u8)
    ims[1].set_data(noisy_u8)
    ims[2].set_data(out_u8)
    ims[3].set_data(err_noisy)
    ims[4].set_data(err_out)
    suptitle.set_text(title)

    fig = ims[0].figure
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def read_line(ser: serial.Serial, timeout_s: float) -> str:
    """Read one text line. Do not use this for binary payload.

    Important: this implementation reads one byte at a time so it does not
    accidentally discard bytes that arrive after the newline in the same USB packet.
    That is important because OUT lines are followed by binary image bytes.
    """
    t0 = time.time()
    buf = bytearray()

    while (time.time() - t0) < timeout_s:
        b = ser.read(1)
        if not b:
            continue

        if b == b"\n":
            return buf.decode("utf-8", errors="replace").strip()

        if b != b"\r":
            buf.extend(b)

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


def send_image_chunked(ser: serial.Serial, img28_u8: np.ndarray):
    payload = img28_u8.reshape(-1).astype(np.uint8).tobytes()

    ser.write(b"IMG")
    ser.flush()

    # Wait for firmware to say it is ready for chunks.
    while True:
        line = read_line(ser, timeout_s=5.0)
        print(f"[rx] {line!r}")

        if line == "RDYIMG":
            break

        # Some firmware prints READY periodically while idle.
        # Ignore it here; we only proceed on RDYIMG.
        if line.startswith("ERR"):
            raise RuntimeError(line)

    offset = 0
    while offset < len(payload):
        chunk = payload[offset:offset + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        offset += len(chunk)

        # Wait ACK after each chunk.
        while True:
            line = read_line(ser, timeout_s=5.0)
            print(f"[rx] {line!r}")

            if line == "ACK":
                break

            if line.startswith("ERR"):
                raise RuntimeError(line)


def send_image_get_output(ser: serial.Serial, img28_u8: np.ndarray, timeout_s: float):
    assert img28_u8.shape == (IMG_H, IMG_W)
    assert img28_u8.dtype == np.uint8

    payload = img28_u8.reshape(-1).tobytes()
    print(f"TX bytes: {len(payload)} header: True")

    send_image_chunked(ser, img28_u8)

    # Read OUT line.
    t0 = time.time()
    us = None

    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=2.0)
        except TimeoutError:
            continue

        if not line:
            continue

        # In clean timing runs, this should usually be only OUT.
        # If you enable firmware debug, this also shows text lines before OUT.
        print(f"[rx wait OUT] {line!r}")

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

    # Read exact binary payload.
    out_bytes = recv_exact(ser, IMG_SIZE, timeout_s=timeout_s)
    out_u8 = np.frombuffer(out_bytes, dtype=np.uint8).copy().reshape(IMG_H, IMG_W)

    # The firmware may send an extra newline after binary data.
    # wait_for_ready() will skip blank lines until READY appears.
    wait_for_ready(ser, timeout_s=timeout_s)

    return out_u8, us


def request_mem(ser: serial.Serial, timeout_s: float = 10.0):
    """Ask the ESP32 for memory state.

    Call this only after a completed image, i.e. after wait_for_ready().
    """
    ser.write(b"MEM")
    ser.flush()

    mem_line = None
    while True:
        line = read_line(ser, timeout_s=timeout_s)
        print(f"[mem] {line!r}")

        if line.startswith("MEM "):
            mem_line = line

        if line == "READY" or "READY" in line:
            return mem_line


def parse_mem_line(line: str | None) -> dict:
    """Parse the MEM key=value line into a dictionary."""
    if not line:
        return {}

    out = {}
    parts = line.split()
    if not parts or parts[0] != "MEM":
        return out

    # parts[1] is tag, for example "query" or "after_prealloc".
    if len(parts) > 1:
        out["mem_tag"] = parts[1]

    for p in parts[2:]:
        if "=" not in p:
            continue

        k, v = p.split("=", 1)
        v = v.strip()

        if v.endswith("%"):
            try:
                out[k] = float(v[:-1])
            except ValueError:
                out[k] = v
        else:
            try:
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v

    return out


def main():
    X, y = load_mnist_csv(CSV_PATH)
    n_samples = X.shape[0]

    if not RANDOM and (START_INDEX < 0 or START_INDEX >= n_samples):
        raise ValueError(f"START_INDEX out of range: {START_INDEX}, dataset size={n_samples}")

    fig = axes = ims = suptitle = None
    if SHOW_PLOT_EACH:
        fig, axes, ims, suptitle = plot_init()

    rng = np.random.default_rng(RNG_SEED)

    rows = []

    print(f"Opening serial {SERIAL_PORT} @ {BAUD} ...")
    with serial.Serial(
        SERIAL_PORT,
        BAUD,
        timeout=0.1,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        # Match PlatformIO monitor_dtr=0 and monitor_rts=0 behavior.
        ser.dtr = False
        ser.rts = False

        # Do not reset input before first READY; it can erase boot messages.
        time.sleep(1.0)
        wait_for_ready(ser, timeout_s=20.0)

        mse_noisy_list = []
        mse_out_list = []
        times_us = []

        for k in range(N_TIMES):
            idx = int(rng.integers(0, n_samples)) if RANDOM else (START_INDEX + k) % n_samples
            clean_u8 = X[idx].reshape(IMG_H, IMG_W).astype(np.uint8)
            gt = int(y[idx])

            angle = 0.0
            if DO_RANDOM_ROT:
                angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                clean_eval_u8 = rotate_keep_size_u8(clean_u8, angle_deg=angle, fill=ROT_FILL)
            else:
                clean_eval_u8 = clean_u8

            noisy_u8 = (
                add_gaussian_noise_u8(clean_eval_u8, rng, NOISE_FACTOR)
                if ADD_NOISE else clean_eval_u8.copy()
            )

            try:
                out_u8, us = send_image_get_output(ser, noisy_u8, timeout_s=TIMEOUT_S)

                mem_line = None
                mem = {}
                if REQUEST_MEM_EACH:
                    mem_line = request_mem(ser, timeout_s=TIMEOUT_S)
                    mem = parse_mem_line(mem_line)

            except Exception as e:
                print(f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} -> ERROR: {e}")
                try:
                    wait_for_ready(ser, timeout_s=5.0)
                except Exception:
                    pass
                continue

            m_noisy = mse_u8(noisy_u8, clean_eval_u8)
            m_out = mse_u8(out_u8, clean_eval_u8)
            reduction = 100.0 * (m_noisy - m_out) / m_noisy if m_noisy > 0 else 0.0

            mse_noisy_list.append(m_noisy)
            mse_out_list.append(m_out)
            if us is not None:
                times_us.append(us)

            print(
                f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} rot={angle:+.1f}° "
                f"us={us} mse_noisy={m_noisy:.6f} mse_out={m_out:.6f} red={reduction:.2f}%"
            )

            if mem:
                print(
                    "    MEM "
                    f"bytes={mem.get('bytes')} "
                    f"heap={mem.get('heap')} heap_frag={mem.get('heap_frag')}% "
                    f"psram={mem.get('psram')} psram_frag={mem.get('psram_frag')}%"
                )

            row = {
                "iter": k + 1,
                "idx": idx,
                "gt": gt,
                "rot_deg": angle,
                "time_us": us,
                "mse_noisy": m_noisy,
                "mse_out": m_out,
                "reduction_percent": reduction,
                "mem_line": mem_line,
            }
            row.update({f"mem_{kk}": vv for kk, vv in mem.items()})
            rows.append(row)

            if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                plot_update(
                    ims,
                    suptitle,
                    clean_eval_u8,
                    noisy_u8,
                    out_u8,
                    title=(
                        f"iter={k+1}/{N_TIMES} idx={idx} gt={gt} "
                        f"MSE noisy={m_noisy:.4f}, output={m_out:.4f}, red={reduction:.1f}%"
                    ),
                )

            if SAVE_CSV:
                pd.DataFrame(rows).to_csv(LOG_CSV_PATH, index=False)

            if PAUSE_BETWEEN_S > 0:
                time.sleep(PAUSE_BETWEEN_S)

    print("\nDone.")
    if mse_noisy_list:
        mean_noisy = float(np.mean(mse_noisy_list))
        mean_out = float(np.mean(mse_out_list))
        print(f"Mean noisy MSE : {mean_noisy:.9g}")
        print(f"Mean output MSE: {mean_out:.9g}")
        print(f"Mean reduction : {100.0 * (mean_noisy - mean_out) / mean_noisy:.2f}%")

    if times_us:
        print(f"Mean inference : {np.mean(times_us):.1f} us")
        print(f"Std inference  : {np.std(times_us):.1f} us")

    if rows and REQUEST_MEM_EACH:
        df = pd.DataFrame(rows)
        for col in ["mem_bytes", "mem_heap", "mem_heap_largest", "mem_heap_frag",
                    "mem_psram", "mem_psram_largest", "mem_psram_frag"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty:
                    print(f"{col}: first={vals.iloc[0]} last={vals.iloc[-1]} mean={vals.mean():.3f}")

    if SAVE_CSV and rows:
        pd.DataFrame(rows).to_csv(LOG_CSV_PATH, index=False)
        print(f"Saved log CSV: {LOG_CSV_PATH}")

    if SHOW_PLOT_EACH:
        if SAVE_FINAL_FIG and fig is not None:
            fig.savefig(FINAL_FIG_PATH, dpi=200)
            print(f"Saved final figure: {FINAL_FIG_PATH}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
