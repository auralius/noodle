#!/usr/bin/env python3
"""
ECG500 Noodle/ESP32 serial sender with live/static matplotlib plotting.

Protocol expected by the ESP32 sketch:
  PC waits for READY
  PC sends b"IMG"
  MCU prints RDYIMG
  PC sends 500 float32 samples in 64-byte chunks, waiting for ACK after each chunk
  MCU prints OUT <dt_us>
  MCU sends 4 float32 probabilities
  MCU prints READY

Input file:
  ecg500_testset.npz containing at least:
    X_test      shape (N, 500, 1) or (N, 500)
    y_true      shape (N,) optional but recommended
    pred_python shape (N, 4) optional but recommended
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import serial
import matplotlib.pyplot as plt


CLASS_NAMES = ["Brady", "Normal", "Tachy", "Irregular"]
CHUNK_SIZE = 64
ECG_LEN = 500
N_CLASSES = 4


@dataclass
class FrameResult:
    idx: int
    true_class: int | None
    mcu_pred: int
    py_pred: int | None
    mcu_probs: np.ndarray
    py_probs: np.ndarray | None
    dt_us: int | None
    ok: bool


def read_line_until(ser: serial.Serial, wanted: str, timeout_s: float = 5.0) -> str:
    """Read text lines until a line contains wanted."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            line = repr(raw)
        if line:
            last = line
            print(f"[rx] {line}")
        if wanted in line:
            return line
    raise TimeoutError(f"Timeout waiting for {wanted!r}. Last line: {last!r}")


def wait_ready(ser: serial.Serial, timeout_s: float = 10.0) -> None:
    read_line_until(ser, "READY", timeout_s)


def send_one_frame(ser: serial.Serial, x500: np.ndarray, timeout_s: float = 5.0) -> tuple[np.ndarray, int | None]:
    """Send one 500-float ECG frame and return (4 probabilities, dt_us)."""
    x = np.asarray(x500, dtype="<f4").reshape(ECG_LEN)
    payload = x.tobytes(order="C")
    assert len(payload) == ECG_LEN * 4

    # Header. The firmware waits for bytes I, M, G.
    ser.write(b"IMG")
    ser.flush()

    read_line_until(ser, "RDYIMG", timeout_s)

    # Chunked payload, matching noodle_serial.cpp CHUNK_SIZE=64.
    sent = 0
    while sent < len(payload):
        chunk = payload[sent:sent + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        sent += len(chunk)
        read_line_until(ser, "ACK", timeout_s)

    # The firmware prints OUT <dt_us>, then writes 16 raw bytes.
    out_line = read_line_until(ser, "OUT", timeout_s)
    dt_us = None
    parts = out_line.split()
    if len(parts) >= 2:
        try:
            dt_us = int(parts[1])
        except ValueError:
            dt_us = None

    raw = ser.read(N_CLASSES * 4)
    if len(raw) != N_CLASSES * 4:
        raise TimeoutError(f"Expected {N_CLASSES * 4} output bytes, got {len(raw)}")

    probs = np.frombuffer(raw, dtype="<f4").astype(np.float32)

    # Consume the next READY if it arrives. If not, the next iteration will wait for it.
    try:
        wait_ready(ser, timeout_s=2.0)
    except TimeoutError:
        pass

    return probs, dt_us


def load_dataset(npz_path: str):
    data = np.load(npz_path)
    X = data["X_test"].astype(np.float32)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], X.shape[1])
    if X.shape[1] != ECG_LEN:
        raise ValueError(f"Expected X_test with length {ECG_LEN}, got shape {X.shape}")

    y_true = data["y_true"].astype(np.int32) if "y_true" in data.files else None
    pred_python = data["pred_python"].astype(np.float32) if "pred_python" in data.files else None

    return X, y_true, pred_python


def plot_frame(x, result: FrameResult, save_path: str | None = None, show: bool = True):
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x)
    ax1.set_title(f"ECG500 sample #{result.idx}")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Normalized ECG")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    pos = np.arange(N_CLASSES)
    width = 0.35
    ax2.bar(pos - width / 2, result.mcu_probs, width, label="ESP32/Noodle")
    if result.py_probs is not None:
        ax2.bar(pos + width / 2, result.py_probs, width, label="Python/Keras")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(CLASS_NAMES, rotation=20)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Probability")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    true_txt = "?" if result.true_class is None else f"{result.true_class}={CLASS_NAMES[result.true_class]}"
    py_txt = "?" if result.py_pred is None else f"{result.py_pred}={CLASS_NAMES[result.py_pred]}"
    dt_txt = "?" if result.dt_us is None else f"{result.dt_us / 1000.0:.3f} ms"
    fig.suptitle(
        f"True: {true_txt} | MCU: {result.mcu_pred}={CLASS_NAMES[result.mcu_pred]} | "
        f"Python: {py_txt} | MCU time: {dt_txt}"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(results: list[FrameResult], save_path: str | None = None, show: bool = True):
    if not results:
        return

    idxs = np.array([r.idx for r in results])
    mcu_preds = np.array([r.mcu_pred for r in results])
    true_available = all(r.true_class is not None for r in results)
    py_available = all(r.py_pred is not None for r in results)
    dt = np.array([np.nan if r.dt_us is None else r.dt_us / 1000.0 for r in results])

    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(idxs, mcu_preds, marker="o", linestyle="-", label="MCU pred")
    if true_available:
        y_true = np.array([r.true_class for r in results])
        ax1.plot(idxs, y_true, marker="x", linestyle="--", label="True")
    if py_available:
        py_preds = np.array([r.py_pred for r in results])
        ax1.plot(idxs, py_preds, marker=".", linestyle=":", label="Python pred")
    ax1.set_yticks(np.arange(N_CLASSES))
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Class")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(idxs, dt, marker="o")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("MCU inference time [ms]")
    ax2.grid(True, alpha=0.3)

    title = "ECG500 Noodle/ESP32 inference summary"
    if true_available:
        ok = np.array([r.mcu_pred == r.true_class for r in results])
        title += f" | accuracy={ok.mean():.3f}"
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Must match NoodleSerial::begin(...)")
    parser.add_argument("--npz", required=True, help="Path to ecg500_testset.npz")
    parser.add_argument("--start", type=int, default=0, help="First sample index")
    parser.add_argument("--count", type=int, default=20, help="Number of samples to send")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--plot-every", type=int, default=1, help="Plot every N frames; use 0 to disable per-frame plots")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots interactively")
    parser.add_argument("--save-prefix", default=None, help="Save plots using this prefix, e.g. runs/ecg")
    args = parser.parse_args()

    X, y_true, pred_python = load_dataset(args.npz)
    end = min(args.start + args.count, len(X))

    print("=== DATASET ===")
    print("X:", X.shape)
    print("y_true:", None if y_true is None else y_true.shape)
    print("pred_python:", None if pred_python is None else pred_python.shape)
    print(f"Sending indices [{args.start}, {end})")

    results: list[FrameResult] = []

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("[host] waiting for READY ...")
        wait_ready(ser, timeout_s=10.0)

        for i in range(args.start, end):
            x = X[i].astype(np.float32)
            true_class = int(y_true[i]) if y_true is not None else None
            py_probs = pred_python[i].astype(np.float32) if pred_python is not None else None
            py_pred = int(np.argmax(py_probs)) if py_probs is not None else None

            print(f"\n[tx] idx={i}")
            mcu_probs, dt_us = send_one_frame(ser, x, timeout_s=args.timeout)
            mcu_pred = int(np.argmax(mcu_probs))

            true_txt = "?" if true_class is None else f"{true_class}:{CLASS_NAMES[true_class]}"
            py_txt = "?" if py_pred is None else f"{py_pred}:{CLASS_NAMES[py_pred]}"
            dt_txt = "?" if dt_us is None else f"{dt_us / 1000.0:.3f} ms"
            print(f"[res] true={true_txt} mcu={mcu_pred}:{CLASS_NAMES[mcu_pred]} py={py_txt} dt={dt_txt}")
            print(f"[res] mcu_probs={np.array2string(mcu_probs, precision=6, floatmode='fixed')}")
            if py_probs is not None:
                print(f"[res] py_probs ={np.array2string(py_probs, precision=6, floatmode='fixed')}")
                print(f"[res] abs diff ={np.array2string(np.abs(mcu_probs - py_probs), precision=6, floatmode='fixed')}")

            ok = true_class is None or mcu_pred == true_class
            result = FrameResult(i, true_class, mcu_pred, py_pred, mcu_probs, py_probs, dt_us, ok)
            results.append(result)

            if args.plot_every and ((len(results) - 1) % args.plot_every == 0):
                save_path = None
                if args.save_prefix:
                    save_path = f"{args.save_prefix}_frame_{i:05d}.png"
                plot_frame(x, result, save_path=save_path, show=not args.no_show)

    print("\n=== SUMMARY ===")
    if results and all(r.true_class is not None for r in results):
        acc = np.mean([r.mcu_pred == r.true_class for r in results])
        print(f"MCU accuracy on sent frames: {acc:.4f}")
    if results and all(r.py_pred is not None for r in results):
        agree = np.mean([r.mcu_pred == r.py_pred for r in results])
        print(f"MCU/Python prediction agreement: {agree:.4f}")
        diffs = np.array([np.abs(r.mcu_probs - r.py_probs) for r in results])
        print(f"Mean abs probability diff: {diffs.mean():.6g}")
        print(f"Max  abs probability diff: {diffs.max():.6g}")
    times = np.array([r.dt_us for r in results if r.dt_us is not None])
    if len(times):
        print(f"MCU time mean: {times.mean() / 1000.0:.3f} ms")
        print(f"MCU time min : {times.min() / 1000.0:.3f} ms")
        print(f"MCU time max : {times.max() / 1000.0:.3f} ms")

    summary_path = None
    if args.save_prefix:
        summary_path = f"{args.save_prefix}_summary.png"
    plot_summary(results, save_path=summary_path, show=not args.no_show)


if __name__ == "__main__":
    main()
