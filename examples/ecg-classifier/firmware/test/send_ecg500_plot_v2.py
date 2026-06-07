#!/usr/bin/env python3
"""
Robust ECG500 sender for Noodle / ESP32.

Protocol expected by firmware:
  ESP32 prints READY
  Host sends b"IMG"
  ESP32 prints RDYIMG
  Host sends 500 float32 samples in 64-byte chunks
  ESP32 prints ACK after each chunk
  ESP32 prints OUT <dt_us>
  ESP32 sends 4 float32 probabilities
  ESP32 prints READY

Example:
  python send_ecg500_plot_v2.py --port /dev/ttyUSB0 --baud 115200 --npz ecg500_testset.npz --count 20
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import serial
import matplotlib.pyplot as plt


CLASS_NAMES = ["Brady", "Normal", "Tachy", "Irregular"]


@dataclass
class Result:
    idx: int
    true_class: int
    py_pred: int
    mcu_pred: int
    dt_us: int
    py_probs: np.ndarray
    mcu_probs: np.ndarray


def open_serial(port: str, baud: int, timeout: float = 0.05) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = timeout
    ser.write_timeout = 2.0

    # Important for many ESP32 USB-serial boards:
    # prevent Python from holding EN/BOOT in a bad reset state.
    ser.dtr = False
    ser.rts = False

    ser.open()
    time.sleep(0.2)

    # Keep both low after opening. Some boards reset on open.
    ser.setDTR(False)
    ser.setRTS(False)

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def read_available_text(ser: serial.Serial) -> str:
    n = ser.in_waiting
    if n <= 0:
        return ""
    b = ser.read(n)
    return b.decode("utf-8", errors="replace")


def wait_for_token(ser: serial.Serial, token: str, timeout_s: float = 10.0, verbose: bool = True) -> str:
    """
    Byte-level token search. More robust than readline() when the MCU prints
    fragments such as "E", "RAD", or repeats READY while the host is opening.
    """
    target = token.encode("ascii")
    buf = bytearray()
    t0 = time.time()

    while (time.time() - t0) < timeout_s:
        try:
            b = ser.read(1)
        except serial.SerialException as e:
            raise RuntimeError(
                f"Serial disappeared while waiting for {token!r}. "
                f"Close PlatformIO monitor/Arduino serial monitor, check cable, "
                f"and verify the port is still {ser.port}."
            ) from e

        if b:
            buf += b
            if verbose:
                # Print complete lines when possible, but do not require lines.
                while b"\n" in buf:
                    line, _, rest = buf.partition(b"\n")
                    buf = bytearray(rest)
                    s = line.decode("utf-8", errors="replace").strip()
                    if s:
                        print(f"[rx] {s}")

            if target in buf:
                return buf.decode("utf-8", errors="replace")
        else:
            time.sleep(0.005)

    tail = buf[-120:].decode("utf-8", errors="replace")
    raise TimeoutError(f"Timeout waiting for {token!r}. Last bytes: {tail!r}")


def wait_ready(ser: serial.Serial, timeout_s: float = 10.0) -> None:
    print("[host] waiting for READY ...")
    wait_for_token(ser, "READY", timeout_s=timeout_s, verbose=True)
    print("[host] READY")


def send_frame(ser: serial.Serial, x500: np.ndarray, chunk_size: int = 64) -> tuple[np.ndarray, int]:
    x = np.asarray(x500, dtype="<f4").reshape(500)
    payload = x.tobytes()

    # Clear stale text before starting a transaction.
    stale = read_available_text(ser)
    if stale.strip():
        print("[rx stale]", stale.strip())

    ser.write(b"IMG")
    ser.flush()

    wait_for_token(ser, "RDYIMG", timeout_s=5.0, verbose=False)

    # Send payload with ACK after each chunk.
    sent = 0
    while sent < len(payload):
        chunk = payload[sent:sent + chunk_size]
        ser.write(chunk)
        ser.flush()
        sent += len(chunk)
        wait_for_token(ser, "ACK", timeout_s=3.0, verbose=False)

    # OUT line comes before binary output.
    out_text = wait_for_token(ser, "OUT", timeout_s=10.0, verbose=True)

    # Parse dt_us from the text around OUT.
    dt_us = -1
    # We may not have the full line yet; read a little more until newline.
    t0 = time.time()
    while "\n" not in out_text and (time.time() - t0) < 1.0:
        more = ser.read(1)
        if more:
            out_text += more.decode("utf-8", errors="replace")

    for line in out_text.splitlines():
        line = line.strip()
        if line.startswith("OUT"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    dt_us = int(parts[1])
                except ValueError:
                    dt_us = -1

    raw = bytearray()
    while len(raw) < 16:
        b = ser.read(16 - len(raw))
        if b:
            raw.extend(b)
        else:
            time.sleep(0.002)

    probs = np.frombuffer(bytes(raw), dtype="<f4").copy()

    # Consume trailing READY if it is already there, but do not fail if not.
    try:
        wait_for_token(ser, "READY", timeout_s=2.0, verbose=False)
    except Exception:
        pass

    return probs, dt_us


def load_npz(path: str):
    data = np.load(path)
    X = data["X_test"]
    if X.ndim == 3:
        X = X.reshape(X.shape[0], X.shape[1])
    y_true = data["y_true"] if "y_true" in data else np.argmax(data["y_test"], axis=1)
    pred_python = data["pred_python"] if "pred_python" in data else None
    return X.astype(np.float32), y_true.astype(int), pred_python


def plot_single(idx: int, x: np.ndarray, true_class: int, py_probs: np.ndarray | None,
                mcu_probs: np.ndarray, dt_us: int, save_prefix: str | None, show: bool):
    t = np.arange(500)

    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.title(f"ECG window idx={idx}, true={true_class} ({CLASS_NAMES[true_class]})")
    plt.xlabel("Sample")
    plt.ylabel("Normalized amplitude")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_idx{idx:05d}_ecg.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    labels = np.arange(4)
    width = 0.35
    plt.figure(figsize=(8, 4))
    if py_probs is not None:
        plt.bar(labels - width/2, py_probs, width, label="Python")
        plt.bar(labels + width/2, mcu_probs, width, label="ESP32")
    else:
        plt.bar(labels, mcu_probs, width, label="ESP32")
    plt.xticks(labels, CLASS_NAMES, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Probability")
    plt.title(f"Probabilities idx={idx}, dt={dt_us} us")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_idx{idx:05d}_probs.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_summary(results: list[Result], save_prefix: str | None, show: bool):
    if not results:
        return

    idxs = [r.idx for r in results]
    true_classes = [r.true_class for r in results]
    py_preds = [r.py_pred for r in results]
    mcu_preds = [r.mcu_pred for r in results]
    dt_ms = [r.dt_us / 1000.0 for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(idxs, true_classes, marker="o", label="True")
    plt.plot(idxs, py_preds, marker="x", label="Python pred")
    plt.plot(idxs, mcu_preds, marker="s", label="ESP32 pred")
    plt.yticks(range(4), CLASS_NAMES)
    plt.xlabel("Dataset index")
    plt.ylabel("Class")
    plt.title("Predicted classes")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_summary_classes.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(idxs, dt_ms, marker="o")
    plt.xlabel("Dataset index")
    plt.ylabel("ESP32 inference time (ms)")
    plt.title("ESP32 inference time")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_summary_time.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--plot-every", type=int, default=1)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-prefix", default=None)
    args = ap.parse_args()

    X, y_true, pred_python = load_npz(args.npz)

    print("=== DATASET ===")
    print("X:", X.shape)
    print("y_true:", y_true.shape)
    if pred_python is not None:
        print("pred_python:", pred_python.shape)

    end = min(args.start + args.count, len(X))
    print(f"Sending indices [{args.start}, {end})")

    if args.save_prefix:
        out_dir = os.path.dirname(args.save_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    ser = open_serial(args.port, args.baud)

    try:
        # ESP32 often resets after port open. Give it time to reboot.
        time.sleep(1.5)
        wait_ready(ser, timeout_s=15.0)

        results: list[Result] = []

        for idx in range(args.start, end):
            x = X[idx].reshape(500).astype(np.float32)
            true_c = int(y_true[idx])

            py_probs = None
            py_pred = -1
            if pred_python is not None:
                py_probs = pred_python[idx].astype(np.float32)
                py_pred = int(np.argmax(py_probs))

            print(f"\n[tx] idx={idx} true={true_c} {CLASS_NAMES[true_c]}")
            mcu_probs, dt_us = send_frame(ser, x, chunk_size=args.chunk_size)
            mcu_pred = int(np.argmax(mcu_probs))

            print(f"[rx] dt_us={dt_us}")
            print(f"[rx] mcu_probs={np.array2string(mcu_probs, precision=6)} pred={mcu_pred} {CLASS_NAMES[mcu_pred]}")
            if py_probs is not None:
                diff = np.abs(mcu_probs - py_probs)
                print(f"[py] py_probs ={np.array2string(py_probs, precision=6)} pred={py_pred} {CLASS_NAMES[py_pred]}")
                print(f"[df] max_abs_diff={diff.max():.6g} mean_abs_diff={diff.mean():.6g}")

            results.append(Result(
                idx=idx,
                true_class=true_c,
                py_pred=py_pred,
                mcu_pred=mcu_pred,
                dt_us=dt_us,
                py_probs=py_probs if py_probs is not None else np.zeros(4, dtype=np.float32),
                mcu_probs=mcu_probs,
            ))

            if args.plot_every > 0 and ((idx - args.start) % args.plot_every == 0):
                plot_single(idx, x, true_c, py_probs, mcu_probs, dt_us, args.save_prefix, not args.no_show)

        plot_summary(results, args.save_prefix, not args.no_show)

        if results:
            mcu_acc = np.mean([r.mcu_pred == r.true_class for r in results])
            print("\n=== SUMMARY ===")
            print(f"Frames: {len(results)}")
            print(f"ESP32 accuracy on sent frames: {mcu_acc:.4f}")
            if pred_python is not None:
                agree = np.mean([r.mcu_pred == r.py_pred for r in results])
                print(f"ESP32/Python prediction agreement: {agree:.4f}")
                maxdiff = max(float(np.max(np.abs(r.mcu_probs - r.py_probs))) for r in results)
                meandiff = np.mean([float(np.mean(np.abs(r.mcu_probs - r.py_probs))) for r in results])
                print(f"Max probability abs diff: {maxdiff:.6g}")
                print(f"Mean probability abs diff: {meandiff:.6g}")
            print(f"Mean dt: {np.mean([r.dt_us for r in results]) / 1000.0:.3f} ms")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
