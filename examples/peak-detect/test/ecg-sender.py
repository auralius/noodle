#!/usr/bin/env python3
"""
PeakNet serial test (single-file)

- Generates 10 s synthetic ECG at 256 Hz (2560 samples)
- Splits into 256-sample frames
- Sends each frame to Arduino as: b"ECG" + 256 float32 (little-endian)
- Reads back: "SCORES <elapsed_sec> <256 floats...>"
- Stitches score traces, picks peaks, prints peaks to terminal, plots ECG + peaks

Requirements:
  pip install pyserial numpy matplotlib
"""

import time
import numpy as np
import serial
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
PORT = "/dev/ttyUSB0"   # change (Windows: "COM5")
BAUD = 115200

FS = 256               # Hz
DURATION_S = 10.0
L = 256                # window length
HOP = 256              # set 128 for overlap

THRESH = 0.5
REFRACTORY_S = 0.25    # seconds


def synthetic_ecg(fs=256, duration_s=10.0, hr_bpm=72, noise=0.02, seed=2):
    """Cartoon ECG generator (good for pipeline tests)."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    rr = 60.0 / hr_bpm
    beats = np.arange(0.5, duration_s - 0.5, rr)

    x = np.zeros_like(t)

    def add_gaussian(center, amp, width):
        nonlocal x
        x += amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    # baseline wander
    x += 0.05 * np.sin(2 * np.pi * 0.33 * t)

    for bt in beats:
        add_gaussian(bt - 0.18,  0.12, 0.035)  # P
        add_gaussian(bt - 0.04, -0.15, 0.010)  # Q
        add_gaussian(bt,        1.00, 0.012)   # R
        add_gaussian(bt + 0.03, -0.25, 0.012)  # S
        add_gaussian(bt + 0.22,  0.35, 0.060)  # T

    x += noise * rng.standard_normal(size=n)
    x /= (np.max(np.abs(x)) + 1e-9)
    return t, x, beats


def score_to_peaks(score, fs, thresh=0.5, refractory_s=0.25):
    """Threshold + local max + refractory."""
    score = np.asarray(score)
    refractory = max(1, int(refractory_s * fs))

    mask = score > thresh
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)

    # local maxima per contiguous region
    peaks = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        region = np.arange(start, prev + 1)
        p = region[np.argmax(score[region])]
        peaks.append(p)
        start = k
        prev = k
    region = np.arange(start, prev + 1)
    p = region[np.argmax(score[region])]
    peaks.append(p)

    peaks = np.array(peaks, dtype=int)

    # refractory filter (keep strongest)
    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= refractory:
            kept.append(p)
        else:
            if score[p] > score[kept[-1]]:
                kept[-1] = p
    return np.array(kept, dtype=int)


def wait_for_ready(ser, timeout_s=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print("[dev]", line)
        if line == "READY":
            return True
    return False


def send_frame(ser, frame_f32):
    frame_f32 = np.asarray(frame_f32, dtype=np.float32)
    ser.write(b"ECG" + frame_f32.tobytes(order="C"))


def read_scores_line(ser, L_expected=256, timeout_s=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("SCORES "):
            parts = line.split()
            # parts[0] = "SCORES", parts[1] = elapsed seconds, parts[2:] = L floats
            if len(parts) < 2 + L_expected:
                print("[warn] short SCORES line len=", len(parts))
                continue
            et = float(parts[1])
            vals = np.array([float(x) for x in parts[2:2 + L_expected]], dtype=np.float32)
            print(vals)
            return et, vals
        else:
            print("[dev]", line)
    return None, None


def main():
    t, x, true_beats = synthetic_ecg(fs=FS, duration_s=DURATION_S)
    n = len(x)

    print(f"[host] opening serial {PORT} @ {BAUD} ...")
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(1.5)  # allow auto-reset on open

    if not wait_for_ready(ser):
        print("[host] ERROR: did not see READY from device")
        ser.close()
        return

    score_all = np.zeros(n, dtype=np.float32)
    count_all = np.zeros(n, dtype=np.float32)
    infer_times = []

    for start in range(0, n - L + 1, HOP):
        frame = x[start:start + L].astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-6)
        send_frame(ser, frame)


        et, scores = read_scores_line(ser, L_expected=L)
        if scores is None:
            print(f"[host] warn: no SCORES for window start={start}")
            continue

        infer_times.append(et)
        score_all[start:start + L] += scores
        count_all[start:start + L] += 1.0

    ser.close()

    count_all[count_all == 0] = 1.0
    score_avg = score_all / count_all

    peaks = score_to_peaks(score_avg, FS, thresh=THRESH, refractory_s=REFRACTORY_S)

    true_idx = (true_beats * FS).astype(int)
    true_idx = true_idx[(true_idx >= 0) & (true_idx < n)]

    # -----------------------------
    # Terminal printout
    # -----------------------------
    print("\n=== Predicted Peaks ===")
    for i, p in enumerate(peaks):
        print(f"{i:02d}: sample={p:4d}, time={t[p]:6.3f} s, score={score_avg[p]:.3f}")
    print(f"Total predicted peaks: {len(peaks)}")

    print("\n=== True R (reference) ===")
    for i, p in enumerate(true_idx):
        print(f"{i:02d}: sample={p:4d}, time={t[p]:6.3f} s")
    print(f"Total true beats: {len(true_idx)}")

    if len(peaks) > 0 and len(true_idx) > 0:
        m = min(len(peaks), len(true_idx))
        print("\n=== Peak Errors (paired by order) ===")
        for i in range(m):
            err_samp = int(peaks[i] - true_idx[i])
            err_ms = 1000.0 * err_samp / FS
            print(f"{i:02d}: err = {err_samp:+4d} samples ({err_ms:+6.1f} ms)")

    if infer_times:
        import numpy as _np
        print("\nDevice inference seconds (min/avg/max):",
              float(_np.min(infer_times)),
              float(_np.mean(infer_times)),
              float(_np.max(infer_times)))

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure()
    plt.plot(t, x, label="ECG (synthetic)")
    plt.plot(t, 0.6 * score_avg - 0.8, label="score (scaled)")

    plt.scatter(
        t[peaks], x[peaks],
        marker="x", s=90, linewidths=2.5,
        color="red", zorder=5,
        label="Pred peaks"
    )

    plt.scatter(
        t[true_idx], x[true_idx],
        marker="o", s=60,
        facecolors="none", edgecolors="black",
        zorder=4,
        label="True R (ref)"
    )

    plt.title("PeakNet serial test (256-sample frames)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
