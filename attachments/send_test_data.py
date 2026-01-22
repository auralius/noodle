import os
import glob
import struct
import time
import serial
from PIL import Image
import numpy as np

# =====================
# CONFIG
# =====================
PORT = "/dev/ttyACM0"
BAUD = 115200

# Point this to ONE folder you want to iterate (e.g. "person" or "non_person")
FOLDER = "person"

W = 96
N = W * W * 3  # 27648 bytes

# Optional: wait for this line from ESP before sending each frame
WAIT_READY_LINE = False
READY_TOKEN = "READY"   # change if your ESP prints something else

# =====================
# IMAGE -> BYTES (matches training: RGB + resize 96 + rescale 1/255 on device)
# =====================
def image_to_rgb_bytes(path: str) -> bytes:
    im = Image.open(path).convert("RGB").resize((W, W), Image.BILINEAR)
    rgb = im.tobytes()
    if len(rgb) != N:
        raise ValueError(f"{path}: got {len(rgb)} bytes, expected {N}")
    return rgb

def debug_chw_first_pixel(rgb: bytes):
    hwc = np.frombuffer(rgb, dtype=np.uint8).reshape((W, W, 3))
    chw = np.transpose(hwc.astype(np.float32) / 255.0, (2, 0, 1))
    r, g, b = float(chw[0, 0, 0]), float(chw[1, 0, 0]), float(chw[2, 0, 0])
    print(f"PY DBG first pixel normalized: R={r:.6f} G={g:.6f} B={b:.6f}")

# =====================
# SERIAL PROTOCOL (simple firmware)
#   u32_le(length) + payload bytes
# =====================
def send_frame_len_only(ser: serial.Serial, rgb: bytes):
    ser.write(struct.pack("<I", len(rgb)))
    ser.write(rgb)
    ser.flush()

def read_one_result(ser: serial.Serial, timeout_s: float = 60.0):
    """
    Reads until it sees:
      - a line containing 'P0=' (your inference output)
      - and (optionally) 'time_ms=' line
    Returns: (p0, p1, pred, time_ms, raw_lines)
    """
    t0 = time.time()
    p0 = p1 = None
    pred = None
    time_ms = None
    raw = []

    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        raw.append(line)

        # Example: "ms=0 P0=0.660586 P1=0.339414 pred=0"
        if "P0=" in line and "P1=" in line:
            try:
                parts = line.replace("=", " ").split()
                # parts like: ['ms', '0', 'P0', '0.660586', 'P1', '0.339414', 'pred', '0']
                def get_val(key):
                    i = parts.index(key)
                    return parts[i + 1]
                p0 = float(get_val("P0"))
                p1 = float(get_val("P1"))
                pred = int(get_val("pred"))
            except Exception:
                pass

        # Example: "time_ms=29869"
        if line.startswith("time_ms="):
            try:
                time_ms = int(line.split("=", 1)[1])
            except Exception:
                pass

        # Stop once we have probs; time_ms may come later, but usually right after
        if p0 is not None and p1 is not None and pred is not None:
            # give it a short chance to also catch time_ms
            t_short = time.time()
            while time.time() - t_short < 2.0 and time_ms is None:
                l2 = ser.readline().decode(errors="ignore").strip()
                if l2:
                    raw.append(l2)
                    if l2.startswith("time_ms="):
                        try:
                            time_ms = int(l2.split("=", 1)[1])
                        except Exception:
                            pass
            return p0, p1, pred, time_ms, raw

    return None, None, None, None, raw

def drain_startup(ser: serial.Serial, seconds: float = 1.5):
    t0 = time.time()
    while time.time() - t0 < seconds:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP:", line)

def wait_for_token(ser: serial.Serial, token: str, timeout_s: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP:", line)
            if token in line:
                return True
    return False

# =====================
# MAIN: run all images in FOLDER
# =====================
if __name__ == "__main__":
    # Collect images
    exts = ("*.jpg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(FOLDER, e)))
    files = files[0:10]

    if not files:
        raise SystemExit(f"No images found in folder: {FOLDER}")

    print(f"Found {len(files)} images in '{FOLDER}'")

    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(2.0)
        drain_startup(ser, seconds=1.5)

        results = []
        for idx, path in enumerate(files, 1):
            try:
                rgb = image_to_rgb_bytes(path)
            except Exception as e:
                print(f"[{idx}/{len(files)}] SKIP {path} ({e})")
                continue

            # Optional debug for the first image only
            if idx == 1:
                debug_chw_first_pixel(rgb)

            if WAIT_READY_LINE:
                print("Waiting for ESP ready token...")
                if not wait_for_token(ser, READY_TOKEN, timeout_s=30.0):
                    print("Timeout waiting for READY; continuing anyway.")

            # Send frame
            send_frame_len_only(ser, rgb)

            # Read result
            p0, p1, pred, time_ms, raw = read_one_result(ser, timeout_s=120.0)

            if p0 is None:
                print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> NO RESULT")
                # Uncomment to see raw lines if debugging:
                # for l in raw: print("ESP:", l)
                continue

            print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> "
                  f"P0={p0:.6f} P1={p1:.6f} pred={pred} time_ms={time_ms}")

            results.append((path, p0, p1, pred, time_ms))

    # Optional: summary
    if results:
        avg_p1 = sum(r[2] for r in results) / len(results)
        avg_t  = sum(r[4] for r in results if r[4] is not None) / max(1, sum(1 for r in results if r[4] is not None))
        print(f"\nSummary: n={len(results)} avg_P1={avg_p1:.6f} avg_time_ms={avg_t:.1f}")
