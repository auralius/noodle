import time
import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt

# =========================
# EDIT THESE
# =========================
CSV_PATH    = "./mnist_train.csv"
SERIAL_PORT = "/dev/ttyUSB0"    # Windows: "COM5"
BAUD        = 115200            # must match Arduino sketch

N_TIMES     = 100
START_INDEX = 0
RANDOM      = False

TIMEOUT_S   = 30.0              # UNO + SD inference may be slow
SHOW_PLOT_EACH = True
PAUSE_BETWEEN_S = 0.2
PLOT_EVERY = 1

# --- AUGMENTATION on original 28x28 image ---
DO_RANDOM_ROT = True
ROT_DEG_MIN = -20.0
ROT_DEG_MAX = 20.0
ROT_FILL = 0

# --- Downsample settings ---
OUT_W = 16
OUT_H = 16
# =========================


def load_mnist_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    return X, y


def rotate_keep_size_u8(img28_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
    """
    Rotate 28x28 uint8 image by angle_deg, keeping size fixed.
    Uses pure NumPy bilinear sampling.
    """
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

    out = ((1 - wx) * (1 - wy) * Ia +
           wx * (1 - wy) * Ib +
           (1 - wx) * wy * Ic +
           wx * wy * Id)

    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def resize_bilinear_u8(img_u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Pure NumPy bilinear resize.
    Maps output pixel centers to input pixel centers.
    """
    assert img_u8.ndim == 2 and img_u8.dtype == np.uint8

    in_h, in_w = img_u8.shape

    if out_h == 1:
        ys = np.array([(in_h - 1) / 2.0], dtype=np.float32)
    else:
        ys = np.linspace(0, in_h - 1, out_h, dtype=np.float32)

    if out_w == 1:
        xs = np.array([(in_w - 1) / 2.0], dtype=np.float32)
    else:
        xs = np.linspace(0, in_w - 1, out_w, dtype=np.float32)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = yy - y0
    wx = xx - x0

    Ia = img_u8[y0, x0].astype(np.float32)
    Ib = img_u8[y0, x1].astype(np.float32)
    Ic = img_u8[y1, x0].astype(np.float32)
    Id = img_u8[y1, x1].astype(np.float32)

    out = ((1 - wx) * (1 - wy) * Ia +
           wx * (1 - wy) * Ib +
           (1 - wx) * wy * Ic +
           wx * wy * Id)

    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def mnist28_to_16_u8(img28_u8: np.ndarray) -> np.ndarray:
    """
    Convert MNIST 28x28 to 16x16 uint8 before sending to Arduino.

    This assumes the model was trained on direct 16x16 resized MNIST.
    If your training used center-of-mass normalization, apply the same
    preprocessing here before resize.
    """
    assert img28_u8.shape == (28, 28) and img28_u8.dtype == np.uint8
    return resize_bilinear_u8(img28_u8, OUT_H, OUT_W)


def plot_init():
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), dpi=100)

    im28 = axes[0].imshow(np.zeros((28, 28), dtype=np.uint8),
                          cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("28x28\n")
    axes[0].axis("off")

    im16 = axes[1].imshow(np.zeros((16, 16), dtype=np.uint8),
                          cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("16x16 sent\n")
    axes[1].axis("off")

    title_obj = fig.suptitle("")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, im28, im16, title_obj


def plot_update(im28, im16, title_obj, img28_u8: np.ndarray, img16_u8: np.ndarray, title: str):
    im28.set_data(img28_u8)
    im16.set_data(img16_u8)
    title_obj.set_text(title)
    im28.figure.canvas.draw_idle()
    im28.figure.canvas.flush_events()


def send_raw_256(ser: serial.Serial, x256: np.ndarray):
    assert x256.dtype == np.uint8 and x256.size == 256
    ser.reset_input_buffer()
    ser.write(x256.tobytes())
    ser.flush()


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


def wait_for_ready(ser: serial.Serial, timeout_s: float = 30.0):
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=0.5)
        except TimeoutError:
            continue
        if line:
            if "READY" in line:
                return
            print(f"[rx] {line}")

    raise TimeoutError("Device never said READY.")


def read_pred_line(ser: serial.Serial, timeout_s: float) -> str:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=0.5)
        except TimeoutError:
            continue
        if line and line.startswith("PRED "):
            return line
    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_from_predline(line: str):
    """
    Arduino format:
      PRED <digit> <seconds> <max_val>
    """
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "PRED":
        return None, None, None

    pred = seconds = max_val = None

    try:
        pred = int(parts[1])
    except Exception:
        pass

    if len(parts) >= 3:
        try:
            seconds = float(parts[2])
        except Exception:
            pass

    if len(parts) >= 4:
        try:
            max_val = float(parts[3])
        except Exception:
            pass

    return pred, seconds, max_val


def main():
    X, y = load_mnist_csv(CSV_PATH)
    n_samples = X.shape[0]

    if not RANDOM and (START_INDEX < 0 or START_INDEX >= n_samples):
        raise ValueError(f"START_INDEX out of range: {START_INDEX}, dataset size={n_samples}")

    if SHOW_PLOT_EACH:
        fig, im28, im16, title_obj = plot_init()

    print(f"Opening serial {SERIAL_PORT} @ {BAUD} ...")
    with serial.Serial(SERIAL_PORT, BAUD, timeout=0.1) as ser:
        time.sleep(0.5)
        wait_for_ready(ser, timeout_s=30.0)

        correct = 0
        total = 0
        rng = np.random.default_rng()

        for k in range(N_TIMES):
            idx = int(rng.integers(0, n_samples)) if RANDOM else (START_INDEX + k) % n_samples

            img28 = X[idx].reshape(28, 28)
            gt = int(y[idx])

            angle = 0.0
            if DO_RANDOM_ROT:
                angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                img28 = rotate_keep_size_u8(img28, angle_deg=angle, fill=ROT_FILL)

            img16 = mnist28_to_16_u8(img28)
            x256_send = img16.reshape(256).astype(np.uint8)

            if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                plot_update(
                    im28, im16, title_obj, img28, img16,
                    title=f"iter={k+1}/{N_TIMES} idx={idx} gt={gt} rot={angle:+.1f}°"
                )

            send_raw_256(ser, x256_send)

            try:
                pred_line = read_pred_line(ser, TIMEOUT_S)
            except TimeoutError:
                print(f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} rot={angle:+.1f}° -> TIMEOUT waiting PRED")
                continue

            pred, seconds, max_val = parse_pred_from_predline(pred_line)
            ok = (pred == gt)

            total += 1
            correct += 1 if ok else 0

            extra = ""
            if seconds is not None:
                extra += f"  t={seconds:.3f}s"
            if max_val is not None:
                extra += f"  P={max_val:.4f}"

            print(f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} rot={angle:+.1f}° -> {pred_line}  {'OK' if ok else 'NO'}{extra}")

            # Arduino prints READY after prediction. Consume it before next send.
            try:
                wait_for_ready(ser, timeout_s=5.0)
            except TimeoutError:
                pass

            if PAUSE_BETWEEN_S > 0:
                time.sleep(PAUSE_BETWEEN_S)

    print(f"\nDone. Parsed {total}/{N_TIMES} predictions.")
    if total > 0:
        acc = 100.0 * correct / total
        print(f"Accuracy: {correct}/{total} = {acc:.2f}%")

    if SHOW_PLOT_EACH:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
