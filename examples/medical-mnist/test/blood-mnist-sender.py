import time
import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt

# MedMNIST (BloodMNIST)
try:
    from medmnist.dataset import BloodMNIST
except Exception:
    BloodMNIST = None

# =========================
# EDIT THESE
# =========================
# If USE_BLOODMNIST=True, CSV_PATH is ignored.
CSV_PATH    = "./mnist_train.csv"
SERIAL_PORT = "/dev/ttyUSB0"    # Windows: "COM5"
BAUD        = 115200

# Dataset selector
USE_BLOODMNIST = True

# BloodMNIST split: "train" | "val" | "test"
BLOOD_SPLIT = "test"

# Channel order for CxWxW streaming. For BloodMNIST images are RGB.
# This should match training-time channel order (typically RGB).
CHANNEL_ORDER = "RGB"  # or "BGR" if you trained that way

N_TIMES     = 100
START_INDEX = 0
RANDOM      = False

TIMEOUT_S   = 100.0              # per-iteration wait for PRED
SHOW_PLOT_EACH = True
PAUSE_BETWEEN_S = 0.2

# Optional plotting speed knob: update UI every M iterations
PLOT_EVERY = 1                  # set 5/10 for faster UI

# --- AUGMENTATION ---
DO_RANDOM_ROT = True
ROT_DEG_MIN = -10.0
ROT_DEG_MAX = 10.0
ROT_FILL = 0                    # background fill (0=black)
# =========================


def load_mnist_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    return X, y


def load_bloodmnist(split: str = "test"):
    """Return X,y for BloodMNIST.

    X: uint8 images, shape (N, 28, 28, 3)
    y: int labels, shape (N,)
    """
    if BloodMNIST is None:
        raise ImportError("medmnist is not installed. Run: pip install medmnist")
    ds = BloodMNIST(split=split, download=True)
    X = ds.imgs
    y = ds.labels.squeeze().astype(np.int64)

    # Ensure NHWC uint8
    if X.ndim == 3:
        # (N,28,28) -> grayscale; expand to 3 by repeating
        X = X[..., None]
    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)
    return X.astype(np.uint8), y


# ---------- Random rotation (keeps 28x28) ----------
def rotate_keep_size_u8(img28_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
    """
    Rotate 28x28 uint8 image by angle_deg, keeping size fixed.
    Uses pure NumPy bilinear sampling (no extra deps).
    """
    assert img28_u8.shape == (28, 28) and img28_u8.dtype == np.uint8

    H, W = img28_u8.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    # output grid
    ys, xs = np.indices((H, W), dtype=np.float32)
    x = xs - cx
    y = ys - cy

    # inverse rotate (dest -> src)
    # [xs] = [ c  s][x]
    # [ys]   [-s  c][y]
    xs_src =  c * x + s * y + cx
    ys_src = -s * x + c * y + cy

    # bilinear sampling
    x0 = np.floor(xs_src).astype(np.int32)
    y0 = np.floor(ys_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # weights
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
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out
# --------------------------------------------------


def rotate_keep_size_u8_rgb(img28_rgb_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
    """Rotate a 28x28x3 uint8 RGB image by angle_deg, keeping size fixed."""
    assert img28_rgb_u8.shape == (28, 28, 3) and img28_rgb_u8.dtype == np.uint8
    out = np.empty_like(img28_rgb_u8)
    for c in range(3):
        out[..., c] = rotate_keep_size_u8(img28_rgb_u8[..., c], angle_deg=angle_deg, fill=fill)
    return out


# ---------- Fast plotting (init once, update data) ----------
def plot_init(channels: int):
    plt.ion()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    if channels == 1:
        img0 = np.zeros((28, 28), dtype=np.uint8)
        im = ax.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        img0 = np.zeros((28, 28, 3), dtype=np.uint8)
        im = ax.imshow(img0, vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    title_obj = ax.set_title("")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, im, title_obj


def plot_update(im, title_obj, img, title: str):
    im.set_data(img)
    title_obj.set_text(title)
    im.figure.canvas.draw_idle()
    im.figure.canvas.flush_events()
# -----------------------------------------------------------


def send_raw_bytes(ser: serial.Serial, payload_u8: np.ndarray):
    assert payload_u8.dtype == np.uint8
    ser.reset_input_buffer()
    ser.write(payload_u8.tobytes())
    ser.flush()


def to_chw_u8(img: np.ndarray, channel_order: str = "RGB") -> np.ndarray:
    """Convert image to uint8 CxWxW flattened, channel-stacked."""
    if img.ndim == 2:
        # grayscale 28x28 -> 1x28x28
        chw = img.reshape(1, 28, 28)
        return chw.reshape(-1).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 3:
        # HWC -> CHW
        if channel_order.upper() == "RGB":
            chw = np.transpose(img, (2, 0, 1))
        elif channel_order.upper() == "BGR":
            chw = np.transpose(img[..., ::-1], (2, 0, 1))
        else:
            raise ValueError(f"Unsupported CHANNEL_ORDER: {channel_order}")
        return chw.reshape(-1).astype(np.uint8)

    raise ValueError(f"Unexpected image shape: {img.shape}")


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
            line = read_line(ser, timeout_s=5)
        except TimeoutError:
            continue
        if line:
            print(f"[rx] {line}")
            if "READY" in line:
                return
    raise TimeoutError("Device never said READY.")


def read_pred_line(ser: serial.Serial, timeout_s: float) -> str:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            line = read_line(ser, timeout_s=5)
            print(line)
        except TimeoutError:
            continue
        if line and line.startswith("PRED "):
            return line
    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_from_predline(line: str):
    """
    ESP format:
      PRED <digit> <seconds> <max_val>
    """
    parts = line.strip().split()
    if len(parts) < 4 or parts[0] != "PRED":
        return None, None, None

    pred = seconds = max_val = None
    try:
        pred = int(parts[1])
    except Exception:
        pass
    try:
        seconds = float(parts[2])
    except Exception:
        pass
    try:
        max_val = float(parts[3])
    except Exception:
        pass

    return pred, seconds, max_val


def main():
    if USE_BLOODMNIST:
        X, y = load_bloodmnist(split=BLOOD_SPLIT)
        n_channels = 3 if (X.ndim == 4 and X.shape[-1] == 3) else 1
        print(f"Loaded BloodMNIST split='{BLOOD_SPLIT}' with X={X.shape}, y={y.shape}")
    else:
        X, y = load_mnist_csv(CSV_PATH)
        n_channels = 1
        print(f"Loaded MNIST CSV with X={X.shape}, y={y.shape}")

    n_samples = X.shape[0]

    if not RANDOM and (START_INDEX < 0 or START_INDEX >= n_samples):
        raise ValueError(f"START_INDEX out of range: {START_INDEX}, dataset size={n_samples}")

    if SHOW_PLOT_EACH:
        fig, ax, im, title_obj = plot_init(n_channels)

    print(f"Opening serial {SERIAL_PORT} @ {BAUD} ...")
    with serial.Serial(SERIAL_PORT, BAUD, timeout=0.1) as ser:
        time.sleep(0.5)
        wait_for_ready(ser, timeout_s=15.0)

        correct = 0
        total = 0

        rng = np.random.default_rng()

        for k in range(N_TIMES):
            idx = int(rng.integers(0, n_samples)) if RANDOM else (START_INDEX + k) % n_samples

            gt = int(y[idx])

            angle = 0.0
            if USE_BLOODMNIST:
                # X[idx] is (28,28,3) or (28,28)
                img = X[idx]
                if img.ndim == 4:
                    img = img[0]
                if img.ndim == 3 and img.shape[2] == 3:
                    img28 = img.astype(np.uint8)
                    if DO_RANDOM_ROT:
                        angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                        img28 = rotate_keep_size_u8_rgb(img28, angle_deg=angle, fill=ROT_FILL)
                else:
                    img28 = img.reshape(28, 28).astype(np.uint8)
                    if DO_RANDOM_ROT:
                        angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                        img28 = rotate_keep_size_u8(img28, angle_deg=angle, fill=ROT_FILL)

                payload = to_chw_u8(img28, channel_order=CHANNEL_ORDER)

                if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                    plot_update(
                        im, title_obj, img28,
                        title=f"BloodMNIST iter={k+1}/{N_TIMES} idx={idx} gt={gt} rot={angle:+.1f}° C={n_channels} order={CHANNEL_ORDER}"
                    )
            else:
                x784 = X[idx]
                img28 = x784.reshape(28, 28).astype(np.uint8)
                if DO_RANDOM_ROT:
                    angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                    img28 = rotate_keep_size_u8(img28, angle_deg=angle, fill=ROT_FILL)
                payload = img28.reshape(784).astype(np.uint8)

                if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                    plot_update(
                        im, title_obj, np.stack([img28]*3, axis=-1),
                        title=f"MNIST iter={k+1}/{N_TIMES} idx={idx} gt={gt} rot={angle:+.1f}°"
                    )

            # Send uint8 payload:
            # - MNIST:  1*28*28 = 784 bytes
            # - Blood:  3*28*28 = 2352 bytes (CxWxW stacked planes)
            send_raw_bytes(ser, payload)

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

            if PAUSE_BETWEEN_S > 0:
                time.sleep(PAUSE_BETWEEN_S)

    print(f"\nDone. Parsed {total}/{N_TIMES} predictions.")
    if total > 0:
        acc = 100.0 * correct / total
        print(f"Accuracy (over parsed replies): {correct}/{total} = {acc:.2f}%")

    if SHOW_PLOT_EACH:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
