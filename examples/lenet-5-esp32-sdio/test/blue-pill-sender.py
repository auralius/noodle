import time
import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt

# =========================
# EDIT THESE
# =========================
CSV_PATH    = "./mnist_train.csv"
SERIAL_PORT = "/dev/ttyUSB0"    # USB-UART: /dev/ttyUSB0; USB CDC: /dev/ttyACM0; Windows: COM5
BAUD        = 115200

N_TIMES     = 20
START_INDEX = 0
RANDOM      = False

TIMEOUT_S   = 100.0
SHOW_PLOT_EACH = True
PAUSE_BETWEEN_S = 0.2

PLOT_EVERY = 1

# --- AUGMENTATION ---
DO_RANDOM_ROT = True
ROT_DEG_MIN = -20.0
ROT_DEG_MAX = 20.0
ROT_FILL = 0
# =========================

IMG_SIZE = 784
CHUNK_SIZE = 64


def load_mnist_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    return X, y


def rotate_keep_size_u8(img28_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
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

    xs_src = c * x + s * y + cx
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


def plot_init():
    plt.ion()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    img0 = np.zeros((28, 28), dtype=np.uint8)
    im = ax.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    title_obj = ax.set_title("")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, im, title_obj


def plot_update(im, title_obj, img28_u8: np.ndarray, title: str):
    im.set_data(img28_u8)
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
            #print(f"[rx] {line}")
            if "READY" in line:
                return

    raise TimeoutError("Device never said READY.")


def send_image_chunked(ser: serial.Serial, x784: np.ndarray):
    assert x784.dtype == np.uint8 and x784.size == IMG_SIZE

    payload = x784.tobytes()

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
            #print(f"[rx] {line}")

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

        if line and line.startswith("PRED "):
            return line

        if line and line.startswith("ERR"):
            raise RuntimeError(line)

    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_from_predline(line: str):
    parts = line.strip().split()

    if len(parts) < 4 or parts[0] != "PRED":
        return None, None, None

    pred = None
    seconds = None
    max_val = None

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
    X, y = load_mnist_csv(CSV_PATH)
    n_samples = X.shape[0]

    if not RANDOM and (START_INDEX < 0 or START_INDEX >= n_samples):
        raise ValueError(
            f"START_INDEX out of range: {START_INDEX}, dataset size={n_samples}"
        )

    if SHOW_PLOT_EACH:
        fig, ax, im, title_obj = plot_init()

    print(f"Opening serial {SERIAL_PORT} @ {BAUD} ...")

    with serial.Serial(SERIAL_PORT, BAUD, timeout=0.1, rtscts=False, dsrdtr=False) as ser:
        ser.dtr = False
        ser.rts = False

        time.sleep(0.5)
        wait_for_ready(ser, timeout_s=15.0)

        correct = 0
        total = 0
        times = []

        rng = np.random.default_rng()

        for k in range(N_TIMES):
            idx = int(rng.integers(0, n_samples)) if RANDOM else (START_INDEX + k) % n_samples

            x784 = X[idx]
            gt = int(y[idx])

            img28 = x784.reshape(28, 28)
            angle = 0.0

            if DO_RANDOM_ROT:
                angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                img28 = rotate_keep_size_u8(img28, angle_deg=angle, fill=ROT_FILL)

            x784_send = img28.reshape(784).astype(np.uint8)

            if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                plot_update(
                    im,
                    title_obj,
                    img28,
                    title=f"iter={k+1}/{N_TIMES} idx={idx} gt={gt} rot={angle:+.1f}°",
                )

            try:
                send_image_chunked(ser, x784_send)
                pred_line = read_pred_line(ser, TIMEOUT_S)
            except Exception as e:
                print(
                    f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} "
                    f"rot={angle:+.1f}° -> ERROR {e}"
                )

                try:
                    wait_for_ready(ser, timeout_s=5.0)
                except Exception:
                    pass

                continue

            pred, seconds, max_val = parse_pred_from_predline(pred_line)
            ok = pred == gt

            total += 1
            correct += 1 if ok else 0

            if seconds is not None:
                times.append(seconds)

            extra = ""
            if seconds is not None:
                extra += f"  t={seconds:.3f}s"
            if max_val is not None:
                extra += f"  P={max_val:.4f}"

            print(
                f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} "
                f"rot={angle:+.1f}° -> {pred_line} "
                f"{'OK' if ok else 'NO'}{extra}"
            )

            # Consume the READY after prediction before sending next image.
            try:
                wait_for_ready(ser, timeout_s=5.0)
            except Exception:
                pass

            if PAUSE_BETWEEN_S > 0:
                time.sleep(PAUSE_BETWEEN_S)

    print(f"\nDone. Parsed {total}/{N_TIMES} predictions.")

    if total > 0:
        acc = 100.0 * correct / total
        print(f"Accuracy (over parsed replies): {correct}/{total} = {acc:.2f}%")

    if times:
        print(f"Mean inference: {np.mean(times):.6f} s")

    if SHOW_PLOT_EACH:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
