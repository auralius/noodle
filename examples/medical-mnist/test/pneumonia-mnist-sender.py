import time
import struct
import numpy as np
import serial
import matplotlib.pyplot as plt

# =========================
# EDIT THESE
# =========================
NPZ_PATH    = "./pneumoniamnist_64.npz"
SERIAL_PORT = "/dev/ttyACM0"
BAUD        = 921600

SPLIT = "test"

N_TIMES     = 10
START_INDEX = 0
RANDOM      = False

TIMEOUT_S        = 20.0
SHOW_PLOT_EACH   = False
PAUSE_BETWEEN_S  = 0.0
PLOT_EVERY       = 1

DO_RANDOM_ROT = False
ROT_DEG_MIN   = -7.0
ROT_DEG_MAX   =  7.0
ROT_FILL      = 0

VERBOSE_RX         = True
PRINT_LAYER_TABLE  = True

TX_CHUNK_BYTES = 128
TX_CHUNK_PAUSE_S = 0.002
FRAME_MAGIC = b"IMG0"
# =========================


def load_pneumonia_npz(npz_path: str, split: str):
    data = np.load(npz_path)
    key_i = f"{split}_images"
    key_y = f"{split}_labels"

    if key_i not in data.files or key_y not in data.files:
        raise KeyError(f"NPZ missing keys. Found keys: {data.files}")

    X = data[key_i].astype(np.uint8)
    y = data[key_y].squeeze().astype(np.int64)
    return X, y


def rotate_keep_size_u8(img_u8: np.ndarray, angle_deg: float, fill: int = 0) -> np.ndarray:
    assert img_u8.ndim == 2 and img_u8.dtype == np.uint8

    H, W = img_u8.shape
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
        out[mask] = img_u8[yy[mask], xx[mask]].astype(np.float32)
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


def plot_init(W: int):
    plt.ion()
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    img0 = np.zeros((W, W), dtype=np.uint8)
    im = ax.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
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


def img_to_payload_1ch_u8(img_u8: np.ndarray, W: int) -> np.ndarray:
    if img_u8.ndim != 2 or img_u8.shape != (W, W) or img_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image of shape ({W}, {W}), got {img_u8.shape}, dtype={img_u8.dtype}")
    return img_u8.reshape(-1).astype(np.uint8, copy=False)


def build_frame(payload_u8: np.ndarray) -> bytes:
    payload = payload_u8.tobytes()
    return FRAME_MAGIC + struct.pack("<I", len(payload)) + payload


def send_frame_chunked(ser: serial.Serial, frame: bytes, chunk: int, pause_s: float) -> int:
    total = 0
    for i in range(0, len(frame), chunk):
        part = frame[i:i + chunk]
        n = ser.write(part)
        total += n
        ser.flush()
        if pause_s > 0:
            time.sleep(pause_s)
    return total


class SerialLineReader:
    def __init__(self, ser: serial.Serial, verbose: bool = False):
        self.ser = ser
        self.buf = bytearray()
        self.verbose = verbose

    def read_line(self, timeout_s: float) -> str:
        t0 = time.time()

        while (time.time() - t0) < timeout_s:
            nl = self.buf.find(b"\n")
            if nl != -1:
                line = self.buf[:nl]
                del self.buf[:nl + 1]
                s = line.decode("utf-8", errors="replace").strip()
                if self.verbose and s:
                    print(f"[rx] {s}")
                return s

            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buf.extend(chunk)

        raise TimeoutError("Timed out waiting for a line from serial.")


def drain_startup_lines(reader: SerialLineReader, timeout_s: float = 2.0):
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            line = reader.read_line(timeout_s=0.2)
        except TimeoutError:
            break
        if line:
            print(f"[startup] {line}")


def wait_for_ready_and_mem(reader: SerialLineReader, timeout_s: float = 20.0):
    t0 = time.time()
    meminfo = {}
    saw_ready = False

    while (time.time() - t0) < timeout_s:
        try:
            line = reader.read_line(timeout_s=1.0)
        except TimeoutError:
            continue

        if not line:
            continue

        print(f"[rx] {line}")

        if line.startswith("MEMBUF "):
            parts = line.split()
            if len(parts) == 3:
                try:
                    meminfo[parts[1]] = int(parts[2])
                except ValueError:
                    pass
            continue

        if line == "READY":
            saw_ready = True
            return meminfo

    if not saw_ready:
        print("[warn] Device never said READY during startup; continuing anyway.")
    return meminfo


def parse_layer_line(line: str):
    parts = line.strip().split()
    if len(parts) != 6 or parts[0] != "LAYER":
        return None
    try:
        return {
            "name": parts[1],
            "time_us": int(parts[2]),
            "in_bytes": int(parts[3]),
            "out_bytes": int(parts[4]),
            "live_bytes": int(parts[5]),
        }
    except ValueError:
        return None


def parse_pred_line(line: str):
    parts = line.strip().split()
    if len(parts) < 4 or parts[0] != "PRED":
        return None
    try:
        return {
            "pred": int(parts[1]),
            "seconds": float(parts[2]),
            "prob": float(parts[3]),
        }
    except ValueError:
        return None


def read_one_inference_report(reader: SerialLineReader, timeout_s: float):
    t0 = time.time()
    layers = []

    while (time.time() - t0) < timeout_s:
        try:
            line = reader.read_line(timeout_s=1.0)
        except TimeoutError:
            continue

        if not line:
            continue

        if line.startswith("LAYER "):
            info = parse_layer_line(line)
            if info is not None:
                layers.append(info)
            continue

        if line.startswith("PRED "):
            pred = parse_pred_line(line)
            if pred is not None:
                return layers, pred

        # tolerate READY / GOT_FRAME / BAD_LEN and other debug lines

    raise TimeoutError("Timed out waiting for full inference report.")


def print_layer_report(layers):
    if not layers:
        print("  Layer report: <empty>")
        return

    print("  Layer report:")
    print("    {:<10} {:>10} {:>12} {:>12} {:>12}".format(
        "name", "time_us", "in_bytes", "out_bytes", "live_bytes"
    ))

    for x in layers:
        print("    {:<10} {:>10} {:>12} {:>12} {:>12}".format(
            x["name"], x["time_us"], x["in_bytes"], x["out_bytes"], x["live_bytes"]
        ))

    total_us = sum(x["time_us"] for x in layers)
    worst_time = max(layers, key=lambda z: z["time_us"])
    worst_live = max(layers, key=lambda z: z["live_bytes"])

    print(f"  Sum layer time : {total_us} us ({total_us / 1e6:.6f} s)")
    print(f"  Slowest layer  : {worst_time['name']} ({worst_time['time_us']} us)")
    print(f"  Peak live act. : {worst_live['name']} ({worst_live['live_bytes']} bytes)")


def main():
    X, y = load_pneumonia_npz(NPZ_PATH, split=SPLIT)

    if X.ndim != 3 or X.shape[1:] != (64, 64):
        raise ValueError(f"Expected X shape (N,64,64), got {X.shape}")

    W = 64
    PAYLOAD_BYTES = W * W

    print(f"Loaded PneumoniaMNIST split='{SPLIT}' with X={X.shape}, y={y.shape}")
    print(f"Each image sent as 1-channel grayscale payload: {PAYLOAD_BYTES} bytes")

    n_samples = X.shape[0]
    if not RANDOM and (START_INDEX < 0 or START_INDEX >= n_samples):
        raise ValueError(f"START_INDEX out of range: {START_INDEX}, dataset size={n_samples}")

    if SHOW_PLOT_EACH:
        fig, ax, im, title_obj = plot_init(W)

    print(f"Opening serial {SERIAL_PORT} @ {BAUD} ...")
    with serial.Serial(SERIAL_PORT, BAUD, timeout=0.05) as ser:
        time.sleep(1.2)

        reader = SerialLineReader(ser, verbose=VERBOSE_RX)

        drain_startup_lines(reader, timeout_s=1.0)
        meminfo = wait_for_ready_and_mem(reader, timeout_s=8.0)
        time.sleep(0.2)
        ser.reset_input_buffer()

        if meminfo:
            print("\nPersistent buffers from MCU:")
            for k, v in meminfo.items():
                print(f"  {k:<18} {v} bytes")
            print()

        correct = 0
        total = 0
        rng = np.random.default_rng()

        host_latencies = []
        mcu_latencies = []

        for k in range(N_TIMES):
            idx = int(rng.integers(0, n_samples)) if RANDOM else (START_INDEX + k) % n_samples
            gt = int(y[idx])

            img = X[idx].astype(np.uint8)
            angle = 0.0

            if DO_RANDOM_ROT:
                angle = float(rng.uniform(ROT_DEG_MIN, ROT_DEG_MAX))
                img = rotate_keep_size_u8(img, angle_deg=angle, fill=ROT_FILL)

            payload = img_to_payload_1ch_u8(img, W=W)

            if payload.size != PAYLOAD_BYTES:
                raise RuntimeError(f"Payload size mismatch: got {payload.size}, expected {PAYLOAD_BYTES}")

            if SHOW_PLOT_EACH and (k % max(1, PLOT_EVERY) == 0):
                plot_update(
                    im,
                    title_obj,
                    img,
                    title=f"PneumoniaMNIST iter={k+1}/{N_TIMES} idx={idx} gt={gt} rot={angle:+.1f}°"
                )

            frame = build_frame(payload)

            host_t0 = time.time()
            n = send_frame_chunked(ser, frame, TX_CHUNK_BYTES, TX_CHUNK_PAUSE_S)
            print(f"[tx] sent frame bytes={n}")

            try:
                layers, pred_info = read_one_inference_report(reader, TIMEOUT_S)
            except TimeoutError:
                print(f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} -> TIMEOUT waiting report")
                continue

            host_t1 = time.time()
            host_latency = host_t1 - host_t0

            pred = pred_info["pred"]
            seconds = pred_info["seconds"]
            prob = pred_info["prob"]

            ok = (pred == gt)
            total += 1
            if ok:
                correct += 1

            host_latencies.append(host_latency)
            mcu_latencies.append(seconds)

            print(
                f"[{k+1}/{N_TIMES}] idx={idx} gt={gt} pred={pred} "
                f"prob={prob:.6f} mcu={seconds:.4f}s host={host_latency:.4f}s "
                f"{'OK' if ok else 'NO'}"
            )

            if PRINT_LAYER_TABLE:
                print_layer_report(layers)
                print()

            if PAUSE_BETWEEN_S > 0:
                time.sleep(PAUSE_BETWEEN_S)

    print(f"Done. Parsed {total}/{N_TIMES} predictions.")
    if total > 0:
        acc = 100.0 * correct / total
        print(f"Accuracy: {correct}/{total} = {acc:.2f}%")

        host_arr = np.array(host_latencies, dtype=np.float64)
        mcu_arr  = np.array(mcu_latencies, dtype=np.float64)

        print("\nLatency summary:")
        print(f"  Host mean   : {host_arr.mean():.4f} s")
        print(f"  Host min    : {host_arr.min():.4f} s")
        print(f"  Host max    : {host_arr.max():.4f} s")
        print(f"  MCU mean    : {mcu_arr.mean():.4f} s")
        print(f"  MCU min     : {mcu_arr.min():.4f} s")
        print(f"  MCU max     : {mcu_arr.max():.4f} s")
        print(f"  Host-MCU avg: {(host_arr - mcu_arr).mean():.4f} s")

    if SHOW_PLOT_EACH:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
