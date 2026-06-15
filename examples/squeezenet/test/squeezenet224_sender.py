#!/usr/bin/env python3
"""
squeezenet224_sender_revised.py

Robust sender for the Noodle SqueezeNet-1.1 224x224 ImageNet demo.

Protocol:
  wait READY
  send b"IMG"
  wait RDYIMG
  send image in 64-byte chunks
  wait ACK after every 64-byte chunk
  wait PRED
  wait READY before next image

This follows noodle_serial.cpp:
  - wait_for_img_header() looks for bytes I M G
  - recv_image_chunked() prints RDYIMG
  - recv_image_chunked() receives CHUNK_SIZE=64 bytes
  - recv_image_chunked() prints ACK after each 64-byte chunk

Important fix compared with the earlier sender:
  - SerialLineReader keeps leftover bytes between reads, so fast multi-line MCU
    logs are not truncated or mixed.
  - If the MCU prints READY again while we are waiting for RDYIMG, the sender
    re-sends the IMG header. This helps with native USB CDC reset/boot timing.

Label file:
  If imagenet_classes.txt is in the same folder as this script, the sender
  automatically prints the ImageNet label for the predicted class id.
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import serial


SERIAL_PORT = "/dev/ttyACM1"
BAUD = 921600

IMG_W = 224
IMG_H = 224
IMG_C = 3
IMG_SIZE = IMG_W * IMG_H * IMG_C

# Must match noodle_serial.cpp
CHUNK_SIZE = 64

TIMEOUT_S = 600.0

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS_PATH = SCRIPT_DIR / "imagenet_classes.txt"


class SerialLineReader:
    """Stateful serial line reader.

    pyserial can return several MCU log lines in one read. A stateless
    read_line() that creates a new buffer each time will return the first line
    and discard the rest. This class keeps the remaining bytes for the next
    read, so diagnostic messages stay complete.
    """

    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.buf = bytearray()

    def read_line(self, timeout_s: float) -> str:
        t0 = time.time()

        while (time.time() - t0) < timeout_s:
            if b"\n" in self.buf:
                line, _, rest = self.buf.partition(b"\n")
                self.buf = bytearray(rest)
                return line.decode("utf-8", errors="replace").strip()

            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buf.extend(chunk)

        raise TimeoutError("Timed out waiting for a line from serial.")

    def clear(self) -> None:
        self.buf.clear()


def load_labels(path: Optional[str]):
    """Load ImageNet labels, one class name per line."""
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        print(f"[warn] labels file not found: {p}")
        return None

    labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    labels = [x for x in labels if x]

    if len(labels) < 1000:
        print(f"[warn] labels file has only {len(labels)} labels; expected 1000")
    else:
        print(f"[info] loaded {len(labels)} labels from {p}")

    return labels


def label_for(pred: Optional[int], labels) -> Optional[str]:
    if pred is None or labels is None:
        return None
    if 0 <= pred < len(labels):
        return labels[pred]
    return None


def wait_for_ready(reader: SerialLineReader, timeout_s: float = 30.0) -> None:
    t0 = time.time()

    while (time.time() - t0) < timeout_s:
        try:
            line = reader.read_line(timeout_s=5.0)
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")
            if "READY" in line:
                return

    raise TimeoutError("Device never said READY.")


def load_image_224_rgb_u8(path: str, center_crop: bool = True) -> np.ndarray:
    img = Image.open(path).convert("RGB")

    if center_crop:
        # Standard ImageNet-ish preprocessing:
        # resize short side to 256, center crop 224.
        w, h = img.size
        short = min(w, h)
        scale = 256.0 / float(short)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        left = (new_w - IMG_W) // 2
        top = (new_h - IMG_H) // 2
        img = img.crop((left, top, left + IMG_W, top + IMG_H))
    else:
        img = img.resize((IMG_W, IMG_H), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.uint8)

    if arr.shape != (IMG_H, IMG_W, IMG_C):
        raise RuntimeError(f"unexpected image shape: {arr.shape}")

    return arr


def send_img_header_until_rdyimg(ser: serial.Serial, reader: SerialLineReader) -> None:
    """Send IMG until MCU responds with RDYIMG.

    Native USB CDC boards may reboot or miss the first three-byte header. If the
    firmware returns to READY, this function simply sends IMG again.
    """
    while True:
        print("[tx] IMG")
        ser.write(b"IMG")
        ser.flush()

        t0 = time.time()
        while (time.time() - t0) < 10.0:
            try:
                line = reader.read_line(timeout_s=5.0)
            except TimeoutError:
                continue

            if line:
                print(f"[rx] {line}")

            if line == "RDYIMG":
                return

            if line == "READY":
                # MCU timed out waiting for IMG or rebooted. Re-send IMG.
                break

            if line.startswith("ERR"):
                raise RuntimeError(line)


def send_image_chunked(
    ser: serial.Serial,
    reader: SerialLineReader,
    img224_rgb_u8: np.ndarray,
) -> None:
    assert img224_rgb_u8.dtype == np.uint8
    assert img224_rgb_u8.shape == (IMG_H, IMG_W, IMG_C)

    payload = img224_rgb_u8.reshape(-1).tobytes()
    assert len(payload) == IMG_SIZE

    # This already sends IMG and waits until RDYIMG.
    # After this returns, the MCU is waiting for image bytes.
    send_img_header_until_rdyimg(ser, reader)

    offset = 0
    chunk_count = 0

    while offset < len(payload):
        chunk = payload[offset : offset + CHUNK_SIZE]

        ser.write(chunk)
        ser.flush()
        offset += len(chunk)
        chunk_count += 1

        while True:
            line = reader.read_line(timeout_s=5.0)

            if line == "ACK":
                break

            if line:
                print(f"[rx] {line}")

            if line.startswith("ERR"):
                raise RuntimeError(line)

    print(f"[tx] sent {offset} bytes in {chunk_count} chunks")


def read_pred_line(reader: SerialLineReader, timeout_s: float) -> str:
    t0 = time.time()

    while (time.time() - t0) < timeout_s:
        try:
            line = reader.read_line(timeout_s=5.0)
        except TimeoutError:
            continue

        if line:
            print(f"[rx] {line}")

        if line.startswith("PRED "):
            return line

        if line.startswith("ERR"):
            raise RuntimeError(line)

    raise TimeoutError("Timed out waiting for PRED line.")


def parse_pred_from_predline(line: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Send the same image N times without closing the serial port.",
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Path to imagenet_classes.txt. Default: same folder as this sender.",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    img = load_image_224_rgb_u8(args.image, center_crop=not args.no_center_crop)

    print(f"Opening serial {args.port} @ {args.baud} ...")
    print(f"Image payload: {IMG_SIZE} bytes, chunk={CHUNK_SIZE}")

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
        rtscts=False,
        dsrdtr=False,
    ) as ser:
        # Keep native USB CDC boards from being repeatedly reset by control lines.
        ser.dtr = False
        ser.rts = False

        reader = SerialLineReader(ser)

        time.sleep(0.5)
        wait_for_ready(reader, timeout_s=30.0)
        time.sleep(1.0)

        for run_idx in range(args.repeat):
            print()
            print(f"[run] {run_idx + 1}/{args.repeat}")

            send_image_chunked(ser, reader, img)

            pred_line = read_pred_line(reader, args.timeout)
            pred, seconds, max_val = parse_pred_from_predline(pred_line)
            label = label_for(pred, labels)

            extra = ""
            if label is not None:
                extra += f"  label='{label}'"
            if seconds is not None:
                extra += f"  t={seconds:.3f}s"
            if max_val is not None:
                extra += f"  P={max_val:.4f}"

            print(f"[result] run={run_idx + 1} {pred_line}{extra}")

            if pred is not None and label is not None:
                print(f"[label] class_{pred} = {label}")

            # Wait for MCU to become ready before sending the next image.
            wait_for_ready(reader, timeout_s=30.0)


if __name__ == "__main__":
    main()
