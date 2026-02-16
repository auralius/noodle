#!/usr/bin/env python3
import os, glob, time, random, serial, csv
from PIL import Image
import matplotlib.pyplot as plt

PORT="/dev/ttyACM0"
BAUD=115200

PERSON_DIR="person"
NON_PERSON_DIR="non_person"

PREFIX="COCO_val2014"      # required filename prefix
LABELS_CSV="y_labels.csv"  # path to your CSV

W=96
N=W*W*3                    # 27648 bytes/frame

N_PER_CLASS=50             # 50 + 50 = 100 total
TIMEOUT_FRAME_S=40

SHOW_IMAGE = True
SHOW_DELAY_S = 0.001

# ---------- label loading ----------
def load_labels(csv_path: str) -> dict[str, int]:
    """
    y_labels.csv appears to be: filename, 2, label(0/1) with NO header.
    We use col0 -> col2.
    """
    labels = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 3:
                continue
            fname = row[0].strip()
            try:
                lab = int(row[2])
            except Exception:
                continue
            labels[fname] = lab
    return labels

# ---------- image -> payload ----------
def img_and_bytes(path):
    im = Image.open(path).convert("RGB").resize((W,W), Image.BILINEAR)
    b = im.tobytes()  # HWC interleaved RGBRGB...
    if len(b) != N:
        raise ValueError(f"{path}: got {len(b)} bytes, expected {N}")
    return im, b

# ---------- serial helpers ----------
def read_line(ser, timeout=2.0):
    t0=time.time()
    while time.time()-t0<timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            return line
    return None

def parse_pred(line: str):
    # expected: "ms=0 P0=... P1=... pred=0"
    if "pred=" not in line:
        return None
    for tok in line.split():
        if tok.startswith("pred="):
            try:
                return int(tok.split("=", 1)[1])
            except Exception:
                return None
    return None

# ---------- file selection ----------
def pick_files(folder: str, n: int):
    pat = os.path.join(folder, f"{PREFIX}*.jpg")
    files = sorted(glob.glob(pat))
    if len(files) < n:
        raise SystemExit(f"Not enough files in '{folder}' matching {PREFIX}*.jpg: have {len(files)}, need {n}")
    return files[:n]

# ---------- optional live display ----------
def setup_viewer():
    plt.ion()
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_axis_off()
    im_artist = ax.imshow([[0]])
    title_obj = ax.set_title("")
    fig.tight_layout()
    return fig, im_artist, title_obj

def update_viewer(im_pil, title_text, im_artist, title_obj):
    im_artist.set_data(im_pil)
    title_obj.set_text(title_text)
    title_obj.set_fontsize(6)
    plt.pause(SHOW_DELAY_S)

# ---------- main ----------
def main():
    labels = load_labels(LABELS_CSV)
    if not labels:
        raise SystemExit(f"Failed to load labels from {LABELS_CSV}")

    person_files = pick_files(PERSON_DIR, N_PER_CLASS)
    nonperson_files = pick_files(NON_PERSON_DIR, N_PER_CLASS)

    # We still select 50/50 from the two folders, but GT comes from CSV.
    dataset = person_files + nonperson_files
    random.shuffle(dataset)

    print(f"Dataset: total={len(dataset)} (picked {N_PER_CLASS} from each folder)")
    print(f"GT source: {LABELS_CSV} (filename -> label)")
    print(f"Filter: {PREFIX}*.jpg | Frame bytes={N} | Serial {PORT}@{BAUD}")

    total = 0
    correct = 0
    skipped_no_label = 0
    time_ms_list = []

    with serial.Serial(PORT, BAUD, timeout=0.5) as ser:
        time.sleep(1.0)

        # wait for READY
        while True:
            line = read_line(ser, timeout=5.0)
            if line:
                print("[rx]", line)
                if "READY" in line:
                    break

        if SHOW_IMAGE:
            fig, im_artist, title_obj = setup_viewer()
        else:
            fig = im_artist = title_obj = None

        for i, path in enumerate(dataset, 1):
            name = os.path.basename(path)

            # GT from CSV
            gt = labels.get(name, None)
            if gt is None:
                skipped_no_label += 1
                print(f"[{i}/{len(dataset)}] {name} -> SKIP (no label in CSV)")
                continue

            im, payload = img_and_bytes(path)

            if SHOW_IMAGE:
                update_viewer(im, f"[{i}/{len(dataset)}] SENDING  gt={gt}  {name}",
                              im_artist, title_obj)

            ser.write(payload)
            ser.flush()

            # read until pred line and time_ms
            t0=time.time()
            got_pred=False
            got_time=False
            pred=None
            t_ms=None

            while time.time()-t0 < TIMEOUT_FRAME_S:
                line = read_line(ser, timeout=2.0)
                if not line:
                    continue
                # uncomment if you want full logs:
                # print("[rx]", line)

                if ("P0=" in line) and ("P1=" in line) and ("pred=" in line):
                    pred = parse_pred(line)
                    got_pred = (pred is not None)

                if line.startswith("time_ms="):
                    try:
                        t_ms = int(line.split("=", 1)[1])
                        got_time = True
                    except Exception:
                        pass

                if got_pred and got_time:
                    break

            if not got_pred:
                print(f"[{i}/{len(dataset)}] {name} gt={gt} -> TIMEOUT / no pred")
                if SHOW_IMAGE:
                    update_viewer(im, f"[{i}/{len(dataset)}] TIMEOUT  gt={gt}  {name}",
                                  im_artist, title_obj)
                continue

            total += 1
            ok = (pred == gt)
            correct += 1 if ok else 0
            if t_ms is not None:
                time_ms_list.append(t_ms)

            print(f"[{i}/{len(dataset)}] {name} gt={gt} pred={pred} {'OK' if ok else 'NO'} time_ms={t_ms}")

            if SHOW_IMAGE:
                update_viewer(im, f"[{i}/{len(dataset)}] DONE  gt={gt} pred={pred} {'OK' if ok else 'NO'}  {name}",
                              im_artist, title_obj)

        if SHOW_IMAGE:
            plt.ioff()
            plt.show()

    if total == 0:
        print("\nSummary: no predictions received.")
        return

    acc = 100.0 * correct / total
    print("\n====================")
    print(f"Summary: total_scored={total} correct={correct} acc={acc:.2f}%")
    if skipped_no_label:
        print(f"Skipped (no CSV label): {skipped_no_label}")
    if time_ms_list:
        avg = sum(time_ms_list) / len(time_ms_list)
        print(f"Avg time_ms: {avg:.1f} ms over {len(time_ms_list)} samples")
    print("====================")

if __name__ == "__main__":
    main()
