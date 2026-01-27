import os
import glob
import numpy as np
from PIL import Image

# If you have full TensorFlow installed:
import tensorflow as tf

# -------------------
# CONFIG
# -------------------
TFLITE_PATH = "vww_96_float.tflite"  # put next to this script or give full path
FOLDER = "person"                   # or "non_person"
W = 96

# -------------------
# Preprocess: matches training (rescale=1/255), TFLite expects HWC
# -------------------
def load_hwc_float01(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((W, W), Image.BILINEAR)
    hwc_u8 = np.asarray(im, dtype=np.uint8)          # (96,96,3) uint8
    hwc_f  = hwc_u8.astype(np.float32) / 255.0       # (96,96,3) float32 in [0,1]
    return hwc_f

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# -------------------
# Load TFLite
# -------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

print("Input:", in_det["shape"], in_det["dtype"], in_det.get("quantization", None))
print("Output:", out_det["shape"], out_det["dtype"], out_det.get("quantization", None))

# -------------------
# Collect images
# -------------------
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
files = []
for e in exts:
    files.extend(glob.glob(os.path.join(FOLDER, e)))
files = files[0:10]  # LIMIT to 10 images for testing

if not files:
    raise SystemExit(f"No images found in folder: {FOLDER}")

# -------------------
# Run
# -------------------
p1s = []
for i, path in enumerate(files, 1):
    x = load_hwc_float01(path)  # (96,96,3) float32

    # Make batch
    x = np.expand_dims(x, axis=0)  # (1,96,96,3)

    # Some models may want float32; if not, we cast
    if in_det["dtype"] == np.float32:
        x_in = x.astype(np.float32, copy=False)
    else:
        # Unlikely for your float model, but safe
        x_in = x.astype(in_det["dtype"], copy=False)

    interpreter.set_tensor(in_det["index"], x_in)
    interpreter.invoke()

    y = interpreter.get_tensor(out_det["index"])

    # y might already be probabilities, or might be logits.
    # If it doesn't sum ~1, treat as logits and softmax.
    y = np.squeeze(y)
    if y.ndim != 1 or y.size != 2:
        raise RuntimeError(f"Unexpected output shape for {path}: {y.shape}")

    s = float(y[0] + y[1])
    if not (0.95 <= s <= 1.05):
        y = softmax(y)

    p0, p1 = float(y[0]), float(y[1])
    pred = 1 if p1 > p0 else 0
    p1s.append(p1)

    print(f"[{i}/{len(files)}] {os.path.basename(path)} -> P0={p0:.6f} P1={p1:.6f} pred={pred}")

avg_p1 = sum(p1s) / len(p1s)
print(f"\nSummary: n={len(files)} avg_P1={avg_p1:.6f}")
