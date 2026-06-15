import os
import numpy as np
import tensorflow as tf

def to_two_digit_string(n: int) -> str:
    return f"{n:02d}"

def format_c_array(array: np.ndarray) -> str:
    """Format a 1D numpy array into a C-style array string."""
    lines = []
    line = []
    for i, val in enumerate(array):
        line.append(f"{float(val):.6e}f")
        if (i + 1) % 8 == 0:
            lines.append("  " + ", ".join(line))
            line = []
    if line:
        lines.append("  " + ", ".join(line))
    return ",\n".join(lines)

def _is_nd(w, n: int) -> bool:
    return hasattr(w, "shape") and len(w.shape) == n

def _is_1d(w) -> bool:
    return _is_nd(w, 1)

def _same_len_1d(ws) -> bool:
    """All 1D and identical length."""
    if not all(_is_1d(x) for x in ws):
        return False
    L = int(ws[0].shape[0])
    return all(int(x.shape[0]) == L for x in ws)

def _write_array_txt_and_h(out_dir, prefix, idx, arr_1d, header_lines=None):
    """Write both .txt and .h for a 1D float array."""
    if header_lines is None:
        header_lines = []

    arr_1d = np.asarray(arr_1d, dtype=np.float32).reshape(-1)

    fn_txt = os.path.join(out_dir, f"{prefix}{to_two_digit_string(idx)}.txt")
    print(fn_txt)
    np.savetxt(fn_txt, arr_1d, fmt="%.6e", newline="\n")

    fn_h = fn_txt.replace(".txt", ".h")
    print(fn_h)
    var_name = f"{prefix}{to_two_digit_string(idx)}"
    with open(fn_h, "w") as f:
        f.write("#pragma once\n\n")
        for line in header_lines:
            f.write(line.rstrip() + "\n")
        f.write(f"static const float {var_name}[] = {{\n")
        f.write(format_c_array(arr_1d))
        f.write("\n};\n")

def _consume_bias_and_bn(weights, k_after_kernel, out_dir, b_idx, bn_idx):
    """
    After a kernel tensor, consume (in order) one of:
      (A) bias + BN : 5 consecutive 1D vectors same length
      (B) BN only   : 4 consecutive 1D vectors same length
      (C) bias only : 1 consecutive 1D vector
    Returns: (new_k, b_idx, bn_idx)
    Where new_k is the next index to process.
    """
    i = k_after_kernel

    # (A) bias + BN
    if i + 4 < len(weights) and _same_len_1d(weights[i:i+5]):
        # bias
        b = np.float32(weights[i].flatten())
        b_idx += 1
        _write_array_txt_and_h(out_dir, "b", b_idx, b)

        # BN packed: gamma, beta, mean, var
        gamma = np.float32(weights[i+1].flatten())
        beta  = np.float32(weights[i+2].flatten())
        mean  = np.float32(weights[i+3].flatten())
        var   = np.float32(weights[i+4].flatten())
        packed = np.concatenate([gamma, beta, mean, var], axis=0)

        bn_idx += 1
        _write_array_txt_and_h(
            out_dir, "bn", bn_idx, packed,
            header_lines=[
                "// kind=batchnorm packed",
                "// order: gamma(C), beta(C), mean(C), var(C)",
                f"// C={int(gamma.shape[0])}",
            ],
        )

        return i + 5, b_idx, bn_idx

    # (B) BN only
    if i + 3 < len(weights) and _same_len_1d(weights[i:i+4]):
        gamma = np.float32(weights[i].flatten())
        beta  = np.float32(weights[i+1].flatten())
        mean  = np.float32(weights[i+2].flatten())
        var   = np.float32(weights[i+3].flatten())
        packed = np.concatenate([gamma, beta, mean, var], axis=0)

        bn_idx += 1
        _write_array_txt_and_h(
            out_dir, "bn", bn_idx, packed,
            header_lines=[
                "// kind=batchnorm packed",
                "// order: gamma(C), beta(C), mean(C), var(C)",
                f"// C={int(gamma.shape[0])}",
            ],
        )

        return i + 4, b_idx, bn_idx

    # (C) bias only
    if i < len(weights) and _is_1d(weights[i]):
        b = np.float32(weights[i].flatten())
        b_idx += 1
        _write_array_txt_and_h(out_dir, "b", b_idx, b)
        return i + 1, b_idx, bn_idx

    return i, b_idx, bn_idx

def exporter(weights, out_dir: str):
    """
    Export Keras weights (from model.get_weights()) into Noodle-friendly files.

    Supported tensors:
    - 4D Conv2D:            (Kh, Kw, Cin, Cout) -> OIHW : [Cout, Cin, Kh, Kw]
    - 4D DepthwiseConv2D:   (Kh, Kw, Cin, M)    -> CIMHW: [Cin, M, Kh, Kw]
      NOTE: For your MobileLeNet we assume M==1 for DWConv2D; detection is therefore deterministic.

    - 3D Conv1D:            (K, Cin, Cout)      -> OIC : [Cout, Cin, K]
    - 3D DepthwiseConv1D:   (K, Cin, M)         -> CMK : [Cin, M, K]

    - 2D Dense:             (Din, Dout)         -> stored as (Dout, Din) row-major (transpose then flatten)

    Bias and BN:
    - Bias/BN are consumed ONLY immediately after a kernel tensor, in order:
        (A) bias + BN (5x 1D same length)
        (B) BN only   (4x 1D same length)
        (C) bias only (1x 1D)
      This prevents bias being mis-detected as BN elsewhere.
    """
    if not out_dir.endswith("/"):
        out_dir += "/"
    os.makedirs(out_dir, exist_ok=True)

    w_idx = 0
    b_idx = 0
    bn_idx = 0

    k = 0
    while k < len(weights):
        w = weights[k]

        # ---------- 4D: Conv2D / DWConv2D ----------
        if _is_nd(w, 4):
            Kh, Kw, Cin, C4 = w.shape

            # Deterministic for your MobileLeNet-style DW:
            # DepthwiseConv2D kernel is (Kh, Kw, Cin, M), and in your case M == 1.
            if (Kh == 1 and Kw == 1):
                kind, layout = "conv2d", "OIHW"
            elif (C4 == 1) and (Cin >= 2) and not (Kh == 1 and Kw == 1):
                kind, layout = "depthwise2d", "CIMHW"
            else:
                kind, layout = "conv2d", "OIHW"

            w_idx += 1

            if kind == "conv2d":
                # (Kh, Kw, Cin, Cout) -> (Cout, Cin, Kh, Kw)
                w_oihw = np.transpose(w, (3, 2, 0, 1)).astype(np.float32)
                flat = w_oihw.flatten(order="C")
                header = [
                    f"// kind={kind}, layout={layout}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, Cout={C4}",
                ]
            else:
                # depthwise2d (assume depth_multiplier M==1 for your model):
                # Keras: (Kh, Kw, Cin, 1) -> Noodle: (Cin, Kh, Kw)
                if int(C4) != 1:
                    raise ValueError(f"DepthwiseConv2D depth_multiplier != 1 (got M={int(C4)}). "
                                    "Your firmware DW expects M==1. Update firmware/exporter.")

                dw = w[:, :, :, 0]  # (Kh, Kw, Cin)
                w_ckk = np.transpose(dw, (2, 0, 1)).astype(np.float32)  # (Cin, Kh, Kw)
                flat = w_ckk.flatten(order="C")

                header = [
                    f"// kind=depthwise2d, layout=CKK",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, M=1, Cout={Cin}",
                ]

            _write_array_txt_and_h(out_dir, "w", w_idx, flat, header_lines=header)

            # Consume optional bias+BN immediately after this kernel
            k, b_idx, bn_idx = _consume_bias_and_bn(weights, k + 1, out_dir, b_idx, bn_idx)
            continue

        # ---------- 3D: Conv1D / DWConv1D ----------
        if _is_nd(w, 3):
            K1, Cin, C3 = w.shape  # C3 is Cout (conv1d) or M (dwconv1d)

            # Use lookahead length if next is 1D (bias or BN gamma)
            next_len = None
            if k + 1 < len(weights) and _is_1d(weights[k + 1]):
                next_len = int(weights[k + 1].shape[0])

            # Decide:
            if next_len == int(C3):
                kind, layout = "conv1d", "OIC"
            elif next_len == int(Cin * C3):
                kind, layout = "depthwise1d", "CMK"
            elif (C3 <= 4) and (Cin >= 2):
                kind, layout = "depthwise1d", "CMK"
            else:
                kind, layout = "conv1d", "OIC"

            w_idx += 1

            if kind == "conv1d":
                # (K, Cin, Cout) -> (Cout, Cin, K)
                w_oik = np.transpose(w, (2, 1, 0)).astype(np.float32)
                flat = w_oik.flatten(order="C")
                header = [
                    f"// kind={kind}, layout={layout}",
                    f"// dims: K={K1}, Cin={Cin}, Cout={C3}",
                ]
            else:
                # depthwise1d: (K, Cin, M) -> (Cin, M, K)
                M = int(C3)
                w_cmk = np.transpose(w, (1, 2, 0)).astype(np.float32)
                flat = w_cmk.flatten(order="C")
                header = [
                    f"// kind={kind}, layout={layout}",
                    f"// dims: K={K1}, Cin={Cin}, M={M}, Cout={Cin*M}",
                ]

            _write_array_txt_and_h(out_dir, "w", w_idx, flat, header_lines=header)

            # Consume optional bias+BN immediately after this kernel
            k, b_idx, bn_idx = _consume_bias_and_bn(weights, k + 1, out_dir, b_idx, bn_idx)
            continue

        # ---------- 2D: Dense ----------
        if _is_nd(w, 2):
            w_idx += 1
            # Dense kernel: (Din, Dout) -> store as (Dout, Din)
            flat = np.float32(w.transpose().flatten())
            _write_array_txt_and_h(out_dir, "w", w_idx, flat, header_lines=["// kind=dense (stored OI)"])

            # Consume optional bias immediately after this kernel
            k, b_idx, bn_idx = _consume_bias_and_bn(weights, k + 1, out_dir, b_idx, bn_idx)
            continue

        # ---------- 1D standalone (rare): treat as bias ----------
        if _is_1d(w):
            b_idx += 1
            arr = np.float32(w.flatten())
            _write_array_txt_and_h(out_dir, "b", b_idx, arr, header_lines=["// standalone 1D (treated as bias)"])
            k += 1
            continue

        print("Skipping unsupported tensor with shape:", getattr(w, "shape", None))
        k += 1

###############################################################################

# -------------------------------------------------------------------------------------------------
# TFLite front-end: extract weights in execution order and reuse exporter() unchanged.
#
# Goal: "do exactly as current model exporter but with a .tflite input".
# - Reconstruct Keras-like kernel layouts:
#     Conv2D:          HWIO  (Kh, Kw, Cin, Cout)
#     DepthwiseConv2D: HWIM  (Kh, Kw, Cin, M)
#     Dense:           (Din, Dout)
# - Then pass the list to exporter(weights, out_dir) so flattening + channel stacking stays identical.
#
# Notes:
# - BatchNorm is typically folded into Conv/DWConv in TFLite, so you will usually only see kernel+bias.
# - This implementation targets FLOAT models (float32 weights/bias).
# -------------------------------------------------------------------------------------------------

def _tflite_get_tensor(interpreter, tensor_index: int):
    """Return numpy tensor for a TFLite tensor index, or None if not readable."""
    if tensor_index is None or int(tensor_index) < 0:
        return None
    try:
        return interpreter.get_tensor(int(tensor_index))
    except Exception:
        return None

def _tflite_conv2d_to_keras_hwio(w_raw: np.ndarray) -> np.ndarray:
    """TFLite Conv2D weights are commonly OHWI: (Cout, Kh, Kw, Cin).
    Convert to Keras HWIO: (Kh, Kw, Cin, Cout).
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("Conv2D kernel must be 4D")
    return np.transpose(w_raw, (1, 2, 3, 0)).astype(np.float32)

def _tflite_dwconv2d_to_keras_hwim(w_raw: np.ndarray, cin: int) -> np.ndarray:
    """Keras DepthwiseConv2D kernel: (Kh, Kw, Cin, M)

    Common TFLite layout for DEPTHWISE_CONV_2D is either:
      - already HWIM (Kh, Kw, Cin, M), OR
      - (1, Kh, Kw, Cout) with Cout=Cin*M
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("DepthwiseConv2D kernel must be 4D")

    sh = tuple(int(x) for x in w_raw.shape)

    # already HWIM
    if sh[0] != 1 and sh[2] == int(cin):
        return w_raw.astype(np.float32)

    # (1, Kh, Kw, Cout) -> HWIM
    if sh[0] == 1:
        kh, kw, cout = sh[1], sh[2], sh[3]
        if cin <= 0 or (cout % cin) != 0:
            raise ValueError(f"DEPTHWISE_CONV_2D: cannot infer M from cin={cin}, cout={cout}")
        m = cout // cin
        w = w_raw[0, :, :, :]          # (Kh, Kw, Cout)
        return w.reshape((kh, kw, cin, m)).astype(np.float32)

    return w_raw.astype(np.float32)

def _tflite_dense_to_keras_din_dout(w_raw: np.ndarray, dout_hint: int | None) -> np.ndarray:
    """TFLite FULLY_CONNECTED weights are commonly (Dout, Din).
    Convert to Keras Dense kernel (Din, Dout).
    """
    if w_raw is None or w_raw.ndim != 2:
        raise ValueError("Dense kernel must be 2D")

    if dout_hint is not None:
        dout_hint = int(dout_hint)
        if int(w_raw.shape[0]) == dout_hint:
            return w_raw.transpose().astype(np.float32)
        if int(w_raw.shape[1]) == dout_hint:
            return w_raw.astype(np.float32)

    # default: treat as (Dout, Din)
    return w_raw.transpose().astype(np.float32)

def weights_from_tflite(tflite_path: str) -> list:
    """Extract a Keras-like weights list from a float .tflite file.
    The returned list is compatible with exporter(weights, out_dir).
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    tensor_details = {d["index"]: d for d in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()  # execution order

    out_weights = []

    for op in ops:
        op_name = op.get("op_name", "")

        ins = op.get("inputs", None)
        ins = [] if ins is None else list(ins)

        if len(ins) < 2:
            continue

        # Infer Cin from input activation (NHWC)
        in0 = int(ins[0])
        in0_shape = tensor_details.get(in0, {}).get("shape", None)
        cin = None
        if in0_shape is not None and len(in0_shape) == 4:
            cin = int(in0_shape[3])

        w_raw = _tflite_get_tensor(interpreter, int(ins[1]))
        b_raw = _tflite_get_tensor(interpreter, int(ins[2])) if len(ins) >= 3 else None

        # float-only path
        if w_raw is None or w_raw.dtype != np.float32:
            continue
        if b_raw is not None and b_raw.dtype != np.float32:
            b_raw = None

        if op_name == "CONV_2D":
            out_weights.append(_tflite_conv2d_to_keras_hwio(w_raw))
            if b_raw is not None:
                out_weights.append(np.float32(b_raw).reshape(-1))

        elif op_name == "DEPTHWISE_CONV_2D":
            if cin is None:
                raise ValueError("DEPTHWISE_CONV_2D: cannot infer Cin from input shape.")
            out_weights.append(_tflite_dwconv2d_to_keras_hwim(w_raw, cin))
            if b_raw is not None:
                out_weights.append(np.float32(b_raw).reshape(-1))

        elif op_name == "FULLY_CONNECTED":
            dout_hint = int(b_raw.shape[0]) if b_raw is not None else None
            out_weights.append(_tflite_dense_to_keras_din_dout(w_raw, dout_hint))
            if b_raw is not None:
                out_weights.append(np.float32(b_raw).reshape(-1))

        # ignore activations, reshape, etc.

    return out_weights

def exporter_tflite(tflite_path: str, out_dir: str):
    """Export a float .tflite model into Noodle-friendly files using the same layout as exporter()."""
    ws = weights_from_tflite(tflite_path)
    exporter(ws, out_dir)

def debug_tflite_ops(tflite_path):
    itp = tf.lite.Interpreter(model_path=tflite_path)
    itp.allocate_tensors()
    td = {d["index"]: d for d in itp.get_tensor_details()}
    ops = itp._get_ops_details()

    allowed = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}

    for i, op in enumerate(ops):
        op_name = op.get("op_name", "")
        ins = list(op.get("inputs", []))
        if op_name not in allowed or len(ins) < 2:
            continue
        w = itp.get_tensor(ins[1]) if ins[1] >= 0 else None
        b = itp.get_tensor(ins[2]) if len(ins) >= 3 and ins[2] >= 0 else None
        print(i, op_name, "W:", (None if w is None else (w.shape, w.dtype)),
            "B:", (None if b is None else (b.shape, b.dtype)))


# =============================================================================
# Keras Model front-end with Conv2DTranspose support
# =============================================================================
#
# Why this front-end exists:
# model.get_weights() alone cannot reliably distinguish Conv2D from
# Conv2DTranspose because both are 4D tensors. Conv2D uses Keras layout
#   (Kh, Kw, Cin, Cout)
# while Conv2DTranspose uses
#   (Kh, Kw, Cout, Cin)
#
# Therefore, for autoencoders with Conv2DTranspose, use exporter_model(model,...)
# instead of exporter(model.get_weights(),...).
#
# Noodle layout for both Conv2D and Conv2DTranspose is:
#   [O][I][Kh][Kw]  flattened in C order.
#
# For Conv2D:
#   Keras (Kh, Kw, Cin, Cout) -> Noodle (Cout, Cin, Kh, Kw)
#
# For Conv2DTranspose:
#   Keras (Kh, Kw, Cout, Cin) -> Noodle (Cout, Cin, Kh, Kw)
#
# We already validated the Conv2DTranspose export with NO spatial kernel flip.
# =============================================================================

def _write_bias_if_present(out_dir: str, b_idx: int, weights: list) -> int:
    """Write bias if the second item in layer.get_weights() is a 1D vector."""
    if len(weights) >= 2 and _is_1d(weights[1]):
        b_idx += 1
        _write_array_txt_and_h(out_dir, "b", b_idx, np.float32(weights[1].reshape(-1)))
    return b_idx

def _write_bn_from_layer(out_dir: str, bn_idx: int, weights: list) -> int:
    """Write BatchNormalization weights as packed gamma,beta,mean,var."""
    if len(weights) != 4 or not _same_len_1d(weights):
        raise ValueError("BatchNormalization layer must have gamma,beta,mean,var as four same-length 1D arrays.")

    gamma = np.float32(weights[0].reshape(-1))
    beta  = np.float32(weights[1].reshape(-1))
    mean  = np.float32(weights[2].reshape(-1))
    var   = np.float32(weights[3].reshape(-1))
    packed = np.concatenate([gamma, beta, mean, var], axis=0)

    bn_idx += 1
    _write_array_txt_and_h(
        out_dir, "bn", bn_idx, packed,
        header_lines=[
            "// kind=batchnorm packed",
            "// order: gamma(C), beta(C), mean(C), var(C)",
            f"// C={int(gamma.shape[0])}",
        ],
    )
    return bn_idx

def exporter_model(model, out_dir: str):
    """
    Export a Keras model layer-by-layer into Noodle-friendly files.

    Use this function for models containing Conv2DTranspose, because
    exporter(model.get_weights(), ...) cannot infer the layer type from a
    raw 4D weight tensor.

    Supported Keras layers:
    - Conv2D:
        Keras  (Kh, Kw, Cin, Cout)
        Noodle (Cout, Cin, Kh, Kw), file wXX

    - Conv2DTranspose:
        Keras  (Kh, Kw, Cout, Cin)
        Noodle (Cout, Cin, Kh, Kw), file wXX
        No spatial flip.

    - DepthwiseConv2D with depth_multiplier=1:
        Keras  (Kh, Kw, Cin, 1)
        Noodle (Cin, Kh, Kw), file wXX

    - Conv1D:
        Keras  (K, Cin, Cout)
        Noodle (Cout, Cin, K), file wXX

    - Dense:
        Keras  (Din, Dout)
        Noodle (Dout, Din), file wXX

    - BatchNormalization:
        packed as gamma,beta,mean,var, file bnXX

    Bias vectors are written as bXX immediately after the corresponding
    weighted layer in layer traversal order.
    """
    if not out_dir.endswith("/"):
        out_dir += "/"
    os.makedirs(out_dir, exist_ok=True)

    w_idx = 0
    b_idx = 0
    bn_idx = 0

    for layer in model.layers:
        cls = layer.__class__.__name__
        ws = layer.get_weights()

        if len(ws) == 0:
            continue

        # ---------- Conv2D ----------
        if cls == "Conv2D":
            W = ws[0]
            if not _is_nd(W, 4):
                raise ValueError(f"{layer.name}: Conv2D kernel must be 4D, got {W.shape}")
            Kh, Kw, Cin, Cout = W.shape

            w_idx += 1
            Wn = np.transpose(W, (3, 2, 0, 1)).astype(np.float32)  # Cout,Cin,Kh,Kw
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=conv2d, layout=OIHW",
                    f"// layer={layer.name}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, Cout={Cout}",
                    "// Keras: (Kh,Kw,Cin,Cout) -> Noodle: (Cout,Cin,Kh,Kw)",
                ],
            )
            b_idx = _write_bias_if_present(out_dir, b_idx, ws)
            continue

        # ---------- Conv2DTranspose ----------
        if cls == "Conv2DTranspose":
            W = ws[0]
            if not _is_nd(W, 4):
                raise ValueError(f"{layer.name}: Conv2DTranspose kernel must be 4D, got {W.shape}")
            Kh, Kw, Cout, Cin = W.shape

            w_idx += 1
            Wn = np.transpose(W, (2, 3, 0, 1)).astype(np.float32)  # Cout,Cin,Kh,Kw
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=conv2d_transpose, layout=OIHW",
                    f"// layer={layer.name}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, Cout={Cout}",
                    "// Keras: (Kh,Kw,Cout,Cin) -> Noodle: (Cout,Cin,Kh,Kw)",
                    "// spatial_flip=false",
                ],
            )
            b_idx = _write_bias_if_present(out_dir, b_idx, ws)
            continue

        # ---------- DepthwiseConv2D ----------
        if cls == "DepthwiseConv2D":
            W = ws[0]
            if not _is_nd(W, 4):
                raise ValueError(f"{layer.name}: DepthwiseConv2D kernel must be 4D, got {W.shape}")
            Kh, Kw, Cin, M = W.shape
            if int(M) != 1:
                raise ValueError(f"{layer.name}: depth_multiplier={int(M)} not supported by current Noodle DW path.")

            w_idx += 1
            Wn = np.transpose(W[:, :, :, 0], (2, 0, 1)).astype(np.float32)  # Cin,Kh,Kw
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=depthwise2d, layout=CKK",
                    f"// layer={layer.name}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, M=1, Cout={Cin}",
                ],
            )
            b_idx = _write_bias_if_present(out_dir, b_idx, ws)
            continue

        # ---------- Conv1D ----------
        if cls == "Conv1D":
            W = ws[0]
            if not _is_nd(W, 3):
                raise ValueError(f"{layer.name}: Conv1D kernel must be 3D, got {W.shape}")
            K1, Cin, Cout = W.shape

            w_idx += 1
            Wn = np.transpose(W, (2, 1, 0)).astype(np.float32)  # Cout,Cin,K
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=conv1d, layout=OIC",
                    f"// layer={layer.name}",
                    f"// dims: K={K1}, Cin={Cin}, Cout={Cout}",
                ],
            )
            b_idx = _write_bias_if_present(out_dir, b_idx, ws)
            continue

        # ---------- Dense ----------
        if cls == "Dense":
            W = ws[0]
            if not _is_nd(W, 2):
                raise ValueError(f"{layer.name}: Dense kernel must be 2D, got {W.shape}")
            Din, Dout = W.shape

            w_idx += 1
            Wn = W.transpose().astype(np.float32)  # Dout,Din
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=dense, layout=OI",
                    f"// layer={layer.name}",
                    f"// dims: Din={Din}, Dout={Dout}",
                ],
            )
            b_idx = _write_bias_if_present(out_dir, b_idx, ws)
            continue

        # ---------- BatchNormalization ----------
        if cls == "BatchNormalization":
            bn_idx = _write_bn_from_layer(out_dir, bn_idx, ws)
            continue

        print(f"Skipping unsupported weighted layer {layer.name} ({cls}) with shapes {[w.shape for w in ws]}")

    print(f"Export complete: w={w_idx}, b={b_idx}, bn={bn_idx}, out_dir={out_dir}")

# =============================================================================
# Optional TFLite helper for TRANSPOSE_CONV
# =============================================================================
#
# TFLite TRANSPOSE_CONV layout may vary by converter/version. The common float
# layout is (Cout, Kh, Kw, Cin). This helper converts that to Keras
# Conv2DTranspose layout (Kh, Kw, Cout, Cin).
#
# For highest confidence with autoencoders, prefer exporter_model(keras_model,...)
# directly from the original Keras model. Use TFLite export only after checking
# debug_tflite_ops().
# =============================================================================

def _tflite_transpose_conv2d_to_keras_hwoi(w_raw: np.ndarray) -> np.ndarray:
    """Convert common TFLite TRANSPOSE_CONV kernel to Keras Conv2DTranspose layout.

    Common TFLite: (Cout, Kh, Kw, Cin)
    Keras:         (Kh, Kw, Cout, Cin)
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("TRANSPOSE_CONV kernel must be 4D")

    # Common TFLite layout: (Cout, Kh, Kw, Cin)
    return np.transpose(w_raw, (1, 2, 0, 3)).astype(np.float32)



# =============================================================================
# TFLite direct exporter with TRANSPOSE_CONV support
# =============================================================================
#
# This overrides the earlier exporter_tflite() definition. The older TFLite path
# reconstructed a Keras-like weights list and then reused exporter(). That is not
# safe for Conv2DTranspose because a raw 4D tensor cannot be distinguished from
# Conv2D by shape alone:
#
#   Conv2D Keras:            (Kh, Kw, Cin,  Cout)
#   Conv2DTranspose Keras:   (Kh, Kw, Cout, Cin)
#
# Therefore this direct path writes Noodle layout immediately while walking TFLite
# ops in execution order.
#
# Noodle layouts written:
#   CONV_2D:          [O][I][Kh][Kw]
#   TRANSPOSE_CONV:   [O][I][Kh][Kw]
#   DEPTHWISE_CONV_2D [C][Kh][Kw]  (only depth_multiplier=1)
#   FULLY_CONNECTED:  [O][I]
# =============================================================================

def _shape_tuple(x):
    return tuple(int(v) for v in getattr(x, "shape", []))

def _tensor_shape_from_details(tensor_details, idx):
    d = tensor_details.get(int(idx), None)
    if d is None:
        return None
    sh = d.get("shape", None)
    if sh is None:
        return None
    return tuple(int(v) for v in sh)

def _find_first_float_tensor_input(interpreter, inputs, ndim=None, exclude_positions=None):
    """Return (position, tensor_index, tensor) for the first readable float32 input tensor."""
    if exclude_positions is None:
        exclude_positions = set()
    for pos, idx in enumerate(inputs):
        if pos in exclude_positions:
            continue
        if int(idx) < 0:
            continue
        t = _tflite_get_tensor(interpreter, int(idx))
        if t is None:
            continue
        if t.dtype != np.float32:
            continue
        if ndim is not None and t.ndim != ndim:
            continue
        return pos, int(idx), t
    return None, None, None

def _find_bias_input(interpreter, inputs, expected_len=None, exclude_positions=None):
    """Find a readable 1D float32 bias vector among op inputs."""
    if exclude_positions is None:
        exclude_positions = set()
    for pos, idx in enumerate(inputs):
        if pos in exclude_positions:
            continue
        if int(idx) < 0:
            continue
        t = _tflite_get_tensor(interpreter, int(idx))
        if t is None or t.dtype != np.float32 or t.ndim != 1:
            continue
        if expected_len is not None and int(t.shape[0]) != int(expected_len):
            continue
        return np.float32(t).reshape(-1)
    return None

def _tflite_conv2d_to_noodle_oihw(w_raw: np.ndarray, cin_hint=None, cout_hint=None) -> np.ndarray:
    """Convert TFLite CONV_2D kernel to Noodle [O][I][Kh][Kw].

    Common TFLite layout is OHWI: (Cout, Kh, Kw, Cin).
    Some tooling may expose HWIO: (Kh, Kw, Cin, Cout), so hints are used.
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("CONV_2D kernel must be 4D")

    sh = _shape_tuple(w_raw)

    # Common TFLite OHWI -> OIHW
    if cout_hint is not None and cin_hint is not None:
        if sh[0] == int(cout_hint) and sh[3] == int(cin_hint):
            return np.transpose(w_raw, (0, 3, 1, 2)).astype(np.float32)
        # Already Keras HWIO -> OIHW
        if sh[2] == int(cin_hint) and sh[3] == int(cout_hint):
            return np.transpose(w_raw, (3, 2, 0, 1)).astype(np.float32)

    # Default TFLite OHWI
    return np.transpose(w_raw, (0, 3, 1, 2)).astype(np.float32)

def _tflite_transpose_conv_to_noodle_oihw(w_raw: np.ndarray, cin_hint=None, cout_hint=None) -> np.ndarray:
    """Convert TFLite TRANSPOSE_CONV kernel to Noodle [O][I][Kh][Kw].

    Common TFLite TRANSPOSE_CONV weight layout:
      (Cout, Kh, Kw, Cin)

    Keras Conv2DTranspose layout:
      (Kh, Kw, Cout, Cin)

    Noodle expects:
      (Cout, Cin, Kh, Kw)

    No spatial kernel flip is applied.
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("TRANSPOSE_CONV kernel must be 4D")

    sh = _shape_tuple(w_raw)

    if cout_hint is not None and cin_hint is not None:
        cout_hint = int(cout_hint)
        cin_hint = int(cin_hint)

        # Common TFLite: (Cout, Kh, Kw, Cin)
        if sh[0] == cout_hint and sh[3] == cin_hint:
            return np.transpose(w_raw, (0, 3, 1, 2)).astype(np.float32)

        # Keras-style: (Kh, Kw, Cout, Cin)
        if sh[2] == cout_hint and sh[3] == cin_hint:
            return np.transpose(w_raw, (2, 3, 0, 1)).astype(np.float32)

        # Some flatbuffer dumps/tools may expose (Kh, Kw, Cin, Cout)
        if sh[2] == cin_hint and sh[3] == cout_hint:
            return np.transpose(w_raw, (3, 2, 0, 1)).astype(np.float32)

    # Default to common TFLite layout: (Cout, Kh, Kw, Cin)
    return np.transpose(w_raw, (0, 3, 1, 2)).astype(np.float32)

def _tflite_dwconv2d_to_noodle_ckk(w_raw: np.ndarray, cin_hint=None) -> np.ndarray:
    """Convert TFLite DEPTHWISE_CONV_2D kernel to Noodle [C][Kh][Kw].

    Current Noodle depthwise path supports depth_multiplier=1 only.
    """
    if w_raw is None or w_raw.ndim != 4:
        raise ValueError("DEPTHWISE_CONV_2D kernel must be 4D")

    sh = _shape_tuple(w_raw)

    # TFLite common: (1, Kh, Kw, Cout), Cout = Cin * M
    if sh[0] == 1:
        kh, kw, cout = sh[1], sh[2], sh[3]
        if cin_hint is None:
            raise ValueError("DEPTHWISE_CONV_2D: cannot infer Cin from input shape.")
        cin_hint = int(cin_hint)
        if cout % cin_hint != 0:
            raise ValueError(f"DEPTHWISE_CONV_2D: cannot infer depth_multiplier from Cin={cin_hint}, Cout={cout}.")
        m = cout // cin_hint
        if m != 1:
            raise ValueError(f"DEPTHWISE_CONV_2D depth_multiplier={m}; current Noodle DW path expects M=1.")
        # (1, Kh, Kw, Cin) -> (Cin, Kh, Kw)
        return np.transpose(w_raw[0, :, :, :], (2, 0, 1)).astype(np.float32)

    # Keras-like HWIM: (Kh, Kw, Cin, M)
    if cin_hint is not None and sh[2] == int(cin_hint):
        m = sh[3]
        if m != 1:
            raise ValueError(f"DEPTHWISE_CONV_2D depth_multiplier={m}; current Noodle DW path expects M=1.")
        return np.transpose(w_raw[:, :, :, 0], (2, 0, 1)).astype(np.float32)

    # Last-resort guess: HWIM with M=1
    if sh[3] == 1:
        return np.transpose(w_raw[:, :, :, 0], (2, 0, 1)).astype(np.float32)

    raise ValueError(f"Unsupported DEPTHWISE_CONV_2D kernel layout/shape: {sh}")

def _tflite_dense_to_noodle_oi(w_raw: np.ndarray, dout_hint=None) -> np.ndarray:
    """Convert TFLite FULLY_CONNECTED weights to Noodle [O][I].

    Common TFLite layout is already (Dout, Din).
    """
    if w_raw is None or w_raw.ndim != 2:
        raise ValueError("FULLY_CONNECTED kernel must be 2D")

    if dout_hint is not None:
        dout_hint = int(dout_hint)
        if int(w_raw.shape[0]) == dout_hint:
            return w_raw.astype(np.float32)
        if int(w_raw.shape[1]) == dout_hint:
            return w_raw.transpose().astype(np.float32)

    return w_raw.astype(np.float32)

def exporter_tflite(tflite_path: str, out_dir: str):
    """Export a float .tflite model into Noodle-friendly files.

    Supports CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and TRANSPOSE_CONV.
    The function walks TFLite ops in execution order and writes wXX/bXX files
    directly, so Conv2DTranspose tensors are not confused with Conv2D tensors.
    """
    if not out_dir.endswith("/"):
        out_dir += "/"
    os.makedirs(out_dir, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    tensor_details = {int(d["index"]): d for d in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()

    w_idx = 0
    b_idx = 0

    for op_i, op in enumerate(ops):
        op_name = op.get("op_name", "")
        ins = list(op.get("inputs", []))
        outs = list(op.get("outputs", []))

        if op_name == "CONV_2D":
            if len(ins) < 2:
                continue

            in_shape = _tensor_shape_from_details(tensor_details, ins[0])
            out_shape = _tensor_shape_from_details(tensor_details, outs[0]) if outs else None
            cin_hint = in_shape[3] if in_shape is not None and len(in_shape) == 4 else None
            cout_hint = out_shape[3] if out_shape is not None and len(out_shape) == 4 else None

            w_raw = _tflite_get_tensor(interpreter, int(ins[1]))
            if w_raw is None or w_raw.dtype != np.float32:
                continue

            Wn = _tflite_conv2d_to_noodle_oihw(w_raw, cin_hint=cin_hint, cout_hint=cout_hint)
            w_idx += 1
            O, I, Kh, Kw = Wn.shape
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=conv2d, layout=OIHW",
                    f"// tflite_op_index={op_i}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={I}, Cout={O}",
                    "// TFLite common: (Cout,Kh,Kw,Cin) -> Noodle: (Cout,Cin,Kh,Kw)",
                ],
            )

            b = _find_bias_input(interpreter, ins, expected_len=O, exclude_positions={0, 1})
            if b is not None:
                b_idx += 1
                _write_array_txt_and_h(out_dir, "b", b_idx, b)
            continue

        if op_name == "DEPTHWISE_CONV_2D":
            if len(ins) < 2:
                continue

            in_shape = _tensor_shape_from_details(tensor_details, ins[0])
            cin_hint = in_shape[3] if in_shape is not None and len(in_shape) == 4 else None

            w_raw = _tflite_get_tensor(interpreter, int(ins[1]))
            if w_raw is None or w_raw.dtype != np.float32:
                continue

            Wn = _tflite_dwconv2d_to_noodle_ckk(w_raw, cin_hint=cin_hint)
            w_idx += 1
            C, Kh, Kw = Wn.shape
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=depthwise2d, layout=CKK",
                    f"// tflite_op_index={op_i}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={C}, M=1, Cout={C}",
                ],
            )

            b = _find_bias_input(interpreter, ins, expected_len=C, exclude_positions={0, 1})
            if b is not None:
                b_idx += 1
                _write_array_txt_and_h(out_dir, "b", b_idx, b)
            continue

        if op_name == "FULLY_CONNECTED":
            if len(ins) < 2:
                continue

            w_raw = _tflite_get_tensor(interpreter, int(ins[1]))
            if w_raw is None or w_raw.dtype != np.float32:
                continue

            # Bias is usually input 2. Use it as output-size hint if present.
            b = _find_bias_input(interpreter, ins, exclude_positions={0, 1})
            dout_hint = int(b.shape[0]) if b is not None else None

            Wn = _tflite_dense_to_noodle_oi(w_raw, dout_hint=dout_hint)
            w_idx += 1
            O, I = Wn.shape
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=dense, layout=OI",
                    f"// tflite_op_index={op_i}",
                    f"// dims: Din={I}, Dout={O}",
                ],
            )

            if b is not None:
                b_idx += 1
                _write_array_txt_and_h(out_dir, "b", b_idx, b)
            continue

        if op_name == "TRANSPOSE_CONV":
            # TFLite TRANSPOSE_CONV commonly has inputs:
            #   [output_shape, weights, input_activation, bias?]
            # where output_shape is int32 and weights are float32 4D.
            if len(ins) < 3:
                continue

            # Find the 4D float weight tensor among inputs.
            w_pos, w_idx_tensor, w_raw = _find_first_float_tensor_input(interpreter, ins, ndim=4)
            if w_raw is None:
                continue

            # Input activation is the other 4D tensor, usually position 2.
            act_pos = None
            act_shape = None
            for pos, tidx in enumerate(ins):
                if pos == w_pos or int(tidx) < 0:
                    continue
                sh = _tensor_shape_from_details(tensor_details, tidx)
                if sh is not None and len(sh) == 4:
                    # Prefer tensor-details shape over get_tensor(), because activation
                    # tensors may not be readable constants.
                    act_pos = pos
                    act_shape = sh
                    break

            out_shape = _tensor_shape_from_details(tensor_details, outs[0]) if outs else None
            cin_hint = act_shape[3] if act_shape is not None and len(act_shape) == 4 else None
            cout_hint = out_shape[3] if out_shape is not None and len(out_shape) == 4 else None

            Wn = _tflite_transpose_conv_to_noodle_oihw(w_raw, cin_hint=cin_hint, cout_hint=cout_hint)
            w_idx += 1
            O, I, Kh, Kw = Wn.shape
            _write_array_txt_and_h(
                out_dir, "w", w_idx, Wn.flatten(order="C"),
                header_lines=[
                    "// kind=conv2d_transpose, layout=OIHW",
                    f"// tflite_op_index={op_i}",
                    f"// dims: Kh={Kh}, Kw={Kw}, Cin={I}, Cout={O}",
                    "// TFLite common: (Cout,Kh,Kw,Cin) -> Noodle: (Cout,Cin,Kh,Kw)",
                    "// spatial_flip=false",
                ],
            )

            b = _find_bias_input(interpreter, ins, expected_len=O, exclude_positions={w_pos})
            if b is not None:
                b_idx += 1
                _write_array_txt_and_h(out_dir, "b", b_idx, b)
            continue

    write_model_weights_header(out_dir, w_idx, b_idx)
    print(f"Export complete: w={w_idx}, b={b_idx}, out_dir={out_dir}")

def debug_tflite_ops(tflite_path):
    """Print readable parameter tensors for supported TFLite ops, including TRANSPOSE_CONV."""
    itp = tf.lite.Interpreter(model_path=tflite_path)
    itp.allocate_tensors()
    td = {int(d["index"]): d for d in itp.get_tensor_details()}
    ops = itp._get_ops_details()

    allowed = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV"}

    for i, op in enumerate(ops):
        op_name = op.get("op_name", "")
        if op_name not in allowed:
            continue

        ins = list(op.get("inputs", []))
        outs = list(op.get("outputs", []))
        print(i, op_name, "inputs:", ins, "outputs:", outs)

        for pos, idx in enumerate(ins):
            if int(idx) < 0:
                print("  in", pos, "idx", idx, "<none>")
                continue

            d = td.get(int(idx), {})
            try:
                t = itp.get_tensor(int(idx))
                t_info = (t.shape, t.dtype)
            except Exception:
                t_info = "<not readable activation/intermediate>"

            print("  in", pos,
                  "idx", int(idx),
                  "name", d.get("name", ""),
                  "shape", d.get("shape", None),
                  "dtype", d.get("dtype", None),
                  "tensor", t_info)

        for pos, idx in enumerate(outs):
            d = td.get(int(idx), {})
            print("  out", pos,
                  "idx", int(idx),
                  "name", d.get("name", ""),
                  "shape", d.get("shape", None),
                  "dtype", d.get("dtype", None))
        print()

def write_model_weights_header(out_dir: str, w_count: int, b_count: int):
    path = os.path.join(out_dir, "model_weights.h")
    with open(path, "w") as f:
        f.write("#pragma once\n\n")
        n = max(w_count, b_count)
        for i in range(1, n + 1):
            if i <= w_count:
                f.write(f'#include "w{i:02d}.h"\n')
            if i <= b_count:
                f.write(f'#include "b{i:02d}.h"\n')
    print(path)
                
if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a float TFLite model into Noodle-friendly .h and .txt files."
    )
    
    # Required arguments
    parser.add_argument(
        "tflite_path", 
        type=str, 
        help="Path to the input .tflite file"
    )
    parser.add_argument(
        "out_dir", 
        type=str, 
        help="Path to the directory where exported files will be saved"
    )
    
    # Optional flags
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Print verbose TFLite operator information before exporting"
    )

    args = parser.parse_args()

    # Run debug print if requested by the user
    if args.debug:
        print(f"=== Debugging operators for {args.tflite_path} ===")
        debug_tflite_ops(args.tflite_path)
        print("==================================================\n")

    # Execute the exporter
    print(f"Exporting {args.tflite_path} to directory '{args.out_dir}'...")
    exporter_tflite(args.tflite_path, args.out_dir)
    