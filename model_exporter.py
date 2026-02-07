import os
import numpy as np

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
