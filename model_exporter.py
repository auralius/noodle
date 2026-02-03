import os
import numpy as np

def to_two_digit_string(n: int) -> str:
    return f"{n:02d}"

def format_c_array(array):
    """Format a 1D numpy array into a C-style array string."""
    lines = []
    line = []
    for i, val in enumerate(array):
        line.append(f"{val:.4e}f")
        if (i + 1) % 8 == 0:
            lines.append("  " + ", ".join(line))
            line = []
    if line:
        lines.append("  " + ", ".join(line))
    return ",\n".join(lines)

def _write_meta(fn_meta: str, kind: str, layout: str, dims: dict):
    with open(fn_meta, "w") as f:
        f.write(f"kind={kind}\n")
        f.write(f"layout={layout}\n")
        for k, v in dims.items():
            f.write(f"{k}={v}\n")

def exporter(weights, dir):
    """
    Export model weights in Noodle-friendly streaming format.

    Conventions:
    - 4D conv2d kernel (Keras): [Kh, Kw, Cin, Cout] -> OIHW stream (O-major, then I, then r,c)
      File order: for O, for I, for r, for c: write w[r,c,I,O]
    - 4D depthwise2d kernel (Keras): [Kh, Kw, Cin, M] -> CIMHW stream (C-major, then m, then r,c)
      File order: for C, for m, for r, for c: write w[r,c,C,m]

    - 3D conv1d kernel (Keras): [K, Cin, Cout] -> OIK stream (O-major, then I, then k)
      File order: for O, for I, for k: write w[k,I,O]
    - 3D depthwise1d-like kernel: [K, Cin, M] -> CIMK stream (C-major, then m, then k)
      File order: for C, for m, for k: write w[k,C,m]

    - 2D dense: transpose then flatten (same as your original)
    - 1D bias: one float per line

    Notes:
    - Conv vs depthwise detection is heuristic based on the NEXT 1D bias length:
        * if bias_len == Cout            => Conv
        * if bias_len == Cin*M           => Depthwise-like
      If unsure, defaults to Conv.
    """
    if not dir.endswith('/'):
        dir += '/'

    os.makedirs(dir, exist_ok=True)

    w_idx = 0
    b_idx = 0

    k = 0
    while k < len(weights):
        w = weights[k]

        # ---------- 4D tensors: Conv2D or DepthwiseConv2D ----------
        if len(w.shape) == 4:
            Kh, Kw, Cin, C4 = w.shape  # C4 is Cout for Conv2D, or M for Depthwise
            kind = "conv2d"
            layout = "OIHW"

            bias_len = None
            if k + 1 < len(weights) and len(weights[k + 1].shape) == 1:
                bias_len = int(weights[k + 1].shape[0])
                if bias_len == C4:
                    kind = "conv2d"
                    layout = "OIHW"
                elif bias_len == Cin * C4:
                    kind = "depthwise2d"
                    layout = "CIMHW"
                else:
                    kind = "conv2d"
                    layout = "OIHW"

            w_idx += 1
            fn_txt = dir + "w" + to_two_digit_string(w_idx) + ".txt"
            print(fn_txt)

            if kind == "conv2d":
                # [Kh, Kw, Cin, Cout] -> [Cout, Cin, Kh, Kw] then flatten (O,I,r,c)
                w_oihw = np.transpose(w, (3, 2, 0, 1)).astype(np.float32)
                flat = w_oihw.flatten(order="C")
                np.savetxt(fn_txt, flat, fmt="%.4e", newline="\n")
            else:
                # depthwise2d: [Kh, Kw, Cin, M] -> [Cin, M, Kh, Kw] then flatten (C,m,r,c)
                M = C4
                w_cmrw = np.transpose(w, (2, 3, 0, 1)).astype(np.float32)  # [Cin, M, Kh, Kw]
                flat = w_cmrw.flatten(order="C")
                np.savetxt(fn_txt, flat, fmt="%.4e", newline="\n")

            # Also generate .h
            fn_h = fn_txt.replace(".txt", ".h")
            var_name = "w" + to_two_digit_string(w_idx)
            c_array_content = format_c_array(flat)
            with open(fn_h, "w") as f_h:
                f_h.write("#pragma once\n\n")
                f_h.write(f"// kind={kind}, layout={layout}\n")
                if kind == "conv2d":
                    f_h.write(f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, Cout={C4}\n")
                else:
                    f_h.write(f"// dims: Kh={Kh}, Kw={Kw}, Cin={Cin}, M={C4}, Cout={Cin*C4}\n")
                f_h.write(f"static const float {var_name}[] = {{\n")
                f_h.write(c_array_content)
                f_h.write("\n};\n")
            print(fn_h)

            k += 1
            continue

        # ---------- 3D tensors: Conv1D or Depthwise1D-like ----------
        if len(w.shape) == 3:
            K, Cin, C3 = w.shape  # C3 is Cout for Conv1D, or M for depthwise-like
            kind = "conv1d"
            layout = "OIK"

            bias_len = None
            if k + 1 < len(weights) and len(weights[k + 1].shape) == 1:
                bias_len = int(weights[k + 1].shape[0])
                if bias_len == C3:
                    kind = "conv1d"
                    layout = "OIK"
                elif bias_len == Cin * C3:
                    kind = "depthwise1d"
                    layout = "CIMK"
                else:
                    kind = "conv1d"
                    layout = "OIK"

            w_idx += 1
            fn_txt = dir + "w" + to_two_digit_string(w_idx) + ".txt"
            print(fn_txt)

            if kind == "conv1d":
                # [K, Cin, Cout] -> [Cout, Cin, K] then flatten (O, I, k)
                w_oik = np.transpose(w, (2, 1, 0)).astype(np.float32)  # [Cout, Cin, K]
                flat = w_oik.flatten(order="C")
                np.savetxt(fn_txt, flat, fmt="%.4e", newline="\n")
            else:
                # depthwise1d-like: [K, Cin, M] -> [Cin, M, K] then flatten (C, m, k)
                M = C3
                w_cmk = np.transpose(w, (1, 2, 0)).astype(np.float32)  # [Cin, M, K]
                flat = w_cmk.flatten(order="C")
                np.savetxt(fn_txt, flat, fmt="%.4e", newline="\n")

            # Also generate .h
            fn_h = fn_txt.replace(".txt", ".h")
            var_name = "w" + to_two_digit_string(w_idx)
            c_array_content = format_c_array(flat)
            with open(fn_h, "w") as f_h:
                f_h.write("#pragma once\n\n")
                f_h.write(f"// kind={kind}, layout={layout}\n")
                if kind == "conv1d":
                    f_h.write(f"// dims: K={K}, Cin={Cin}, Cout={C3}\n")
                else:
                    f_h.write(f"// dims: K={K}, Cin={Cin}, M={C3}, Cout={Cin*C3}\n")
                f_h.write(f"static const float {var_name}[] = {{\n")
                f_h.write(c_array_content)
                f_h.write("\n};\n")
            print(fn_h)

            k += 1
            continue

        # ---------- 2D tensors: Dense ----------
        if len(w.shape) == 2:
            w_idx += 1
            fn_txt = dir + "w" + to_two_digit_string(w_idx) + ".txt"
            print(fn_txt)
            arr = np.float32(w.transpose().flatten())
            np.savetxt(fn_txt, arr, fmt="%.4e", newline="\n")

            fn_h = fn_txt.replace(".txt", ".h")
            var_name = "w" + to_two_digit_string(w_idx)
            c_array_content = format_c_array(arr)
            with open(fn_h, "w") as f_h:
                f_h.write("#pragma once\n\n")
                f_h.write(f"static const float {var_name}[] = {{\n")
                f_h.write(c_array_content)
                f_h.write("\n};\n")
            print(fn_h)

            k += 1
            continue

        # ---------- 1D tensors: Bias ----------
        if len(w.shape) == 1:
            b_idx += 1
            fn_txt = dir + "b" + to_two_digit_string(b_idx) + ".txt"
            print(fn_txt)
            arr = np.float32(w.flatten())
            np.savetxt(fn_txt, arr, fmt="%.4e", newline="\n")

            fn_h = fn_txt.replace(".txt", ".h")
            var_name = "b" + to_two_digit_string(b_idx)
            c_array_content = format_c_array(arr)
            with open(fn_h, "w") as f_h:
                f_h.write("#pragma once\n\n")
                f_h.write(f"static const float {var_name}[] = {{\n")
                f_h.write(c_array_content)
                f_h.write("\n};\n")
            print(fn_h)

            k += 1
            continue

        # ---------- Fallback ----------
        print("Skipping unsupported tensor with shape:", w.shape)
        k += 1
