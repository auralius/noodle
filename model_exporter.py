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
    - 4D conv2d kernel (Keras): [Kh, Kw, Cin, Cout]  -> wXX.txt, layout OIHW (O-major, then I, then row/col)
      File order: for O, for I, for r, for c: write w[r,c,I,O] (one float per line)
    - 4D depthwise kernel (Keras): [Kh, Kw, Cin, M] -> wXX.txt, layout CIMHW (C-major, then multiplier, then row/col)
      File order: for C, for m, for r, for c: write w[r,c,C,m]
      (If M==1 this is simply C-major K*K per channel, which matches common DW streaming.)
    - 2D dense: transposed flatten same as original exporter
    - 1D bias: bXX.txt one float per line

    Notes:
    - Depthwise vs Conv2D detection is heuristic based on the NEXT 1D bias length:
        * if bias_len == Cout            => Conv2D
        * if bias_len == Cin*M           => Depthwise
      If unsure, defaults to Conv2D.
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
            # Lookahead for bias to infer layer type
            kind = "conv2d"
            layout = "OIHW"
            bias_len = None
            if k + 1 < len(weights) and len(weights[k + 1].shape) == 1:
                bias_len = int(weights[k + 1].shape[0])
                if bias_len == C4:
                    kind = "conv2d"
                    layout = "OIHW"
                elif bias_len == Cin * C4:
                    kind = "depthwise"
                    layout = "CIMHW"
                else:
                    # default conv2d
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

                #fn_meta = fn_txt.replace(".txt", ".meta.txt")
                #_write_meta(fn_meta, "conv2d", "OIHW", {"Kh": Kh, "Kw": Kw, "Cin": Cin, "Cout": C4})
                #print(fn_meta)

            else:
                # depthwise: [Kh, Kw, Cin, M] -> stream as C-major then multiplier then r,c
                M = C4
                w_cmrw = np.transpose(w, (2, 3, 0, 1)).astype(np.float32)  # [Cin, M, Kh, Kw]
                flat = w_cmrw.flatten(order="C")  # (C, m, r, c)
                np.savetxt(fn_txt, flat, fmt="%.4e", newline="\n")

                #fn_meta = fn_txt.replace(".txt", ".meta.txt")
                #_write_meta(fn_meta, "depthwise", "CIMHW", {"Kh": Kh, "Kw": Kw, "Cin": Cin, "M": M, "Cout": Cin * M})
                #print(fn_meta)

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
