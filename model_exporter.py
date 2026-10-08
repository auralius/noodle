#!/usr/bin/env python3
"""TFLite-only Noodle parameter exporter; detects FP32 or full INT8 automatically.

Usage: python model_exporter.py model.tflite output_directory/
Parameter exporter only: reconstruct the operator graph in Noodle firmware.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Supported weighted op types have their parameter arrays extracted.
WEIGHTED_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV"}
# Listed in the manifest but NOT turned into firmware code by this tool.
METADATA_ONLY_OPS = {"MAX_POOL_2D", "AVERAGE_POOL_2D", "RESHAPE", "SOFTMAX", "RELU", "RELU6", "LOGISTIC", "TANH", "SQUEEZE"}


def _i8_quant(tensor_detail, *, per_channel=False, channels=None, axis=None):
    """Validate TFLite quantization metadata, returning scales and zero-points."""
    q = tensor_detail.get("quantization_parameters", {})
    scales = np.asarray(q.get("scales", []), dtype=np.float64).reshape(-1)
    zps = np.asarray(q.get("zero_points", []), dtype=np.int64).reshape(-1)
    if not len(scales) or len(zps) != len(scales):
        raise ValueError(f"{tensor_detail.get('name')}: missing quantization scales/zero-points")
    if not np.all(np.isfinite(scales)) or not np.all(scales > 0):
        raise ValueError(f"{tensor_detail.get('name')}: invalid quantization scales")
    if per_channel:
        if len(scales) not in (1, int(channels)):
            raise ValueError(f"{tensor_detail.get('name')}: expected 1 or {channels} weight scales, found {len(scales)}")
        if len(scales) > 1 and int(q.get("quantized_dimension", -1)) != int(axis):
            raise ValueError(f"{tensor_detail.get('name')}: quantized_dimension must be {axis} for Noodle output-channel order")
        if len(scales) == 1:
            scales = np.repeat(scales, int(channels))
            zps = np.repeat(zps, int(channels))
    elif len(scales) != 1:
        raise ValueError(f"{tensor_detail.get('name')}: activation must have per-tensor quantization")
    return scales, zps

def _i8_requantize(real_multiplier):
    """Bit-equivalent multiplier encoding to noodle_quantize_multiplier()."""
    import math
    x = float(real_multiplier)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Invalid requantization scale: {x}")
    if x == 0:
        return 0, 0
    q, exponent = math.frexp(x)
    fixed = int(math.floor(q * (1 << 31) + 0.5))  # C++ llround for nonnegative values
    if fixed == 1 << 31:
        fixed >>= 1
        exponent += 1
    if exponent < -31:
        return 0, 0
    if exponent > 30:
        raise ValueError(f"Requantization exponent {exponent} exceeds Noodle range")
    return fixed, exponent

def _write_i8_parameter(out_dir, prefix, idx, values, ctype, *, notes=()):
    """Emit raw LE .bin, decimal .txt, and C/C++ .h using current Noodle layout."""
    dtype = {'int8_t': np.dtype('i1'), 'int32_t': np.dtype('<i4')}[ctype]
    a = np.asarray(values, dtype=dtype).reshape(-1)
    name = f"{prefix}{idx:02d}"
    a.tofile(os.path.join(out_dir, f"{name}.bin"))
    np.savetxt(os.path.join(out_dir, f"{name}.txt"), a, fmt="%d")
    with open(os.path.join(out_dir, f"{name}.h"), "w", encoding="utf-8") as f:
        f.write('#pragma once\n#include <stdint.h>\n#include "noodle_export_storage.h"\n\n')
        for note in notes:
            f.write('// ' + str(note).replace('\n', ' ') + '\n')
        f.write(f'static const {ctype} {name}[] NOODLE_EXPORT_STORAGE = {{\n')
        ints = a.tolist()
        for start in range(0, len(ints), 12):
            f.write('  ' + ', '.join(str(int(x)) for x in ints[start:start+12]))
            f.write(',' if start+12 < len(ints) else '')
            f.write('\n')
        f.write('};\n')
    return {'bin': f'{name}.bin', 'txt': f'{name}.txt', 'header': f'{name}.h', 'symbol': name, 'elements': int(a.size)}

def _i8_tensor(interpreter, index, details, *, expected_dtype=None):
    index = int(index)
    if index < 0 or index not in details:
        raise ValueError(f'Missing TFLite tensor index {index}')
    detail = details[index]
    if expected_dtype is not None and detail['dtype'] != np.dtype(expected_dtype).type:
        raise ValueError(f"{detail.get('name', index)}: expected {expected_dtype}, got {detail['dtype']}")
    return detail, np.asarray(interpreter.get_tensor(index))

def _i8_activation_q(details, index):
    d = details[int(index)]
    if d['dtype'] != np.int8:
        raise ValueError(f"{d.get('name')}: INT8 export needs fully quantized int8 activations, got {d['dtype']}")
    s, zp = _i8_quant(d)
    if not -128 <= int(zp[0]) <= 127:
        raise ValueError(f"{d.get('name')}: invalid signed int8 zero point")
    return float(s[0]), int(zp[0])

def _export_tflite_int8_interpreter(interpreter, out_dir):
    """Export already-converted full-INT8 TFLite model. No host requantization of weights."""
    import json
    from pathlib import Path
    dest = Path(out_dir)
    # Never silently replace an existing FP32 export (identical w01/b01 filenames).
    if (dest / 'w01.h').exists() and not (dest / 'model_int8_manifest.json').exists():
        raise FileExistsError(f'{dest}: existing parameters found. Use a separate INT8 output directory.')
    dest.mkdir(parents=True, exist_ok=True)
    interpreter.allocate_tensors()
    details = {int(d['index']): d for d in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    supported = {'CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED', 'TRANSPOSE_CONV'}
    entries = []
    unsupported = []
    for op_i, op in enumerate(ops):
        kind = op.get('op_name', '')
        if kind not in supported:
            unsupported.append({'index': op_i, 'op': kind})
            continue
        ins = [int(x) for x in op.get('inputs', [])]
        outs = [int(x) for x in op.get('outputs', [])]
        if not outs:
            raise ValueError(f'{kind}: missing output tensor')
        if kind == 'TRANSPOSE_CONV':
            if len(ins) < 3:
                raise ValueError('TRANSPOSE_CONV: expected [output_shape, weights, input]')
            input_idx, weight_idx, bias_candidates = ins[2], ins[1], ins[3:]
        else:
            if len(ins) < 2:
                raise ValueError(f'{kind}: missing weight/input tensor')
            input_idx, weight_idx, bias_candidates = ins[0], ins[1], ins[2:]
        input_scale, input_zp = _i8_activation_q(details, input_idx)
        output_scale, output_zp = _i8_activation_q(details, outs[0])
        wd, wraw = _i8_tensor(interpreter, weight_idx, details, expected_dtype=np.int8)
        in_shape = tuple(map(int, details[input_idx]['shape']))
        out_shape = tuple(map(int, details[outs[0]]['shape']))
        if kind == 'CONV_2D':
            if len(in_shape) != 4 or len(out_shape) != 4 or wraw.ndim != 4:
                raise ValueError('CONV_2D: expected NHWC input/output and OHWI weights')
            channels = int(out_shape[3]); axis = 0
            if wraw.shape[0] != channels or wraw.shape[3] != in_shape[3]:
                raise ValueError('CONV_2D: expected TFLite OHWI weight layout')
            weights = np.transpose(wraw, (0, 3, 1, 2))  # OIHW
            if weights.shape[2] != weights.shape[3]:
                raise ValueError('CONV_2D: current Noodle kernel requires square filters')
            layout = 'OIHW'
        elif kind == 'DEPTHWISE_CONV_2D':
            if len(in_shape) != 4 or len(out_shape) != 4 or wraw.ndim != 4:
                raise ValueError('DEPTHWISE_CONV_2D: expected NHWC and 4D weights')
            channels = int(out_shape[3]); axis = 3
            if int(wraw.shape[0]) != 1 or int(wraw.shape[3]) != channels:
                raise ValueError('DEPTHWISE_CONV_2D: expected TFLite [1, Kh, Kw, Cout]')
            if channels % int(in_shape[3]):
                raise ValueError('DEPTHWISE_CONV_2D: invalid depth multiplier')
            mult = channels // int(in_shape[3])
            # TFLite depthwise weights channel order is [input channel, multiplier].
            weights = np.transpose(wraw[0], (2, 0, 1))  # CKK
            if weights.shape[1] != weights.shape[2]:
                raise ValueError('DEPTHWISE_CONV_2D: current Noodle kernel requires square filters')
            layout = 'CKK'
        elif kind == 'FULLY_CONNECTED':
            if wraw.ndim != 2:
                raise ValueError('FULLY_CONNECTED: expected [Dout, Din] weights')
            channels = int(wraw.shape[0]); axis = 0
            if int(out_shape[-1]) != channels:
                raise ValueError('FULLY_CONNECTED: output neurons do not match weights')
            weights = wraw  # OI
            layout = 'OI'
            if int(np.prod(in_shape[1:])) != int(wraw.shape[1]):
                raise ValueError('FULLY_CONNECTED: flattened input length disagrees with weight width')
        else:  # TRANSPOSE_CONV
            if len(in_shape) != 4 or len(out_shape) != 4 or wraw.ndim != 4:
                raise ValueError('TRANSPOSE_CONV: expected NHWC input/output and 4D weights')
            channels = int(out_shape[3]); axis = 0
            if int(wraw.shape[0]) != channels or int(wraw.shape[3]) != int(in_shape[3]):
                raise ValueError('TRANSPOSE_CONV: expected TFLite [Cout, Kh, Kw, Cin] weights')
            weights = np.transpose(wraw, (0, 3, 1, 2))
            if weights.shape[2] != weights.shape[3]:
                raise ValueError('TRANSPOSE_CONV: current Noodle kernel requires square filters')
            layout = 'OIHW'
        w_scales, w_zps = _i8_quant(wd, per_channel=True, channels=channels, axis=axis)
        if np.any(w_zps != 0):
            raise ValueError(f'{kind}: Noodle INT8 expects symmetric weights (zero point 0)')
        # Optional bias tensor: use fixed expected location, not heuristic activation search.
        bias = np.zeros(channels, dtype=np.int32)
        if bias_candidates and bias_candidates[0] >= 0:
            bd, braw = _i8_tensor(interpreter, bias_candidates[0], details, expected_dtype=np.int32)
            if braw.ndim != 1 or int(braw.size) != channels:
                raise ValueError(f'{kind}: bias shape must be [{channels}]')
            bs, bz = _i8_quant(bd, per_channel=True, channels=channels, axis=0)
            # Quantization parameters often use float32 precision; allow small relative error.
            if np.any(bz != 0) or not np.allclose(bs, input_scale * w_scales, rtol=2e-3, atol=0):
                raise ValueError(f'{kind}: bias quantization must use input_scale * weight_scale and zero point 0')
            bias = np.asarray(braw, dtype=np.int32)
        mul = np.empty(channels, dtype=np.int32)
        shift = np.empty(channels, dtype=np.int32)
        for ci, ws in enumerate(w_scales):
            mul[ci], shift[ci] = _i8_requantize(input_scale * float(ws) / output_scale)
        idx = len(entries) + 1
        notes = [f'op={kind}, index={op_i}, layout={layout}', f'shape={tuple(map(int,weights.shape))}']
        files = {
            'weight': _write_i8_parameter(dest, 'w', idx, weights, 'int8_t', notes=notes),
            'bias': _write_i8_parameter(dest, 'b', idx, bias, 'int32_t', notes=notes),
            'multiplier': _write_i8_parameter(dest, 'm', idx, mul, 'int32_t', notes=notes),
            'shift': _write_i8_parameter(dest, 's', idx, shift, 'int32_t', notes=notes),
        }
        layer = {
            'layer': idx, 'tflite_op_index': op_i, 'op': kind, 'layout': layout,
            'weight_shape': list(map(int, weights.shape)),
            'input_shape_nhwc': list(in_shape), 'output_shape_nhwc': list(out_shape),
            'input_scale': input_scale, 'input_zero_point': input_zp,
            'output_scale': output_scale, 'output_zero_point': output_zp,
            'weight_scales': w_scales.tolist(), 'weight_zero_points': w_zps.tolist(),
            'activation_min': -128, 'activation_max': 127,
            'activation_note': 'Default unclamped range; set fused RELU/RELU6 manually if present in TFLite op options.',
            'files': files,
        }
        if kind == 'DEPTHWISE_CONV_2D':
            layer['depth_multiplier'] = int(mult)
        entries.append(layer)
    if not entries:
        raise ValueError('No INT8 weighted ops found in model')
    with open(dest / 'noodle_export_storage.h', 'w', encoding='utf-8') as f:
        f.write('#pragma once\n')
        f.write('#if defined(__AVR__)\n#include <avr/pgmspace.h>\n#define NOODLE_EXPORT_STORAGE PROGMEM\n')
        f.write('#else\n#define NOODLE_EXPORT_STORAGE\n#endif\n')
    with open(dest / 'model_weights.h', 'w', encoding='utf-8') as f:
        f.write('#pragma once\n')
        for layer in entries:
            for v in ('weight', 'bias', 'multiplier', 'shift'):
                f.write('#include "' + layer['files'][v]['header'] + '"\n')
    # Layer-specific quantization scalar constants suitable for filling Noodle's Conv/FCN descriptor.
    with open(dest / 'model_quant.h', 'w', encoding='utf-8') as f:
        f.write('#pragma once\n#include <stdint.h>\nnamespace noodle_export_i8 {\n')
        for layer in entries:
            sym = f"layer{layer['layer']:02d}"
            for v in ('input_scale', 'output_scale'):
                f.write(f'static constexpr float {sym}_{v} = {layer[v]:.9g}' + ('' if 'e' in f'{layer[v]:.9g}' or '.' in f'{layer[v]:.9g}' else '.0') + 'f;\n')
            for v in ('input_zero_point', 'output_zero_point', 'activation_min', 'activation_max'):
                f.write(f'static constexpr int32_t {sym}_{v} = {layer[v]};\n')
        f.write('}\n')
    manifest = {'format': 'noodle_int8_parameters_v1', 'source': 'full_INT8_TFLite',
                'layer_count': len(entries), 'layers': entries,
                'non_weighted_ops': unsupported,
                'notes': ['File .bin = headerless little-endian int8/int32 (NOODLE_FILE_FORMAT_BIN).',
                          'Input and output activations are TFLite NHWC; Noodle convolution activations are CHW.',
                          'Weights use zero-point=0; per-output-channel multipliers/shifts match Noodle C++ algorithm.',
                          'Layer descriptors (stride, padding, fused activation) must be matched to model manually.',
                          'Unsupported or unexported non-weighted operations are listed but NOT converted.',
                          'Do not mix FP32 and INT8 exports in one directory.']}
    with open(dest / 'model_int8_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest

def _model_tensors(interpreter):
    """Inspect a TFLite interpreter (public tensor details + op order)."""
    interpreter.allocate_tensors()
    details = {int(d['index']): d for d in interpreter.get_tensor_details()}
    ops = list(interpreter._get_ops_details())  # TFLite Python op walk; private interpreter API
    return details, ops


def _detect_precision(interpreter):
    """Detect full FP32 vs *full* signed INT8. Refuse hybrid/mixed/unsupported graphs."""
    details, ops = _model_tensors(interpreter)
    if not ops:
        raise ValueError('TFLite file has no operations.')
    unsupported = sorted({op['op_name'] for op in ops
                          if op.get('op_name') not in WEIGHTED_OPS | METADATA_ONLY_OPS})
    if unsupported:
        raise ValueError('Unsupported TFLite operator(s): ' + ', '.join(unsupported)
                         + '. Parameter extraction cannot safely reproduce this graph.')

    weighted = [op for op in ops if op['op_name'] in WEIGHTED_OPS]
    if not weighted:
        raise ValueError('No supported weighted TFLite operators (Conv2D/DWConv2D/FCN/TransposeConv).')

    def dtype_at(idx):
        d = details.get(int(idx))
        if d is None:
            raise ValueError(f'Unknown TFLite tensor index {idx}.')
        return np.dtype(d['dtype'])

    kinds = set()
    for op in weighted:
        ins = list(map(int, op.get('inputs', [])))
        wi = 1  # TFLite CONV_2D / DEPTHWISE / FULLY_CONNECTED / TRANSPOSE_CONV
        if len(ins) <= wi or ins[wi] < 0:
            raise ValueError(f"{op['op_name']}: weights are missing.")
        kinds.add(dtype_at(ins[wi]))

    if len(kinds) != 1:
        raise ValueError('Hybrid/mixed TFLite weights detected: ' + ', '.join(map(str,kinds)))
    kind = next(iter(kinds))
    if kind == np.dtype('float32'):
        precision, activation_dtype = 'fp32', np.dtype('float32')
    elif kind == np.dtype('int8'):
        precision, activation_dtype = 'int8', np.dtype('int8')
    else:
        raise ValueError(f'Unsupported parameter dtype {kind}; expected float32 or signed int8.')

    # Reject FLOAT32 input / output around INT8 layers, and vice versa.
    # In particular do not silently accept a mixed-precision TFLite conversion.
    for direction, ds in [('input', interpreter.get_input_details()),
                          ('output', interpreter.get_output_details())]:
        for d in ds:
            if np.dtype(d['dtype']) != activation_dtype:
                raise ValueError(f"{precision.upper()} model has incompatible {direction} "
                                 f"{d.get('name', d['index'])}: {d['dtype']}. "
                                 'Convert to a fully FLOAT32 or fully signed INT8 TFLite model.')

    # Verify execution graph activations (not op constants) match the selected type.
    for idx, op in enumerate(ops):
        ins, outs = list(map(int, op.get('inputs', []))), list(map(int, op.get('outputs', [])))
        if op['op_name'] == 'TRANSPOSE_CONV':
            ai = ins[2] if len(ins) > 2 else -1
        else:
            ai = ins[0] if ins else -1
        activation_indices = ([ai] if ai >= 0 else []) + [o for o in outs if o >= 0]
        for ti in activation_indices:
            if dtype_at(ti) != activation_dtype:
                raise ValueError(f"{op['op_name']} (op #{idx}) has a mixed-precision activation "
                                 f"tensor {ti} ({dtype_at(ti)}).")

    return precision, details, ops


def _float_array(out_dir, prefix, index, values, *, notes=()):
    """Export native Noodle layout as .h, plain text and raw little-endian .bin."""
    a = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{prefix}{index:02d}: nonfinite FP32 values.')
    name = f'{prefix}{index:02d}'
    a.astype('<f4').tofile(Path(out_dir) / (name + '.bin'))
    np.savetxt(Path(out_dir) / (name + '.txt'), a, fmt='%.9g')
    with open(Path(out_dir)/(name+'.h'), 'w', encoding='utf-8') as f:
        f.write('#pragma once\n\n')
        for item in notes:
            f.write('// '+str(item).replace('\n',' ')+'\n')
        f.write(f'static const float {name}[] = {{\n')
        for k in range(0, len(a), 8):
            entries = []
            for v in a[k:k+8]:
                text = format(float(v), '.9g')
                if 'e' not in text and '.' not in text:
                    text += '.0'
                entries.append(text + 'f')
            f.write('  ' + ', '.join(entries) + (',' if k+8 < len(a) else '') + '\n')
        f.write('};\n')
    return {'bin': name+'.bin', 'txt': name+'.txt',
            'header': name+'.h', 'symbol': name, 'elements': int(a.size)}


def _export_tflite_float_interpreter(interpreter, out_dir):
    """FP32 extraction in TFLite execution order, preserving existing wNN/bNN names."""
    details, ops = _model_tensors(interpreter)
    out_dir = Path(out_dir)
    entries = []
    for op_i, op in enumerate(ops):
        kind = op['op_name']
        if kind not in WEIGHTED_OPS:
            continue
        ins = list(map(int, op.get('inputs', [])))
        outs = list(map(int, op.get('outputs', [])))
        if len(ins) < 2 or not outs:
            raise ValueError(f'{kind} op {op_i}: missing input/weight/output')
        ai = ins[2] if kind == 'TRANSPOSE_CONV' else ins[0]
        if ai < 0:
            raise ValueError(f'{kind} op {op_i}: missing activation input')
        in_shape = tuple(map(int, details[ai]['shape']))
        out_shape = tuple(map(int, details[outs[0]]['shape']))
        w_raw = np.asarray(interpreter.get_tensor(ins[1]))
        if w_raw.dtype != np.float32:
            raise ValueError(f'{kind} op {op_i}: weights must be FLOAT32')
        if kind == 'CONV_2D':
            if len(in_shape) != 4 or len(out_shape) != 4 or w_raw.ndim != 4:
                raise ValueError('CONV_2D requires NHWC input/output and OHWI weights')
            O = out_shape[3]
            if w_raw.shape[0] != O or w_raw.shape[3] != in_shape[3]:
                raise ValueError('CONV_2D: expected TFLite OHWI weight layout')
            weights, layout = w_raw.transpose(0, 3, 1, 2), 'OIHW'
        elif kind == 'DEPTHWISE_CONV_2D':
            if len(in_shape) != 4 or len(out_shape) != 4 or w_raw.ndim != 4:
                raise ValueError('DEPTHWISE_CONV_2D requires NHWC input/output and 4D weights')
            O = out_shape[3]
            if w_raw.shape[0] != 1 or w_raw.shape[3] != O:
                raise ValueError('DEPTHWISE_CONV_2D: expected TFLite [1,Kh,Kw,Cout]')
            if O != in_shape[3]:
                raise ValueError('DEPTHWISE_CONV_2D: FP32 Noodle export currently expects depth multiplier=1')
            weights, layout = w_raw[0].transpose(2, 0, 1), 'CKK'
        elif kind == 'FULLY_CONNECTED':
            if w_raw.ndim != 2 or w_raw.shape[0] != out_shape[-1]:
                raise ValueError('FULLY_CONNECTED: expected TFLite [Dout,Din]')
            if int(np.prod(in_shape[1:])) != w_raw.shape[1]:
                raise ValueError('FULLY_CONNECTED: input shape disagrees with weights')
            weights, layout, O = w_raw, 'OI', w_raw.shape[0]
        else:
            if len(in_shape) != 4 or len(out_shape) != 4 or w_raw.ndim != 4:
                raise ValueError('TRANSPOSE_CONV requires NHWC input/output and 4D weights')
            O = out_shape[3]
            if w_raw.shape[0] != O or w_raw.shape[3] != in_shape[3]:
                raise ValueError('TRANSPOSE_CONV: expected TFLite [Cout,Kh,Kw,Cin]')
            weights, layout = w_raw.transpose(0,3,1,2), 'OIHW'
        if kind != 'FULLY_CONNECTED' and weights.shape[-2] != weights.shape[-1]:
            raise ValueError(f'{kind}: current Noodle convolution kernels require square filters')
        bi = ins[3] if kind == 'TRANSPOSE_CONV' and len(ins) >= 4 else (
            ins[2] if kind != 'TRANSPOSE_CONV' and len(ins) >= 3 else -1)
        bias = np.zeros(O, dtype=np.float32)
        if bi >= 0:
            bias = np.asarray(interpreter.get_tensor(bi))
            if bias.dtype != np.float32 or bias.ndim != 1 or bias.size != O:
                raise ValueError(f'{kind}: expected FP32 bias of length {O}')
        seq = len(entries)+1
        notes = (f'op={kind}; tflite_index={op_i}; layout={layout}',f'shape={tuple(weights.shape)}')
        files = {'weight': _float_array(out_dir,'w',seq,weights,notes=notes),
                 'bias': _float_array(out_dir,'b',seq,bias,notes=notes)}
        entries.append({'layer':seq, 'tflite_op_index':op_i,'op':kind,'layout':layout,
                        'weight_shape':list(map(int,weights.shape)), 'input_shape_nhwc':list(in_shape),
                        'output_shape_nhwc':list(out_shape),'bias_synthesized':bi<0,'files':files})
    if not entries:
        raise ValueError('No weighted operations to export')
    with open(out_dir/'model_weights.h','w',encoding='utf-8') as f:
        f.write('#pragma once\n')
        for layer in entries:
            for role in ('weight','bias'):
                f.write(f'#include "{layer["files"][role]["header"]}"\n')
    manifest = {'format':'noodle_fp32_parameters_v1','precision':'fp32',
                'layer_count':len(entries),'layers':entries,
                'non_weighted_ops':[{'index':i,'op':op['op_name']}
                                    for i,op in enumerate(ops) if op['op_name'] not in WEIGHTED_OPS]}
    with open(out_dir/'model_fp32_manifest.json','w',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)
    return manifest


def export_interpreter(interpreter, out_dir):
    """Safe, auto-detected TFLite-only export; works with real or test interpreters."""
    precision, _, _ = _detect_precision(interpreter)
    dest = Path(out_dir)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f'{dest} already contains files. Choose an empty output directory; '
                              'the exporter will not overwrite existing FP32 or INT8 parameters.')
    parent = dest.resolve().parent
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.noodle_export_',dir=parent) as scratch:
        staging = Path(scratch)/'export'
        staging.mkdir()
        if precision == 'int8':
            manifest = _export_tflite_int8_interpreter(interpreter,str(staging))
            manifest['precision'] = 'int8'
        else:
            manifest = _export_tflite_float_interpreter(interpreter,str(staging))
        manifest['warning'] = ('Parameter extraction only. Nonweighted ops, convolution stride/padding, '
                               'fused activations and NHWC-to-CHW handling must be reproduced in Noodle firmware.')
        with open(staging/'model_manifest.json','w',encoding='utf-8') as f:
            json.dump(manifest,f,indent=2)
        if precision == 'int8':
            with open(staging/'model_int8_manifest.json','w',encoding='utf-8') as f:
                json.dump(manifest,f,indent=2)
        if dest.exists():
            dest.rmdir()  # existing *empty* directory
        shutil.move(str(staging),str(dest))
    return manifest


def export_tflite(tflite_path, out_dir):
    """One public entry point for both model precisions."""
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('Install TensorFlow to load .tflite models (pip install tensorflow).') from e
    # Prevent XNNPACK/default delegates from replacing original CONV ops with DELEGATE.
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path),
                                      experimental_preserve_all_tensors=True)
    return export_interpreter(interpreter,out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export FP32 or full INT8 TFLite parameters to Noodle .h/.txt/.bin files.')
    parser.add_argument('tflite_path', help='Input .tflite model')
    parser.add_argument('out_dir', help='New or empty output directory')
    args = parser.parse_args(argv)
    if not args.tflite_path.lower().endswith('.tflite'):
        parser.error('Input must be a .tflite model.')
    try:
        manifest = export_tflite(args.tflite_path,args.out_dir)
    except (ValueError,RuntimeError,FileExistsError,OSError,KeyError,IndexError) as e:
        parser.exit(1, f'ERROR: {e}\n')
    print(f"Noodle {manifest['precision'].upper()} export complete: {manifest['layer_count']} weighted layers -> {args.out_dir}")
    print('Use model_weights.h for on-chip parameters or raw .bin files for SD-backed parameters.')
    print('See model_manifest.json for layer shapes, quantization information and nonweighted operations.')


if __name__ == '__main__':
    main()
