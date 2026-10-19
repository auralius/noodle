"""Generate Noodle layer-descriptor factories from validated TFLite export manifests.

This module does not generate or execute a model graph. Only weight-bearing
operators with verified builtin options receive descriptor factories.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


SUPPORTED = {'CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED', 'TRANSPOSE_CONV'}


def read_tflite_operator_options(tflite_path):
    """Extract operator options using the *actual* TFLite FlatBuffer schema.

    `_get_ops_details()` lacks builtin options; never infer them from shapes.
    Called by the normal CLI, where TensorFlow has already been loaded.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    buf = Path(tflite_path).read_bytes()
    model = schema.Model.GetRootAsModel(buf, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError('Layer factory export currently requires one TFLite subgraph')
    graph = model.Subgraphs(0)
    out = {}
    types = {
        'CONV_2D': schema.Conv2DOptions,
        'DEPTHWISE_CONV_2D': schema.DepthwiseConv2DOptions,
        'TRANSPOSE_CONV': schema.TransposeConvOptions,
        'FULLY_CONNECTED': schema.FullyConnectedOptions,
    }
    for index in range(graph.OperatorsLength()):
        op = graph.Operators(index)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        builtin_code = opcode.BuiltinCode()
        kind = next((name for name in types if builtin_code == getattr(schema.BuiltinOperator, name)), None)
        if kind is None:
            continue
        options = op.BuiltinOptions()
        if options is None:
            raise ValueError(f'{kind} op {index} missing builtin options')
        cls = types[kind]
        opts = cls()
        opts.Init(options.Bytes, options.Pos)
        cfg = {'op': kind}
        if kind != 'FULLY_CONNECTED':
            if opts.StrideH() != opts.StrideW():
                raise ValueError(f'{kind} op {index}: Noodle requires equal height/width stride')
            cfg['stride'] = int(opts.StrideH())
            pad = opts.Padding()
            if pad == schema.Padding.SAME:
                cfg['padding'] = 'SAME'
            elif pad == schema.Padding.VALID:
                cfg['padding'] = 'VALID'
            else:
                raise ValueError(f'{kind} op {index}: unknown padding option {pad}')
            if kind == 'DEPTHWISE_CONV_2D':
                cfg['depth_multiplier'] = int(opts.DepthMultiplier())
        if kind == 'TRANSPOSE_CONV':
            cfg['activation'] = 'NONE'  # builtin transpose convolution has no fused activation
        else:
            active = opts.FusedActivationFunction()
            lut = {
                schema.ActivationFunctionType.NONE: 'NONE',
                schema.ActivationFunctionType.RELU: 'RELU',
                schema.ActivationFunctionType.RELU6: 'RELU6',
                schema.ActivationFunctionType.RELU_N1_TO_1: 'RELU_N1_TO_1',
            }
            cfg['activation'] = lut.get(active, f'UNSUPPORTED_{active}')
        out[index] = cfg
    return out


def _quantize_bound(value, scale, zp):
    # TFLite uses rounded activation limits; ties go away from zero.
    r = value / scale
    q = math.floor(r + 0.5) if r >= 0 else math.ceil(r - 0.5)
    return max(-128, min(127, int(q) + int(zp)))


def _normalize_options(layer, config, precision):
    kind = layer['op']
    if config.get('op', kind) != kind:
        raise ValueError('TFLite operator type does not match exported parameter layer')
    activation = config.get('activation', '').upper()
    if activation not in ('NONE', 'RELU', 'RELU6', 'RELU_N1_TO_1'):
        raise ValueError('missing or unsupported verified fused activation option')
    if precision == 'fp32' and activation not in ('NONE', 'RELU'):
        raise ValueError(f'FP32 Noodle cannot express fused {activation} in its Activation enum')
    shape = layer['weight_shape']
    result = {'activation': activation, 'act': 'ACT_RELU' if activation in ('RELU', 'RELU6') else 'ACT_NONE'}
    if kind != 'FULLY_CONNECTED':
        stride = config.get('stride')
        padding = str(config.get('padding', '')).upper()
        if not isinstance(stride, int) or stride < 1 or stride > 65535:
            raise ValueError('missing/invalid TFLite stride')
        if padding not in ('SAME', 'VALID'):
            raise ValueError('missing/invalid TFLite padding')
        result['S'] = stride
        result['P'] = 65535 if padding == 'SAME' else 0
        result['K'] = shape[-1] if kind == 'DEPTHWISE_CONV_2D' else shape[-1]
        # OIHW/CKK last dimension is always the square kernel width.
        result['O'] = shape[0]
        if kind == 'DEPTHWISE_CONV_2D':
            mult = int(layer.get('depth_multiplier', 1))
            if 'depth_multiplier' in config and int(config['depth_multiplier']) != mult:
                raise ValueError('TFLite depth multiplier does not match exported tensor')
            result['depth_multiplier'] = mult
        if kind == 'TRANSPOSE_CONV':
            if padding == 'SAME':
                result['OP'] = 0
            else:
                # Valid transpose: expected V=(W-1)*S+K+OP.
                W = int(layer['input_shape_nhwc'][1]); V = int(layer['output_shape_nhwc'][1])
                op = V - ((W - 1) * stride + result['K'])
                if op < 0 or op >= stride:
                    raise ValueError(f'incompatible transpose-conv output padding {op}')
                result['OP'] = op
    else:
        result['O'] = shape[0]
    if precision == 'int8':
        zp = int(layer['output_zero_point']); scale = float(layer['output_scale'])
        if activation == 'NONE': lo, hi = -128, 127
        elif activation == 'RELU': lo, hi = max(-128, zp), 127
        elif activation == 'RELU6': lo, hi = max(-128, zp), _quantize_bound(6.0, scale, zp)
        else: lo, hi = _quantize_bound(-1.0, scale, zp), _quantize_bound(1.0, scale, zp)
        if lo > hi:
            raise ValueError('invalid quantized activation limits')
        result['activation_min'] = lo
        result['activation_max'] = hi
    return result


def _fp_literal(number):
    x = format(float(number), '.9g')
    if '.' not in x and 'e' not in x.lower(): x += '.0'
    return x + 'f'


def _ram_arrays(out_dir, layer, precision):
    idx = int(layer['layer']); suf = f'{idx:02d}'
    names = [('w', 'weight', 'int8_t' if precision == 'int8' else 'float', 'i1' if precision == 'int8' else '<f4'),
             ('bias', 'bias', 'int32_t' if precision == 'int8' else 'float', '<i4' if precision == 'int8' else '<f4')]
    if precision == 'int8': names += [('mult', 'multiplier', 'int32_t', '<i4'), ('shift', 'shift', 'int32_t', '<i4')]
    result = []
    packed = layer['files'].get('packed_metadata', {}).get('bin')
    packed_data = None
    if packed:
        packed_data = np.fromfile(out_dir / packed, dtype='<i4').reshape((-1, 3))
    for sym, role, ctype, dtype in names:
        if packed_data is not None and role in ('bias', 'multiplier', 'shift'):
            arr = packed_data[:, {'bias': 0, 'multiplier': 1, 'shift': 2}[role]]
        else:
            arr = np.fromfile(out_dir / layer['files'][role]['bin'], dtype=dtype)
        symbol = f'noodle_ram_{sym}{suf}'
        tokens = [(_fp_literal(v) if ctype == 'float' else str(int(v))) for v in arr]
        lines = [f'static const {ctype} {symbol}[{len(arr)}] = {{']
        step = 8 if ctype == 'float' else 16
        for offset in range(0, len(tokens), step):
            lines.append('  ' + ', '.join(tokens[offset:offset + step]) + ',')
        lines.append('};')
        result.extend(lines)
        result.append('')
    return result


def _field_lines(layer, config, precision, backend):
    kind = layer['op']; idx = int(layer['layer']); suf = f'{idx:02d}'
    file_names = layer['files']
    conv = kind != 'FULLY_CONNECTED'
    if conv:
        typ = {'file': 'Conv', 'pgm': 'ConvProgmem', 'ram': 'ConvMem'}[backend]
    else:
        typ = {'file': 'FCNFile', 'pgm': 'FCNProgmem', 'ram': 'FCNMem'}[backend]
    lines = [f'static inline {typ} noodle_layer{suf}_{backend}() {{', f'  {typ} l;']
    if backend == 'file':
        lines.append(f'  l.weight_fn = "{file_names["weight"]["bin"]}";')
        packed = file_names.get('packed_metadata') if not conv else None
        if packed:
            lines.append(f'  l.metadata_fn = "{packed["bin"]}";')
        else:
            lines.append(f'  l.bias_fn = "{file_names["bias"]["bin"]}";')
            if precision == 'int8':
                lines.append(f'  l.multiplier_fn = "{file_names["multiplier"]["bin"]}";')
                lines.append(f'  l.shift_fn = "{file_names["shift"]["bin"]}";')
    elif backend == 'ram':
        for field, sym in [('weight','w'), ('bias','bias')]+([('multiplier','mult'),('shift','shift')] if precision == 'int8' else []):
            lines.append(f'  l.{field} = noodle_ram_{sym}{suf};')
    elif conv:
        for field, sym in [('weight','w'), ('bias','bias')]+([('multiplier','mult'),('shift','shift')] if precision == 'int8' else []):
            lines.append(f'  l.{field} = noodle_{sym}{suf}_flash;')
    else:
        for field, sym in [('weight_far','w'), ('bias_far','bias')]+([('multiplier_far','mult'),('shift_far','shift')] if precision == 'int8' else []):
            lines.append(f'  l.{field} = NOODLE_PGM_ADDRESS(noodle_{sym}{suf}_flash);')
    lines.append(f'  l.O = {config["O"]};')
    if conv:
        for field in ('K','P','S'):
            lines.append(f'  l.{field} = {config[field]};')
        if kind == 'TRANSPOSE_CONV': lines.append(f'  l.OP = {config["OP"]};')
        if kind == 'DEPTHWISE_CONV_2D' and precision == 'int8':
            lines.append(f'  l.depth_multiplier = {config["depth_multiplier"]};')
    lines.append(f'  l.act = {config["act"]};')
    if precision == 'int8':
        stem = f'noodle_export_i8::layer{suf}'
        for field in ('input_scale','input_zero_point','output_scale','output_zero_point'):
            lines.append(f'  l.{field} = {stem}_{field};')
        lines.append(f'  l.activation_min = {stem}_activation_min;')
        lines.append(f'  l.activation_max = {stem}_activation_max;')
    lines += ['  return l;', '}', '']
    return lines


def generate_layer_headers(out_dir, layers, *, precision, operator_options):
    out_dir = Path(out_dir)
    results = {'headers': {}, 'generated': [], 'omitted': [], 'notes': [
        'Parameter backend does not fix activation backend: choose activation NoodleTensor/NoodleBuffer separately.',
        'PROGMEM FCN is far-addressed AVR; PROGMEM Conv/DWConv is near-addressed and requires all relevant arrays below 64KiB on large AVR.',
        'RAM parameter headers are intentionally optional and allocate full parameter arrays in SRAM on AVR; include in ONE translation unit only.',
        'TFLite separate nonweighted ops are NOT assembled (pooling, reshape, softmax, etc.).',
    ]}
    ready = []
    for layer in layers:
        idx = int(layer['layer']); kind = layer['op']
        try:
            config = _normalize_options(layer, operator_options.get(int(layer['tflite_op_index']), {}), precision)
            ready.append((layer, config))
            available = ['ram']
            if kind != 'TRANSPOSE_CONV' or precision == 'int8': available.append('file')
            if kind != 'TRANSPOSE_CONV': available.append('pgm')
            results['generated'].append({'layer': idx, 'tflite_op_index': layer['tflite_op_index'], 'op': kind, 'options': config, 'backends': available})
            if precision == 'int8':
                layer['activation_min'], layer['activation_max'] = config['activation_min'], config['activation_max']
                layer['activation_note'] = 'Derived from verified TFLite fused activation options.'
        except (ValueError, TypeError, KeyError) as exc:
            results['omitted'].append({'layer': idx, 'op': kind, 'reason': str(exc)})
    for backend in ('file','pgm','ram'):
        name = f'model_layers_{backend}.h'
        lines = [f'// GENERATED by Noodle model_exporter.py. {precision.upper()} descriptor factories.',
                 '// Parameter descriptors only; no execution graph or activation buffers.',
                 '// Do not edit. Re-export to regenerate.', '#pragma once', '#include "noodle.h"',
                 '#include <stdint.h>']
        if precision == 'int8':
            lines += ['#if !defined(NOODLE_USE_INT8)', '#error "Use -D NOODLE_USE_INT8 with these generated layer factories"', '#endif', '#include "model_quant.h"']
        else:
            lines += ['#if defined(NOODLE_USE_INT8)', '#error "FP32 layer factories require a FP32 Noodle build"', '#endif']
        if backend == 'file':
            lines += ['#if NOODLE_FILE_FORMAT != NOODLE_FILE_FORMAT_BIN',
                      '#error "Exported .bin parameters require NOODLE_FILE_FORMAT_BIN"', '#endif']
        if backend == 'pgm':
            lines += ['#if !defined(__AVR__)', '#error "Generated Noodle PROGMEM factories require an AVR target"', '#endif', '#include "model_progmem.h"',
                     '// ConvProgmem uses near reads: on Mega2560, verify placements <64KiB in the linker map.']
        if backend == 'ram':
            lines += ['// RAM arrays below are not PROGMEM: this header consumes SRAM when compiled.',
                      '// Include this header only from ONE translation unit.']
            for layer, cfg in ready: lines += _ram_arrays(out_dir, layer, precision)
        lines += ['']
        for layer, cfg in ready:
            if backend == 'pgm' and layer['op'] == 'TRANSPOSE_CONV':
                continue
            if backend == 'file' and precision == 'fp32' and layer['op'] == 'TRANSPOSE_CONV':
                continue
            # FP32 TransposeConv and INT8 TransposeConv exist for RAM; INT8 also file.
            lines += [f'// Weighted layer {layer["layer"]:02d}: {layer["op"]} ({cfg["activation"]})']
            lines += _field_lines(layer, cfg, precision, backend)
        (out_dir / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        results['headers'][backend] = name
    return results
