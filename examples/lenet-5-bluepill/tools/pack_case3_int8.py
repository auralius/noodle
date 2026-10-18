#!/usr/bin/env python3
"""Pack Case 3 FCN metadata like Case 1; convolution files remain split.

Uses only the Python standard library. For an existing signed INT8 split ZIP:
 python tools/pack_case3_int8.py --source case3_int8_split_parameters_NOT_FP32_firmware.zip --output sd_int8_packed

The 14 output .bin files are the *only* files to copy to the SD root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

CHANNELS = (6, 16, 120, 84, 10)
WEIGHT_BYTES = (150, 2400, 48000, 10080, 840)

def transform(source, out):
    source = Path(source)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; refusing to overwrite SD files')
    out.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise ValueError('Source must be a directory or ZIP')
        z = zipfile.ZipFile(source)
        def read(name):
            paths = [n for n in z.namelist() if n.rsplit('/',1)[-1] == name]
            if len(paths)!=1: raise ValueError(f'Expected one {name}, found {paths}')
            return z.read(paths[0])
    else:
        if not source.is_dir(): raise FileNotFoundError(source)
        def read(name): return (source/name).read_bytes()
    split_hashes = {}
    layout = {}
    for i,(ch,wbytes) in enumerate(zip(CHANNELS,WEIGHT_BYTES),1):
        idx=f'{i:02}'
        w=read(f'w{idx}.bin')
        if len(w)!=wbytes: raise AssertionError(f'w{idx} unexpected size {len(w)}')
        (out/f'w{idx}.bin').write_bytes(w)
        if i<=2:
            for prefix in 'bms':
                name=f'{prefix}{idx}.bin'
                v=read(name)
                if len(v)!=4*ch: raise AssertionError((name,len(v),4*ch))
                (out/name).write_bytes(v)
                split_hashes[name]=hashlib.sha256(v).hexdigest()
            layout[idx]='conv_split'
        else:
            arrays=[]
            for prefix in 'bms':
                name=f'{prefix}{idx}.bin'
                v=read(name)
                if len(v)!=4*ch: raise AssertionError((name,len(v),4*ch))
                split_hashes[name]=hashlib.sha256(v).hexdigest()
                arrays.append(struct.unpack('<'+'i'*ch,v))
            packed=b''.join(struct.pack('<iii',arrays[0][k],arrays[1][k],arrays[2][k]) for k in range(ch))
            assert len(packed)==12*ch
            unpacked=list(struct.iter_unpack('<iii',packed))
            assert all(unpacked[k]==tuple(a[k] for a in arrays) for k in range(ch))
            (out/f'p{idx}.bin').write_bytes(packed)
            layout[idx]='fcn_packed'
    bin_files=sorted(out.glob('*.bin'))
    expected={f'w{i:02}.bin' for i in range(1,6)} | {f'{p}{i:02}.bin' for i in range(1,3) for p in 'bms'} | {f'p{i:02}.bin' for i in range(3,6)}
    assert {p.name for p in bin_files}==expected
    assert sum(p.stat().st_size for p in bin_files)==64302
    manifest={'format':'case3_int8_conv_split_fcn_packed','layout':layout,
              'parameter_files':{p.name:{'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in bin_files},
              'split_metadata_sha256':split_hashes,
              'parameter_total_bytes':64302, 'parameter_file_count':14,
              'note':'Copy only *.bin to SD. Packed pNN stores LE int32 triples (bias,multiplier,shift) by output neuron.'}
    (out/'pack_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('OK: 14 files; 64,302 bytes. Packed FC metadata is exactly lossless.')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    transform(args.source,args.output)
