#!/usr/bin/env python3
"""Compare instrumented Mega FP32 GAP, logits, probabilities with Keras CSV."""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def compare():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hardware',type=Path,default=Path('case2_fp32_numeric_probe.json'))
    ap.add_argument('--keras',type=Path,default=Path('case2_fp32_keras_probe_reference.csv'))
    ap.add_argument('--out',type=Path,default=Path('case2_fp32_probe_comparison.csv'))
    a=ap.parse_args()
    hw=json.loads(a.hardware.read_text())['samples']
    ref=pd.read_csv(a.keras).set_index('idx')
    output=[]
    for d in hw:
        idx=int(d['index'])
        if idx not in ref.index: raise ValueError(f'Keras reference missing index {idx}')
        row=ref.loc[idx]
        if str(row['pixel_crc32']).upper() != str(d['pixel_crc32']).upper():
            raise ValueError(f'Pixel CRC differs for {idx}: hardware {d["pixel_crc32"]} vs Keras {row["pixel_crc32"]}')
        v={name:np.asarray(d['FP32_'+name],dtype=float) for name in ('GAP','LOGITS','PROBS')}
        k={'GAP':np.asarray([row[f'gap{i}'] for i in range(24)],dtype=float),
           'LOGITS':np.asarray([row[f'logit{i}'] for i in range(10)],dtype=float),
           'PROBS':np.asarray([row[f'prob{i}'] for i in range(10)],dtype=float)}
        rec={'index':idx,'gt':int(d['gt']),'mega_pred':int(d['pred']), 'keras_pred':int(row['keras_pred']),
             'mega_confidence':float(d['confidence']), 'keras_winning_probability':float(max(k['PROBS'])),
             'softmax_internal_max_abs':float(d['host_softmax_vs_mcu_probs_max_abs'])}
        for name in v:
            diff = v[name]-k[name]
            rec[name.lower()+'_max_abs']=float(np.max(np.abs(diff)))
            rec[name.lower()+'_mae']=float(np.mean(np.abs(diff)))
            rec[name.lower()+'_worst_index']=int(np.argmax(np.abs(diff)))
        # Shift-invariant logit differences isolate numerical class-boundary changes.
        vrel=v['LOGITS'] - v['LOGITS'].mean()
        krel=k['LOGITS'] - k['LOGITS'].mean()
        rec['centered_logit_max_abs']=float(np.max(np.abs(vrel-krel)))
        output.append(rec)
        print(f"index {idx} (gt={rec['gt']}): Mega={rec['mega_pred']} Keras={rec['keras_pred']}; "
              f"GAP err={rec['gap_max_abs']:.7g} LOGITS err={rec['logits_max_abs']:.7g} "
              f"PROBS err={rec['probs_max_abs']:.7g} "
              f"MCU softmax self-check err={rec['softmax_internal_max_abs']:.7g}")
        print('  Mega top 3:',np.argsort(v['PROBS'])[-3:][::-1].tolist(),
              'Keras top 3:',np.argsort(k['PROBS'])[-3:][::-1].tolist())
    pd.DataFrame(output).to_csv(a.out,index=False)
    print('Saved:',a.out)

if __name__=='__main__': compare()
