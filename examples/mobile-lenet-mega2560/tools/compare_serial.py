#!/usr/bin/env python3
"""Compare Mega INT8 serial capture with TFLite first-100 reference CSV."""
import argparse, csv, re, statistics, pathlib, sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=pathlib.Path,required=True)
    p.add_argument('--log',type=pathlib.Path,required=True)
    p.add_argument('--strict-logits',action='store_true',help='Exit nonzero if any logit differs')
    a=p.parse_args()
    with a.reference.open(newline='') as f:
        reference={int(r['index']):r for r in csv.DictReader(f)}
    seen={}; errors=[]; logit_mismatch=0; total_logit_diff=0; pred_mismatch=0
    pending_logits=None; latencies=[]; free_ram=[]
    for text in a.log.read_text(errors='replace').splitlines():
        m=re.search(r'\bLOGITS((?:\s+-?\d+){10})(?!\d)',text)
        if m:
            pending_logits=[int(x) for x in m.group(1).split()]
        m=re.search(r'\bMEM\s+[^\n]*\bfree_ram=(-?\d+)',text)
        if m:free_ram.append(int(m.group(1)))
        m=re.search(r'\bidx=(\d+)\b.*\bpred=(\d+)\b.*\btime=([\d.]+)s',text)
        if not m:continue
        idx,pred,elapsed=int(m.group(1)),int(m.group(2)),float(m.group(3))
        if idx not in reference:errors.append(f'unexpected index {idx}'); continue
        if idx in seen:errors.append(f'duplicate index {idx}');continue
        ref=reference[idx]
        seen[idx]=(pred,pending_logits)
        latencies.append(elapsed)
        if pred!=int(ref['pred']):pred_mismatch+=1
        if pending_logits is None:
            errors.append(f'index {idx}: missing LOGITS')
        else:
            expected=[int(ref[f'logit{j}']) for j in range(10)]
            diffs=[got-exp for got,exp in zip(pending_logits,expected)]
            diff=sum(x!=0 for x in diffs);logit_mismatch+=diff
            total_logit_diff+=sum(abs(x) for x in diffs)
            if diff:print(f'index {idx:3d}: {diff}/10 different; Δ={diffs}')
        pending_logits=None
    print('Images measured:',len(seen),'/',len(reference))
    print('Predictions different from TFLite:',pred_mismatch)
    print('Raw INT8 logits different:',logit_mismatch,'/',len(seen)*10)
    print('Sum absolute logit differences:',total_logit_diff)
    if latencies:print('Measured latency (mean/min/max):',*(round(x*1000,2) for x in (statistics.mean(latencies),min(latencies),max(latencies))),'ms')
    if free_ram:print('Reported free RAM (min/max):',min(free_ram),max(free_ram),'B')
    for err in errors:print('ERROR:',err,file=sys.stderr)
    return 1 if errors or not seen or (a.strict_logits and logit_mismatch) else 0
if __name__=='__main__':sys.exit(main())
