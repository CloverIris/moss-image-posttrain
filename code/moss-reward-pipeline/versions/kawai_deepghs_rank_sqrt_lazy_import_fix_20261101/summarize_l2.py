#!/usr/bin/env python3
"""Read genuine training records and summarize frozen-base L2 diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import statistics


FIELDS=('update','loss','art_loss','base_reference_beta','base_reference_mse',
        'base_reference_loss','base_reference_loss_to_art_ratio','base_reference_loss_fraction',
        'art_output_grad_norm','base_reference_output_grad_norm',
        'base_reference_output_grad_to_art_ratio','base_reference_output_grad_cosine',
        'base_reference_output_grad_cancellation_fraction','base_reference_output_grad_ratio_defined',
        'base_reference_output_grad_cosine_defined','art_zero_output_grad_fraction',
        'base_reference_zero_output_grad_fraction')


def summarize(rows):
    # A zero-initialized LoRA agrees with base; such rows cannot establish that
    # the coefficient is weak. Undefined ratios/cosines are excluded explicitly.
    active=[r for r in rows if r.get('base_reference_mse',0)>0]
    comparable=[r for r in active if r.get('base_reference_output_grad_ratio_defined',0)
                and r.get('base_reference_output_grad_cosine_defined',0)]
    report=dict(records=len(rows),nonzero_base_deviation_records=len(active),
                defined_competition_records=len(comparable),
                interpretation_scope='Predicted-velocity output cotangents, not LoRA parameter gradients.',
                coefficient_changed=False)
    if not comparable:
        report['assessment']='Insufficient nonzero/defined observations; initial zero LoRA is not evidence of a weak L2 coefficient.'
        return report
    ratios=[r['base_reference_output_grad_to_art_ratio'] for r in comparable]
    cosines=[r['base_reference_output_grad_cosine'] for r in comparable]
    competing=sum(a>1 and b<0 for a,b in zip(ratios,cosines))
    report.update(median_output_gradient_ratio=statistics.median(ratios),
                  min_output_gradient_ratio=min(ratios),max_output_gradient_ratio=max(ratios),
                  median_output_gradient_cosine=statistics.median(cosines),
                  opposed_fraction=sum(x<0 for x in cosines)/len(cosines),
                  stronger_and_opposed_fraction=competing/len(comparable))
    report['assessment']=('L2 is stronger and opposed in some collected output-space observations; inspect trend and compare beta under controlled conditions.'
                          if competing else 'No stronger-and-opposed L2 output cotangent observed in these collected records; this does not prove the coefficient is optimal.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    source=Path(cfg['input']);out=Path(cfg['output_dir'])
    rows=[]
    if source.exists():
        for line in source.read_text().splitlines():
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # Reader may observe a final partial live line.
            if 'base_reference_output_grad_norm' in r:rows.append(r)
    report=summarize(rows);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'updates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(report))


if __name__=='__main__':main()
