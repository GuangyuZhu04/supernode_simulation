"""Finalize derived exports after the frozen scaling study has completed.

Numerical execution outputs and latency/capacity values are not changed.
The per-resource whole-window utilization is derived from exported work and
the completed TPOT; independent sensitivity files are joined without reruns.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/scaling-10case-20261113'))
    args=parser.parse_args()
    root=args.output.resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=manifest['provenance']['source_sha256']
    cases=[]
    for case in manifest['cases']:
        path=root/case['result_path']
        data=json.loads(path.read_text(encoding='utf-8'))
        if data['provenance']['source_sha256']!=expected:
            raise RuntimeError('Case does not match final frozen source: '+case['id'])
        cases.append((case,path,data))
    sensitivity=json.loads((root/'sensitivity.json').read_text(encoding='utf-8'))
    half=json.loads((root/'sensitivity-half/sensitivity.json').read_text(encoding='utf-8'))
    zero=[row for row in sensitivity if row['variant']=='zero_network_startup']
    if len(zero)!=len(cases) or len(half)!=8:
        raise RuntimeError(f'Sensitivity incomplete: zero={len(zero)}, half={len(half)}')
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows_changed=0
    for case,path,data in cases:
        csv_path=path.parent/'resources.csv'
        with csv_path.open('r',encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle)
            fields=reader.fieldnames
            rows=list(reader)
        duration=data['metrics']['tpot_s']
        for row in rows:
            capacity=float(row['capacity'])
            row['utilization']=str(float(row['work'])/(capacity*duration)) if duration else ''
        temporary=csv_path.with_suffix('.finalize.tmp.csv')
        with temporary.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
        rows_changed+=len(rows)
        data['artifact_export']['resource_utilization_scope']='Representative physical resource work divided by capacity and full-template TPOT; per-kind exact execution remains in compressed representative artifacts.'
        data['export_finalization']={'script':'scripts/finalize_scaling_study_outputs.py','script_sha256':script_sha,
                                     'numerical_execution_changed':False,
                                     'resource_utilization_formula':'work / (capacity * metrics.tpot_s)',
                                     'sensitivity_join':'20 zero-network-startup + 8 half-bandwidth hardware cases'}
        save(path,data)
        save(path.parent/'summary.json',data)
    merged={(row['case'],row['variant']):row for row in sensitivity}
    for row in half:
        row=dict(row)
        row['config_path']='sensitivity-half/'+row['config_path']
        merged[row['case'],row['variant']]=row
    combined=list(merged.values())
    save(root/'sensitivity.json',combined)
    with (root/'sensitivity.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['case','variant','status','tpot_s']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in combined)
    snapshot=root/'source-snapshot/scripts/finalize_scaling_study_outputs.py'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    audit={'cases':len(cases),'resource_rows_rederived':rows_changed,'sensitivity_rows':len(combined),
           'source_sha256_consistent':True,'finalizer_sha256':script_sha,'numerical_execution_changed':False}
    save(root/'export-finalization.json',audit)
    print(json.dumps(audit))


if __name__=='__main__':
    main()
