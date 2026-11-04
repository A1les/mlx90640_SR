"""Compare independent complete caches, distinguishing text normalization from numeric changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a=json.loads((args.original/'manifest.json').read_text(encoding='utf-8'))
    b=json.loads((args.prepared/'manifest.json').read_text(encoding='utf-8'))
    keys=['native_frame_counts','quality_counts','numeric_temperature_range_c',
          'rejection_counts','eligible_counts','cross_split_exact_duplicate_groups']
    fields={k:a[k]==b[k] for k in keys}
    arrays={}
    for name in ['lepton_hr_c.npy','mlx_lr_c.npy','lepton_eligible.npy','mlx90640_eligible.npy']:
        x=np.load(args.original/name,mmap_mode='r',allow_pickle=False)
        y=np.load(args.prepared/name,mmap_mode='r',allow_pickle=False)
        same=x.shape==y.shape and x.dtype==y.dtype
        if same:
            same=all(x[i:i+2000].tobytes()==y[i:i+2000].tobytes() for i in range(0,len(x),2000))
        arrays[name]=dict(shape=list(x.shape),dtype=str(x.dtype),byte_identical=same)
    indices={}
    for name in ['lepton_index.csv','mlx90640_index.csv','subject_splits.csv']:
        x,y=args.original/name,args.prepared/name
        with x.open(encoding='utf-8-sig',newline='') as f:
            old=list(csv.DictReader(f))
        with y.open(encoding='utf-8-sig',newline='') as f:
            new=list(csv.DictReader(f))
        indices[name]=dict(byte_identical=x.read_bytes()==y.read_bytes(),records_identical=old==new)
    result=dict(all_relevant_frames_audited=b['all_relevant_native_frames_audited'],
                archive_sha256=b['archive_sha256'],manifest_fields_equal=fields,arrays=arrays,index_files=indices,
                text_normalization='Published split CSV uses UTF-8 without BOM and LF; CSV records must remain identical.',
                scope='Complete preprocessing regression; manifest creation times and source-path representation are intentionally different',
                all_passed=all(fields.values()) and all(v['byte_identical'] for v in arrays.values())
                and all(v['records_identical'] for v in indices.values()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
