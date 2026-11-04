"""Exercise prepare-independent train/checkpoint/evaluate wiring with tiny synthetic inputs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    data=args.output/'data'; data.mkdir()
    rng=np.random.default_rng(19)
    yy,xx=np.indices((48,64))
    frames=np.stack([26.+7.*np.exp(-((xx-22-i%4)**2+(yy-20)**2)/150)+rng.normal(0,.03,(48,64)) for i in range(24)]).astype(np.float32)
    np.save(data/'lepton_hr_c.npy',frames,allow_pickle=False)
    np.save(data/'lepton_eligible.npy',np.ones(24,bool),allow_pickle=False)
    rows=[]
    for i in range(24):
        split=['train','validation','test'][i//8]
        subject=['subject1','subject10','subject3'][i//8]
        rows.append(dict(cache_index=i,subject_id=subject,sequence_id=f'label0/{subject}/video0',view='front',frame_id=i,split=split))
    with (data/'lepton_index.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (data/'manifest.json').write_text(json.dumps({'status':'complete','synthetic':True,'scope':'wiring smoke test only'}),encoding='utf-8')
    config=json.loads((root/'configs/thermal_sr_x2.json').read_text())
    config.update(epochs=1,batch_size=4)
    config_file=args.output/'config.json'
    config_file.write_text(json.dumps(config),encoding='utf-8')
    run=args.output/'run'
    log=args.output/'commands.log'
    commands=[
        [sys.executable,'-B','-m','mlx90640_pipeline.training.train','--data',str(data),'--config',str(config_file),'--run',str(run),'--device','cpu'],
        [sys.executable,'-B','-m','mlx90640_pipeline.training.evaluate','--data',str(data),'--checkpoint',str(run/'best.pt'),'--output',str(args.output/'evaluation'),'--device','cpu'],
    ]
    with log.open('w',encoding='utf-8') as stream:
        for command in commands:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    report=json.loads((args.output/'evaluation/report.json').read_text())
    summary=dict(passed=True,synthetic=True,train_frames=8,validation_frames=8,test_frames=8,
                 epochs=1,selected_epoch=report['selected_epoch'],evaluated_conditions=len(report['test_conditions']),
                 scope='CLI training/backpropagation/checkpoint/evaluation smoke only; not model quality evidence')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
