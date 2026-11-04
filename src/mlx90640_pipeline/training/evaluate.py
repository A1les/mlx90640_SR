"""Frozen-checkpoint comparison on held-out people, with all images per condition."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .experiment import degradation, file_sha256, load_split, metrics_per_frame, summarize
from ..model import load_model

ROOT = Path(__file__).resolve().parent


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preview(path, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(samples), 5, figsize=(13.5, len(samples) * 2.5), constrained_layout=True, squeeze=False)
    for ax_row, sample in zip(axes, samples):
        values = [sample[k] for k in ['lr', 'target', 'bilinear', 'bicubic', 'model']]
        low, high = float(sample['target'].min()), float(sample['target'].max())
        for ax, arr, title in zip(ax_row, values, ['Synthetic LR 32x24', 'Lepton target 64x48', 'Bilinear', 'Bicubic', 'CNN x2']):
            im = ax.imshow(arr, cmap='inferno', vmin=low, vmax=high, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        ax_row[0].set_ylabel(sample['subject'], fontsize=9)
        fig.colorbar(im, ax=ax_row, shrink=0.8, label='degC')
    fig.suptitle('Held-out subjects: synthetic degradation, shared temperature scale per row. Not real MLX/HR pairs.', fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    output = args.output or args.checkpoint.parent / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise SystemExit('Evaluation exists. Use another --output for an explicitly repeated evaluation.')
    model, checkpoint = load_model(args.checkpoint, args.device)
    if file_sha256(args.data / 'manifest.json') != checkpoint['data_manifest_sha256']:
        raise ValueError('Data does not match training manifest')
    hr, metadata = load_split(args.data, 'test')
    config = checkpoint['config']
    bs = config['batch_size']
    records, samples = [], []
    # One preselected example per subject (first sorted frame), never chosen for score.
    example_indices = {}
    for i, row in enumerate(metadata):
        example_indices.setdefault(row['subject_id'], i)
    example_indices = set(example_indices.values())
    started = time.perf_counter()
    for condition_index, condition in enumerate(config['test_conditions']):
        generator = torch.Generator(device=args.device).manual_seed(config['test_seed'] + condition_index)
        for start in range(0, len(hr), bs):
            target = hr[start:start + bs].to(args.device)
            lr, _ = degradation(target, condition['blur_sigma'], condition['noise_std_c'], generator)
            outputs = {'bilinear': F.interpolate(lr, scale_factor=2, mode='bilinear', align_corners=False),
                       'bicubic': F.interpolate(lr, scale_factor=2, mode='bicubic', align_corners=False), 'model': model(lr)}
            for method, pred in outputs.items():
                scores = {k: v.cpu().numpy() for k, v in metrics_per_frame(pred, target).items()}
                for j in range(len(target)):
                    meta = metadata[start + j]
                    r = {k: meta[k] for k in ['cache_index', 'subject_id', 'sequence_id', 'view', 'frame_id']}
                    r.update(method=method, condition=condition['name'])
                    r.update({k: float(v[j]) for k, v in scores.items()})
                    records.append(r)
            if condition_index == 0:
                for j in range(len(target)):
                    if start + j in example_indices:
                        sample = {k: v[j, 0].cpu().numpy() for k, v in outputs.items()}
                        sample.update(lr=lr[j, 0].cpu().numpy(), target=target[j, 0].cpu().numpy(), subject=metadata[start+j]['subject_id'])
                        samples.append(sample)
        print(json.dumps({'condition': condition['name'], 'frames': len(hr), 'seconds': time.perf_counter() - started}), flush=True)
    grouped = summarize(records)
    save_csv(output / 'per_frame_metrics.csv', records)
    save_csv(output / 'grouped_metrics.csv', grouped)
    summary = [r for r in grouped if r['group_kind'] == 'all']
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint), selected_epoch=checkpoint['epoch'],
                  test_frames_per_condition=len(hr), test_subjects=sorted({r['subject_id'] for r in metadata}),
                  test_conditions=config['test_conditions'], results=summary,
                  metric_notes={'p95': 'Reported mean of per-frame pixel-error P95, not pooled global P95.',
                                'temperature': 'Error relative to the resampled Lepton temperature target, NOT absolute object temperature accuracy.',
                                'psnr_ssim': 'Fixed data_range=40 C; SSIM uniform 7x7 valid-window implementation.',
                                'weighting': 'Overall results are frame-weighted; subject/view/label results saved separately.'},
                  scope='Synthetic degradation only; does not establish real MLX, WIRIS or Raspberry Pi performance.',
                  seconds=time.perf_counter() - started)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    preview(output / 'test_comparison.png', samples)
    completion_path = args.checkpoint.parent / 'training_complete.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        completion['test_evaluation_completed'] = True
        completion['evaluation_report'] = str(output / 'report.json')
        completion_path.write_text(json.dumps(completion, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
