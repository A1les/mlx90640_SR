"""Shared synthetic degradation and evaluation definitions, fixed before training."""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(data_dir, split):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Data preparation is incomplete')
    eligible = np.load(data_dir / 'lepton_eligible.npy', allow_pickle=False)
    with (data_dir / 'lepton_index.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split and eligible[int(r['cache_index'])]]
    # Stable order also defines deterministic validation/test noise generation.
    rows.sort(key=lambda r: (r['subject_id'], r['sequence_id'], r['view'], int(r['frame_id'])))
    indices = [int(r['cache_index']) for r in rows]
    source = np.load(data_dir / 'lepton_hr_c.npy', mmap_mode='r', allow_pickle=False)
    return torch.from_numpy(np.array(source[indices]))[:, None], rows


def blur_downsample(hr, sigma):
    if sigma > 0:
        coordinates = torch.arange(-2, 3, dtype=hr.dtype, device=hr.device)
        kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        kernel2d = torch.outer(kernel, kernel).reshape(1, 1, 5, 5)
        hr = F.conv2d(F.pad(hr, (2, 2, 2, 2), mode='reflect'), kernel2d)
    return F.avg_pool2d(hr, 2)


def degradation(hr, sigma, noise_std_c, generator):
    clean_lr = blur_downsample(hr, sigma)
    noise = torch.randn(clean_lr.shape, dtype=clean_lr.dtype, device=clean_lr.device, generator=generator)
    noisy_lr = clean_lr + noise * noise_std_c
    # 0.1 C is this public MLX dataset's storage step, not a sensor accuracy claim.
    return torch.round(noisy_lr * 10) / 10, clean_lr


def edge_mae_per_frame(pred, target):
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return (dx.abs().flatten(1).mean(1) + dy.abs().flatten(1).mean(1)) / 2


def metrics_per_frame(pred, target):
    diff = pred - target
    errors = diff.abs().flatten(1)
    mse = diff.square().flatten(1).mean(1)
    # Fixed data_range=40 C for every frame and every method; no framewise rescale.
    kernel = torch.ones((1, 1, 7, 7), device=pred.device, dtype=pred.dtype) / 49
    mu_x, mu_y = F.conv2d(pred, kernel), F.conv2d(target, kernel)
    var_x = (F.conv2d(pred.square(), kernel) - mu_x.square()).clamp_min(0)
    var_y = (F.conv2d(target.square(), kernel) - mu_y.square()).clamp_min(0)
    cov = F.conv2d(pred * target, kernel) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.4 ** 2) * (2 * cov + 1.2 ** 2)) / ((mu_x.square() + mu_y.square() + 0.4 ** 2) * (var_x + var_y + 1.2 ** 2))
    return {
        'mae_c': errors.mean(1), 'rmse_c': torch.sqrt(mse), 'bias_c': diff.flatten(1).mean(1),
        'p95_pixel_absolute_error_c': torch.quantile(errors, 0.95, dim=1),
        'edge_mae_c_per_pixel': edge_mae_per_frame(pred, target),
        'max_temperature_absolute_error_c': (pred.flatten(1).max(1).values - target.flatten(1).max(1).values).abs(),
        'psnr_db_range40c': 10 * torch.log10(1600 / mse.clamp_min(1e-12)),
        'ssim_range40c_box7': ssim.flatten(1).mean(1),
    }


def summarize(records):
    grouped = defaultdict(list)
    metric_names = [
        'mae_c', 'rmse_c', 'bias_c', 'p95_pixel_absolute_error_c', 'edge_mae_c_per_pixel',
        'max_temperature_absolute_error_c', 'psnr_db_range40c', 'ssim_range40c_box7']
    for r in records:
        for kind, value in [('all', 'all'), ('subject', r['subject_id']), ('view', r['view']), ('label', r['sequence_id'].split('/')[0])]:
            grouped[(r['method'], r['condition'], kind, value)].append(r)
    rows = []
    for (method, condition, kind, value), group in sorted(grouped.items()):
        row = dict(method=method, condition=condition, group_kind=kind, group_value=value, frames=len(group))
        row.update({f'mean_{name}': float(np.mean([r[name] for r in group])) for name in metric_names})
        rows.append(row)
    return rows
