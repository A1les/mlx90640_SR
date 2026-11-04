"""Train on ALL eligible training frames; select checkpoints using held-out subjects."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import torch

from .experiment import blur_downsample, degradation, edge_mae_per_frame, file_sha256, load_split
from ..model import ThermalSRx2

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def validation_mae(model, hr, config, device, batch_size):
    generator = torch.Generator(device=device).manual_seed(config['seed'])
    total = 0.0
    model.eval()
    for start in range(0, len(hr), batch_size):
        target = hr[start:start + batch_size].to(device)
        lr, _ = degradation(target, config['blur_sigma'], config['noise_std_c'], generator)
        total += float((model(lr) - target).abs().sum())
    return total / hr.numel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'last.pt').exists() and not args.resume:
        raise SystemExit('Existing run; use --resume or a new --run directory.')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = random.Random(config['seed'])
    shuffle_rng = torch.Generator().manual_seed(config['seed'])
    noise_rng = torch.Generator(device=args.device).manual_seed(config['seed'])
    train_hr, train_rows = load_split(args.data, 'train')
    val_hr, val_rows = load_split(args.data, 'validation')
    assert set(r['subject_id'] for r in train_rows).isdisjoint(r['subject_id'] for r in val_rows)
    model = ThermalSRx2(**config['model']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] * 0.05)
    best = float('inf')
    first_epoch = 1
    if args.resume:
        checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['data_manifest_sha256'] != file_sha256(args.data / 'manifest.json'):
            raise ValueError('Cannot resume with different configuration/data')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        first_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best_validation_mae_c']
        rng.setstate(checkpoint['python_rng_state'])
        shuffle_rng.set_state(checkpoint['shuffle_rng_state'])
        noise_rng.set_state(checkpoint['noise_rng_state'])
    metadata = dict(config=config, train_frames=len(train_hr), validation_frames=len(val_hr),
                    parameters=sum(p.numel() for p in model.parameters()), device=args.device,
                    torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                    data_manifest_sha256=file_sha256(args.data / 'manifest.json'),
                    source_hashes={p: file_sha256(ROOT.parent / p) for p in ['model.py', 'training/train.py', 'training/experiment.py']},
                    created_at_utc=datetime.now(timezone.utc).isoformat(), test_used_for_training=False)
    (run / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)
    started = time.perf_counter()
    bs = config['batch_size']
    for epoch in range(first_epoch, config['epochs'] + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_hr), generator=shuffle_rng)
        totals = {'loss': 0., 'temperature_c': 0., 'edge_c': 0., 'consistency_c': 0.}
        seen = 0
        for step, start in enumerate(range(0, len(order), bs), 1):
            target = train_hr[order[start:start + bs]].to(args.device)
            if rng.random() < 0.5:
                target = target.flip(-1)
            sigma = rng.choice(config['blur_sigma_choices_hr_pixels'])
            noise_std = rng.uniform(*config['noise_std_uniform_c'])
            lr, clean_lr = degradation(target, sigma, noise_std, noise_rng)
            pred = model(lr)
            l_temp = (pred - target).abs().mean()
            l_edge = edge_mae_per_frame(pred, target).mean()
            # Use the SAME known blur+area operator as generation. Supervise
            # against its clean measurement to avoid forcing noise into the SR.
            l_cons = (blur_downsample(pred, sigma) - clean_lr).abs().mean()
            weights = config['loss_weights']
            loss = weights['temperature'] * l_temp + weights['edge'] * l_edge + weights['consistency'] * l_cons
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(target)
            seen += n
            for key, value in zip(totals, [loss, l_temp, l_edge, l_cons]):
                totals[key] += float(value.detach()) * n
            if step % 150 == 0:
                print(json.dumps({'epoch': epoch, 'frames_seen': seen, 'train_frames': len(order)}), flush=True)
        assert seen == len(train_hr)
        val_mae = validation_mae(model, val_hr, config['validation'], args.device, bs)
        scheduler.step()
        improved = val_mae < best
        best = min(best, val_mae)
        record = dict(epoch=epoch, frames_seen=seen, **{k: v / seen for k, v in totals.items()},
                      validation_mae_c=val_mae, best_validation_mae_c=best,
                      seconds=time.perf_counter() - epoch_started)
        with (run / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        checkpoint = dict(model_state=model.state_dict(), model_config=config['model'], config=config,
                          epoch=epoch, validation_mae_c=val_mae, best_validation_mae_c=best,
                          data_manifest_sha256=metadata['data_manifest_sha256'])
        if improved:
            torch.save(checkpoint, run / 'best.pt')
        checkpoint.update(optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                          python_rng_state=rng.getstate(), shuffle_rng_state=shuffle_rng.get_state(),
                          noise_rng_state=noise_rng.get_state())
        torch.save(checkpoint, run / 'last.pt')
        print(json.dumps(record), flush=True)
    (run / 'training_complete.json').write_text(json.dumps({
        'epochs': config['epochs'], 'train_frames_each_epoch': len(train_hr),
        'best_validation_mae_c': best, 'this_invocation_seconds': time.perf_counter() - started,
        'test_evaluation_completed': False, 'real_mlx_accuracy_verified': False,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
