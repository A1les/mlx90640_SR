"""Audit every relevant native TIFF and cache temperature arrays without extracting ZIPs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

PATTERN = re.compile(r'MlGesturePublic/train_val/frame_data/(label\d+/(subject\d+)/video\d+)/(mlx90640|lepton)_(front|top)/frame_(\d+)\.tiff$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--expected-sha256', default='36020b89e29312ec3fed46b5ea85b9a407a538ba3f8387fea8f4dd153c0c93d8')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output must be new or empty; existing caches are never overwritten.')
    archive = args.archive
    archive_sha = sha256(archive)
    if archive_sha != args.expected_sha256:
        raise RuntimeError('Archive SHA256 differs from the declared input.')
    split_file = args.splits
    out.mkdir(parents=True, exist_ok=True)
    with split_file.open(encoding='utf-8-sig', newline='') as f:
        split_rows = list(csv.DictReader(f))
    splits = {r['subject_id']: r['proposed_split'] for r in split_rows}
    if (len(splits) != len(split_rows) or set(splits) != {f'subject{i}' for i in range(1,25)}
            or set(splits.values()) != {'train', 'validation', 'test'}):
        raise ValueError('Require unique assignments for all 24 participants')
    (out / 'subject_splits.csv').write_bytes(split_file.read_bytes())
    cv2.setNumThreads(1)
    counters = defaultdict(Counter)
    reasons = Counter()
    raw_hash_groups = defaultdict(list)
    stats = {s: {'minimum_c': float('inf'), 'maximum_c': -float('inf')} for s in ['lepton', 'mlx90640']}
    started = time.perf_counter()
    with zipfile.ZipFile(archive) as z:
        entries = [(info, PATTERN.fullmatch(info.filename)) for info in z.infolist()]
        entries = [(info, m) for info, m in entries if m is not None]
        counts = Counter(m[3] for _, m in entries)
        if counts != {'mlx90640': 139348, 'lepton': 79895}:
            raise ValueError(f'Unexpected source inventory: {counts}')
        caches = {
            'lepton': np.lib.format.open_memmap(out / 'lepton_hr_c.npy', mode='w+', dtype=np.float32, shape=(counts['lepton'], 48, 64)),
            'mlx90640': np.lib.format.open_memmap(out / 'mlx_lr_c.npy', mode='w+', dtype=np.float32, shape=(counts['mlx90640'], 24, 32)),
        }
        valid = {s: np.zeros(n, dtype=bool) for s, n in counts.items()}
        times = {}
        indices = Counter()
        fieldnames = ['sensor', 'cache_index', 'sequence_id', 'subject_id', 'split', 'view', 'frame_id',
                      'timestamp_s', 'source_member', 'valid_for_numeric_use', 'reason', 'minimum_c', 'maximum_c',
                      'mean_c', 'std_c', 'outside_declared_range_pixels', 'declared_endpoint_pixels',
                      'isolated_jump_gt10c_pixels', 'native_temperature_sha256']
        handles = {s: (out / f'{s}_index.csv').open('w', encoding='utf-8', newline='') for s in counts}
        writers = {s: csv.DictWriter(f, fieldnames=fieldnames) for s, f in handles.items()}
        for w in writers.values():
            w.writeheader()
        try:
            for processed, (info, m) in enumerate(entries, 1):
                sequence, subject, sensor, view, frame_s = m.groups()
                frame = int(frame_s)
                row = indices[sensor]
                indices[sensor] += 1
                stream = info.filename.rsplit('/', 1)[0]
                if stream not in times:
                    times[stream] = [float(t) for t in z.read(stream + '/timestamp.txt').decode('utf-8-sig').split()]
                record = dict(sensor=sensor, cache_index=row, sequence_id=sequence, subject_id=subject,
                              split=splits[subject], view=view, frame_id=frame, timestamp_s=times[stream][frame], source_member=info.filename)
                counters[sensor][f'total_{splits[subject]}'] += 1
                try:
                    # PIL sometimes returns uint16 when TIFF SampleFormat is absent.
                    # The publisher explicitly specifies signed int16 for both sensors.
                    with Image.open(io.BytesIO(z.read(info))) as im:
                        raw = np.array(im)
                    expected = (120, 160) if sensor == 'lepton' else (24, 32)
                    if raw.shape != expected or raw.dtype.kind not in 'iu' or raw.dtype.itemsize != 2:
                        raise ValueError(f'Unexpected shape/dtype: {raw.shape}/{raw.dtype}')
                    signed = raw.astype(np.int16)
                    temp = signed.astype(np.float32) / (100.0 if sensor == 'lepton' else 10.0)
                    lo, hi = float(temp.min()), float(temp.max())
                    stats[sensor]['minimum_c'] = min(lo, stats[sensor]['minimum_c'])
                    stats[sensor]['maximum_c'] = max(hi, stats[sensor]['maximum_c'])
                    declared_lo, declared_hi = (25.0, 45.0) if sensor == 'lepton' else (-40.0, 300.0)
                    outside = int(((temp < declared_lo) | (temp > declared_hi)).sum())
                    endpoints = int(((temp == declared_lo) | (temp == declared_hi)).sum())
                    isolated = int((np.abs(temp - cv2.medianBlur(temp, 3)) > 10).sum())
                    # Broad corruption check only; README's Lepton range is also logged,
                    # but is not assumed to be its actual operating/calibration range.
                    ok = bool(np.isfinite(temp).all() and lo >= -40 and hi <= 300)
                    reason = '' if ok else 'outside_broad_numeric_range'
                    caches[sensor][row] = cv2.resize(temp, (64, 48), interpolation=cv2.INTER_AREA) if sensor == 'lepton' else temp
                    valid[sensor][row] = ok
                    digest = hashlib.sha256(signed.astype('<i2').tobytes()).hexdigest()
                    if sensor == 'lepton' and ok:
                        raw_hash_groups[digest].append((row, splits[subject]))
                    record.update(valid_for_numeric_use=int(ok), reason=reason, minimum_c=lo, maximum_c=hi,
                                  mean_c=float(temp.mean()), std_c=float(temp.std()), outside_declared_range_pixels=outside,
                                  declared_endpoint_pixels=endpoints, isolated_jump_gt10c_pixels=isolated,
                                  native_temperature_sha256=digest)
                    counters[sensor]['outside_declared_range_frames'] += int(outside > 0)
                    counters[sensor]['endpoint_present_frames'] += int(endpoints > 0)
                    counters[sensor]['suspected_isolated_jump_frames'] += int(isolated > 0)
                    counters[sensor]['uniform_frames'] += int(float(temp.std()) < 1e-6)
                    counters[sensor][f'numeric_valid_{splits[subject]}'] += int(ok)
                    if not ok:
                        reasons[f'{sensor}:{reason}'] += 1
                except (ValueError, OSError, zipfile.BadZipFile) as exc:
                    caches[sensor][row] = np.nan
                    record.update(valid_for_numeric_use=0, reason=f'{type(exc).__name__}: {exc}')
                    reasons[f'{sensor}:decode_or_shape_failure'] += 1
                writers[sensor].writerow(record)
                if processed % 10000 == 0:
                    print(json.dumps({'frames': processed, 'total': len(entries), 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
        finally:
            for f in handles.values():
                f.close()
            for arr in caches.values():
                arr.flush()
        # Prevent identical source frames in different held-out subjects from leaking.
        cross_split_duplicates = []
        for digest, group in raw_hash_groups.items():
            if len({split for _, split in group}) > 1:
                cross_split_duplicates.append({'sha256': digest, 'cache_indices': [idx for idx, _ in group]})
                for idx, _ in group:
                    valid['lepton'][idx] = False
        for sensor, mask in valid.items():
            np.save(out / f'{sensor}_eligible.npy', mask, allow_pickle=False)
        final_counts = defaultdict(Counter)
        for sensor in counts:
            with (out / f'{sensor}_index.csv').open(encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    final_counts[sensor][row['split']] += int(valid[sensor][int(row['cache_index'])])
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
        'source_archive': archive.name, 'archive_sha256': archive_sha,
        'archive_matches_expected_sha256': True,
        'subject_split_sha256': sha256(out / 'subject_splits.csv'),
        'subject_split_status': 'frozen_for_synthetic_v1_before_training',
        'all_relevant_native_frames_audited': sum(counts.values()), 'native_frame_counts': dict(counts),
        'quality_counts': dict(counters), 'numeric_temperature_range_c': stats, 'rejection_counts': dict(reasons),
        'eligible_counts': dict(final_counts), 'cross_split_exact_duplicate_groups': cross_split_duplicates,
        'numeric_eligibility': 'correct signed int16 dimensions, finite, all pixels within broad [-40,300] C; Lepton cross-split exact duplicates excluded',
        'quality_limit': 'Jump, endpoint and constant-frame flags are diagnostics, not proof of faults. Geometric/radiometric cross-sensor calibration NOT performed.',
        'temperature_unit': 'degrees Celsius according to publisher README; not independently calibrated',
        'lepton_target': 'Full native 160x120 temperature field area-resampled to width64 height48. Synthetic supervision only; no MLX alignment implied.',
        'mlx_role': 'Native 32x24 input QC and unpaired inference only; never used as a paired Lepton ground truth.',
        'no_pseudocolor_training': True, 'raw_source_unchanged': True,
        'seconds': time.perf_counter() - started,
    }
    tmp = out / 'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    tmp.replace(out / 'manifest.json')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
