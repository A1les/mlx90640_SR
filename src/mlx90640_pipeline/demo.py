"""Deterministic synthetic scenes, never presented as sensor measurements."""
from datetime import datetime, timedelta, timezone
import json
import cv2
import numpy as np
from .recording import Recording, SCHEMA
from .core.spatial import PARITY

SYNTHETIC_FACTORY = {'sensor_serial_words': [-1, -1, -1], 'eeprom_sha256': '0' * 64,
                     'sensor': 'synthetic_no_physical_device',
                     'factory_broken_pixel_indices': [], 'factory_outlier_pixel_indices': []}


def scene(t, moving):
    y, x = np.indices((24, 32), dtype=float)
    shift = 0. if not moving or t < 5 else (4 * np.sin((t - 5) * (1 if t < 15 else 3)))
    if moving and t >= 25:
        return np.full((24, 32), 26.)
    x = x - shift
    hand = (x > 8) & (x < 23) & (y >= 12) & (y < 20)
    for i, height in enumerate([7, 4, 3, 5, 8]):
        hand |= (x >= 8 + 3*i) & (x < 10 + 3*i) & (y >= height) & (y < 14)
    return 26. + 8 * cv2.GaussianBlur(hand.astype(float), (3, 3), .45)


def synthetic_recording(frames=280, moving=True, seed=20261104):
    if frames < 2:
        raise ValueError('At least two frames are required')
    rng = np.random.default_rng(seed)
    times = np.arange(frames, dtype=float) / 8
    pages = np.arange(frames) % 2
    raw, truths, conversions = [], [], []
    words = np.zeros((frames, 2, 834), np.uint16)
    words[:, :, 832] = 0x1a81
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, (t, p) in enumerate(zip(times, pages)):
        current, older = scene(t, moving), scene(t - 1/16, moving)
        raw.append(np.where(PARITY == p, current, older) + rng.normal(0, .4, (24, 32)))
        truths.append(current)
        words[i, :, 833] = [1-p, p]
        conversions.append(json.dumps([
            {'subpage': int(page), 'control_register_0x800d': 0x1a81,
             'read_at_utc': (epoch + timedelta(seconds=float(at))).isoformat()}
            for page, at in [(1-p, t-1/16), (p, t)]
        ]))
    arrays = dict(raw_temperature_c=np.stack(raw), times_s=times, latest_subpage=pages,
                  raw_subpages_u16=words, conversion_json=np.asarray(conversions),
                  synthetic_truth_c=np.stack(truths))
    metadata = dict(schema=SCHEMA, start_s=0., synthetic=True, seed=seed,
                    raw_ram_status='Synthetic zero placeholders; NOT a DN-to-temperature example',
                    purpose='Algorithm invariants and runnable demonstration, not accuracy evidence')
    return Recording(arrays, metadata, SYNTHETIC_FACTORY.copy()).validate()
