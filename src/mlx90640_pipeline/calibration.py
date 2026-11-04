"""Fit the active v3 detector's robust noise profile on a stationary interval."""
import hashlib
import numpy as np
from .core.motion import SCHEMA, MAX_GAP_S, scores
from .core.spatial import PARITY, paper_subframe
from .recording import write_npz


def mad(values):
    return np.maximum(1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0), 1e-8)


def fit_profile(recording, start_s=5., end_s=17.5, quantile=.999):
    recording.validate()
    if not np.isfinite([start_s, end_s, quantile]).all() or not 0 <= start_s < end_s or not .5 < quantile < 1:
        raise ValueError('Invalid calibration interval or quantile')
    a = recording.arrays
    raw, times, pages = a['raw_temperature_c'], a['times_s'], a['latest_subpage']
    ix = np.flatnonzero((recording.elapsed >= start_s) & (recording.elapsed < end_s))
    if len(ix) < 30 or ix[0] == 0 or np.any(np.diff(ix) != 1):
        raise ValueError('Need at least 30 consecutive stationary frames and a preceding frame')
    controls = a['raw_subpages_u16'][ix, :, 832]
    if (not a['pair_ok'][ix].all() or np.any(np.diff(times)[ix - 1] > MAX_GAP_S)
            or np.unique((controls >> 10) & 3).size != 1):
        raise ValueError('Calibration contains invalid pair timing, frame gaps or changed resolution')
    sample = raw[ix]
    delta = mad(np.diff(sample, axis=0))
    residual = np.empty((24, 32))
    for page in (0, 1):
        old = PARITY != page
        crosses = np.stack([paper_subframe(f, page) for f in sample])
        residual[old] = mad(crosses[:, old] - sample[:, old])
    observed = np.concatenate([
        scores(raw[i], raw[i-1], paper_subframe(raw[i], pages[i]), delta, residual)[PARITY != pages[i]]
        for i in ix
    ])
    cutoff = float(np.quantile(observed, quantile))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Degenerate calibration: no positive score quantile')
    metadata = dict(version=SCHEMA, subpage_hz=16,
                    resolution_register_bits=int((controls[0, 0] >> 10) & 3),
                    fit_interval_s=[start_s, end_s], fit_frames=len(ix), fit_scores=len(observed),
                    fit_pages=np.unique(pages[ix]).tolist(), quantile=quantile, cutoff=cutoff,
                    calibration='per-detector temporal-difference MAD and spatial-residual MAD',
                    sensor_serial_words=recording.factory['sensor_serial_words'],
                    eeprom_sha256=recording.factory['eeprom_sha256'],
                    synthetic=bool(recording.metadata.get('synthetic', False)))
    return delta, residual, metadata


def save_profile(path, profile, recording_path=None):
    delta, residual, metadata = profile
    metadata = dict(metadata)
    if recording_path:
        from pathlib import Path
        p = Path(recording_path)
        metadata.update(fit_recording=p.name, fit_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    write_npz(path, {'sigma_delta': delta, 'sigma_residual': residual}, metadata)
