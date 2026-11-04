"""Calibrated motion scores and observation-preserving update contract."""
import json
from datetime import datetime
import numpy as np
from .spatial import PARITY, SHAPE

SCHEMA = "local_chess_repair_v1"
MAX_GAP_S = 3 / 16
MAX_PAIR_GAP_S = 1.5 / 16


def load_profile(path):
    with np.load(path, allow_pickle=False) as z:
        scales = [z[k] for k in ('sigma_delta', 'sigma_residual')]
        meta = json.loads(str(z['metadata_json']))
    if (meta['version'] != SCHEMA or meta['subpage_hz'] != 16
            or not np.isfinite(meta['cutoff']) or meta['cutoff'] <= 0
            or any(s.shape != SHAPE or not np.isfinite(s).all() or np.any(s <= 0) for s in scales)):
        raise ValueError('Invalid motion repair noise profile')
    return (*scales, meta)


def matches_factory(meta, factory):
    return all(factory.get(k) == meta[k] for k in ('sensor_serial_words', 'eeprom_sha256'))


def timely_pair(conversion, words):
    try:
        if (len(conversion) != 2 or any(c['subpage'] != row[833] or c['control_register_0x800d'] != row[832]
                                        for c, row in zip(conversion, words))
                or (int(words[0,832]) ^ int(words[1,832])) & 0x1f80):
            return False
        t0, t1 = (datetime.fromisoformat(c['read_at_utc']) for c in conversion)
        return 0 < (t1 - t0).total_seconds() <= MAX_PAIR_GAP_S
    except (KeyError, TypeError, ValueError):
        return False


def scores(raw, previous, paper, sigma_delta, sigma_residual):
    residual = np.asarray(paper, np.float64) - raw
    d = np.pad((raw.astype(np.float64) - previous) / sigma_delta, 1, constant_values=np.nan)
    cross = np.stack((d[:-2, 1:-1], d[2:, 1:-1], d[1:-1, :-2], d[1:-1, 2:]))
    signed = cross * np.sign(residual)
    signed[np.isnan(signed)] = -np.inf
    signed.sort(axis=0)
    # Require two supporting current-page neighbours, or one 50% stronger neighbour.
    support = np.maximum(signed[-2], signed[-1] / 1.5)
    return np.minimum(support, np.abs(residual) / sigma_residual)


class MotionRepairBase:
    save_patent_comparison = True

    def __init__(self, profile, calibrated=True):
        self.sigma_delta, self.sigma_residual, meta = profile
        self.cutoff = meta['cutoff']
        self.resolution = meta['resolution_register_bits']
        self.calibrated = calibrated
        self.reset()

    def reset(self):
        self.previous = None
        self.time = None

    def _plan(self, raw, page, paper, tuned75, cold):
        mask = np.zeros(SHAPE, bool)
        candidate_count = 0
        if not cold:
            joint = scores(raw, self.previous, paper, self.sigma_delta, self.sigma_residual)
            candidate = (PARITY != page) & (joint > self.cutoff)
            candidate_count = int(candidate.sum())
            work = raw.astype(np.float32, copy=False)
            mask = candidate & (((paper > work) & (tuned75 > work)) | ((paper < work) & (tuned75 < work)))
        return mask, paper, {}, dict(repair_candidate_count=candidate_count)

    def update(self, frame, timestamp, page, control, paper, tuned75, pair_ok=True):
        raw = np.asarray(frame)
        if (raw.shape != SHAPE or raw.dtype not in (np.float32, np.float64)
                or not np.isfinite(raw).all() or not np.isfinite(timestamp) or page not in (0, 1)
                or not control & 0x1000 or ((control >> 7) & 7) != 5):
            self.reset()
            raise ValueError('Expected native Celsius samples, valid time and a 16 Hz chess subpage')
        if any(a.shape != SHAPE or not np.isfinite(a).all() for a in (paper, tuned75)):
            self.reset()
            raise ValueError('Invalid deinterlacing candidate')
        # Both outputs inherit the original dtype and bit pattern, including signed zero.
        output = raw.copy()
        verified = self.calibrated and ((control >> 10) & 3) == self.resolution
        usable = verified and pair_ok
        cold = not usable or self.previous is None or not 0 < timestamp - self.time <= MAX_GAP_S
        mask, estimate, extra, counts = self._plan(raw, page, paper, tuned75, cold)
        output[mask] = estimate[mask]
        result = dict(deinterlaced_c=output, repair_mask=mask, **extra)
        if self.save_patent_comparison:
            patent_output = raw.copy()
            patent_output[mask] = tuned75[mask]
            result['masked_patent_c'] = patent_output
        self.previous = raw.copy() if usable else None
        self.time = timestamp
        return (result,
                dict(repair_pixel_count=int(mask.sum()), **counts,
                     repair_history_reset=bool(cold), repair_profile_verified=bool(verified), repair_pair_time_ok=bool(pair_ok)))
