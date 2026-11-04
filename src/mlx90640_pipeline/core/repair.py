"""Confirmed frequency repair v3. Only confirmed older observations may change."""
import cv2
import numpy as np
from .frequency import filter_temperature
from .spatial import PARITY, SHAPE
from .motion import MotionRepairBase, scores

VERSION = "confirmed_frequency_repair_v3"
PATENT = {"kernel": "gaussian", "width": .22, "strength": .75, "boundary": "reflect8"}
PASSES = 16
PATENT_NOISE_RATIO = .28
RAW, PATENT_PROJECTED = 0, 3
NEIGHBORS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
ARRAYS = ("deinterlaced_c", "repair_mask", "repair_source", "repair_proposed", "repair_eligible")
STATES = ("repair_pixel_count", "repair_candidate_count", "repair_history_reset", "repair_profile_verified", "repair_pair_time_ok")


def count_near(mask, size):
    return cv2.boxFilter(mask.astype(np.float32), -1, (size,size), normalize=False, borderType=cv2.BORDER_CONSTANT)


def projected_patent(raw, eligible, tuned75):
    # Each application is the frozen frequency filter. Only eligible older sites may move between passes.
    output = raw.copy()
    if eligible.any():
        work = raw.astype(np.float32, copy=True)
        work[eligible] = tuned75[eligible]
        for _ in range(1, PASSES):
            work[eligible] = filter_temperature(work, **PATENT)[eligible]
        output[eligible] = work[eligible]
    return output


class RegionalRepair(MotionRepairBase):
    def _candidates(self, raw, page, paper, tuned75):
        old = PARITY != page
        scale = self.sigma_residual
        cross_z = np.abs(paper.astype(float)-raw)/scale
        patent_z = np.abs(tuned75.astype(float)-raw)/(PATENT_NOISE_RATIO*scale)
        delta_z = np.abs(raw.astype(float)-self.previous)/self.sigma_delta
        joint = scores(raw, self.previous, paper, self.sigma_delta, scale)
        seeds = old & (joint > self.cutoff)
        region = (count_near(delta_z>2.5,5)>=5) | (count_near(seeds,7)>=3)
        eligible = (seeds & (joint>1.5*self.cutoff)) | (old & region & ((cross_z>.85) | (patent_z>1.5)))
        return eligible, cross_z, delta_z


def confirmed_regions(eligible, support):
    if not support.any() or np.array_equal(support, eligible):
        return support
    # The temporary bridges join older chess sites at L1 distance 2. Only original candidates are writable.
    bridges = cv2.dilate(eligible.astype(np.uint8), NEIGHBORS)
    n, labels = cv2.connectedComponents(bridges, connectivity=4)
    active = np.zeros(n, bool)
    active[labels[support]] = True
    return eligible & active[labels]


class MotionAdaptiveDeinterlacer(RegionalRepair):
    save_patent_comparison = False

    def reset(self):
        super().reset()
        self.previous_proposed = np.zeros(SHAPE, bool)

    def _plan(self, raw, page, paper, tuned75, cold):
        proposed = np.zeros(SHAPE, bool)
        eligible = proposed
        if not cold:
            proposed, _, delta_z = self._candidates(raw, page, paper, tuned75)
            strong = count_near(delta_z > 3.5, 5) >= 3
            recent = count_near(self.previous_proposed, 5) >= 2
            eligible = confirmed_regions(proposed, proposed & (strong | recent))
        self.previous_proposed = proposed.copy()
        estimate = projected_patent(raw, eligible, tuned75)
        mask = eligible & (estimate != raw)
        source = np.where(mask, PATENT_PROJECTED, RAW).astype(np.uint8)
        return (mask, estimate, dict(repair_proposed=proposed, repair_eligible=eligible, repair_source=source),
                dict(repair_candidate_count=int(eligible.sum())))


def check_saved(raw, pages, arrays, states, tuned75):
    """Verify actual writes, original bits and the estimator on cached clips, including warm snapshots."""
    proposed, eligible, mask = (arrays[k] for k in ('repair_proposed', 'repair_eligible', 'repair_mask'))
    source, output = arrays['repair_source'], arrays['deinterlaced_c']
    if (any(a.shape != raw.shape or a.dtype != np.bool_ for a in (proposed, eligible, mask))
            or np.any(proposed & (PARITY == pages[:, None, None])) or np.any(eligible & ~proposed)
            or np.any(mask & ~eligible) or source.dtype != np.uint8
            or not np.array_equal(source, np.where(mask, PATENT_PROJECTED, RAW))):
        raise ValueError('Invalid saved candidate, write mask or single repair source')
    if (any(states[k].shape != (len(raw),) for k in STATES)
            or not np.array_equal(eligible.sum(axis=(1, 2)), states['repair_candidate_count'])
            or not np.array_equal(mask.sum(axis=(1, 2)), states['repair_pixel_count'])):
        raise ValueError('Saved counts disagree with the confirmed region or the actual writes')
    unusable = (states['repair_history_reset'].astype(bool) | ~states['repair_profile_verified'].astype(bool)
                | ~states['repair_pair_time_ok'].astype(bool))
    if (proposed[unusable].any() or output.shape != raw.shape or output.dtype != raw.dtype
            or output[~mask].tobytes() != raw[~mask].tobytes()):
        raise ValueError('Invalid saved history or modified original measurements')
    for i, frame in enumerate(raw):
        estimate = projected_patent(frame, eligible[i], tuned75[i])
        if (not np.array_equal(mask[i], eligible[i] & (estimate != frame))
                or output[i][mask[i]].tobytes() != estimate[mask[i]].tobytes()):
            raise ValueError('Saved output differs from the single constrained frequency filter')
