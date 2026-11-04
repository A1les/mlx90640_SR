"""Portable orchestration of the unchanged v3 kernels and two temporal orders."""
import numpy as np
from .core.spatial import PARITY, paper_subframe
from .core.frequency import filter_temperature
from .core.motion import matches_factory
from .core.repair import MotionAdaptiveDeinterlacer, PATENT, VERSION
from .core.postprocess import PostprocessPipeline


class ThermalPipeline:
    def __init__(self, profile, factory, predictor):
        if not matches_factory(profile[2], factory):
            raise ValueError('Noise profile does not match this sensor identity and EEPROM hash')
        self.native = MotionAdaptiveDeinterlacer(profile)
        self.postprocess = PostprocessPipeline(predictor, compare_order=True)
        self.config = dict(native_version=VERSION, frequency=PATENT.copy(),
                           postprocess=self.postprocess.config, latest_observations_protected=True)

    def reset(self):
        self.native.reset()
        self.postprocess.reset()

    def update(self, raw, timestamp, latest_subpage, control_register, pair_ok=True):
        original = np.asarray(raw)
        paper = paper_subframe(original, latest_subpage)
        frequency = filter_temperature(original, **PATENT)
        arrays, states = self.native.update(original, timestamp, latest_subpage, control_register,
                                            paper, frequency, pair_ok=pair_ok)
        more, flags = self.postprocess.update(arrays['deinterlaced_c'], arrays['repair_eligible'],
                                               timestamp, states['repair_history_reset'])
        arrays.update(more)
        states.update(flags)
        arrays.update(raw_temperature_c=original.copy(), paper_c=paper, frequency_once_c=frequency)
        protected = ~arrays['repair_mask']
        if (np.any(arrays['repair_mask'] & (PARITY == latest_subpage))
                or arrays['deinterlaced_c'][protected].tobytes() != original[protected].tobytes()):
            raise AssertionError('Protected original observations changed')
        return arrays, states

    def sequence(self, recording):
        recording.validate()
        self.reset()
        values, flags = {}, {}
        a = recording.arrays
        for i, timestamp in enumerate(a['times_s']):
            arrays, states = self.update(a['raw_temperature_c'][i], float(timestamp),
                                         int(a['latest_subpage'][i]), int(a['raw_subpages_u16'][i, -1, 832]),
                                         bool(a['pair_ok'][i]))
            for k, v in arrays.items():
                values.setdefault(k, []).append(v)
            for k, v in states.items():
                flags.setdefault(k, []).append(v)
        return ({k: np.stack(v) for k, v in values.items()},
                {k: np.asarray(v) for k, v in flags.items()})
