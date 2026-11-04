"""Frozen CNN followed by native-guided temporal filtering of the repaired frame."""
import cv2
import numpy as np

from .guided_ema import NativeGuidedEMA
from .temporal import MotionGatedTemperatureEMA


VERSION = 'confirmed_repair_cnn_temporal_v1'
ARRAYS = ('sr_temperature_c', 'final_temperature_c', 'temporal_native_alpha',
          'temporal_alpha', 'temporal_history_before_c', 'temporal_release_mask')
ORDER_ARRAYS = ('native_temporal_c', 'native_temporal_history_before_c', 'pre_cnn_temperature_c')
STATES = ('temporal_history_reset', 'temporal_global_motion', 'temporal_passthrough',
          'temporal_current_weight_mean')


class PostprocessPipeline:
    """Use deinterlaced_c directly as CNN input and retain all stages separately."""
    def __init__(self, session, compare_order=False):
        self.session = session
        self.gate = MotionGatedTemperatureEMA()
        self.filter = NativeGuidedEMA()
        self.native_filter = NativeGuidedEMA() if compare_order else None
        self.array_keys = ARRAYS + ORDER_ARRAYS if compare_order else ARRAYS
        self.config = {
            'version': VERSION,
            'chain': ['deinterlaced_c', 'sr_temperature_c', 'final_temperature_c'],
            'cnn_input': 'float32 copy of confirmed_frequency_repair_v3 deinterlaced_c',
            'model_sha256': session.model_sha256,
            'model_kind': session.kind,
            'input_grid_width_height': [32, 24],
            'output_grid_width_height': [64, 48],
            'temperature_unit': 'estimated_degC',
            'temporal': {
                'alpha': .5, 'low_c': .9, 'high_c': 1.8,
                'global_change_c': 1., 'global_fraction': .03, 'global_hold_frames': 4,
                'confirmed_region_dilation': [3, 3], 'region_hold_frames': 2,
                'weight_mapping': 'each native weight covers its 2x2 reconstructed cells',
                'reset': 'repair history reset or time gap over 0.75 s',
                'history_key': 'temporal_history_before_c',
            },
        }
        if compare_order:
            self.config.update(
                version='confirmed_repair_order_comparison_v1',
                comparison_chain=['deinterlaced_c', 'native_temporal_c', 'pre_cnn_temperature_c'],
                comparison_panel_order=['deinterlaced_c', 'sr_temperature_c', 'final_temperature_c',
                                        'deinterlaced_c', 'native_temporal_c', 'pre_cnn_temperature_c'],
                comparison_weights='Both orders use the same native alpha and reset decisions')
        self.reset()

    def reset(self):
        self.gate.reset()
        self.filter.reset()
        if self.native_filter is not None:
            self.native_filter.reset()
        self.hold = np.zeros((24, 32), np.int16)

    def update(self, deinterlaced, eligible, timestamp, repair_reset):
        current = np.asarray(deinterlaced, np.float32)
        region = np.asarray(eligible, bool)
        if current.shape != (24, 32) or region.shape != current.shape:
            raise ValueError('Expected native temperatures and confirmed repair region')
        if not np.isfinite(current).all() or not np.isfinite(timestamp):
            raise ValueError('Expected finite temperature and timestamp')
        if repair_reset:
            self.reset()

        # The gate estimates weights only. Its smoothed image never enters the CNN.
        _, gate_reset, native_alpha = self.gate.update(current, timestamp)
        if gate_reset:
            self.hold.fill(0)
        expanded = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
        self.hold = np.maximum(self.hold - 1, np.where(expanded, 2, 0)).astype(np.int16)
        release = self.hold > 0
        native_alpha[release] = 1.
        self.gate.state[release] = current[release]

        sr = self.session.predict(current)
        history = sr.copy() if self.filter.state is None else self.filter.state.copy()
        final, filter_reset, alpha = self.filter.update(sr, timestamp, native_alpha)
        if filter_reset:
            native_alpha = np.ones((24, 32), np.float32)
            history = sr.copy()
        arrays = dict(sr_temperature_c=sr, final_temperature_c=final,
                      temporal_native_alpha=native_alpha.copy(), temporal_alpha=alpha,
                      temporal_history_before_c=history, temporal_release_mask=release)
        if self.native_filter is not None:
            native_history = current.copy() if self.native_filter.state is None else self.native_filter.state.copy()
            native_temporal, native_reset, _ = self.native_filter.update(current, timestamp, native_alpha)
            if native_reset:
                native_history = current.copy()
            arrays.update(native_temporal_c=native_temporal,
                          native_temporal_history_before_c=native_history,
                          pre_cnn_temperature_c=self.session.predict(native_temporal))
        state = dict(temporal_history_reset=bool(gate_reset or filter_reset),
                     temporal_global_motion=bool(self.gate.bypassed),
                     temporal_passthrough=bool(np.all(alpha == 1)),
                     temporal_current_weight_mean=float(alpha.mean()))
        return arrays, state

    def sequence(self, native_arrays, native_states, times):
        self.reset()
        arrays = {key: [] for key in self.array_keys}
        states = {key: [] for key in STATES}
        for i, timestamp in enumerate(times):
            values, flags = self.update(native_arrays['deinterlaced_c'][i],
                                       native_arrays['repair_eligible'][i], float(timestamp),
                                       bool(native_states['repair_history_reset'][i]))
            for key in arrays:
                arrays[key].append(values[key])
            for key in states:
                states[key].append(flags[key])
        return ({key: np.stack(value) for key, value in arrays.items()},
                {key: np.asarray(value) for key, value in states.items()})

    def check_saved(self, deinterlaced, arrays, states):
        count = len(deinterlaced)
        for key in self.array_keys:
            grid = (24, 32) if key in ('temporal_native_alpha', 'temporal_release_mask',
                                      'native_temporal_c', 'native_temporal_history_before_c') else (48, 64)
            if arrays[key].shape != (count, *grid) or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid saved postprocessing array ' + key)
        if any(states[key].shape != (count,) for key in STATES):
            raise ValueError('Invalid saved postprocessing flags')
        native_alpha, alpha = arrays['temporal_native_alpha'], arrays['temporal_alpha']
        if (np.any((alpha < 0) | (alpha > 1)) or
                not np.array_equal(alpha, native_alpha.repeat(2, axis=1).repeat(2, axis=2))):
            raise ValueError('Invalid native to SR temporal weights')
        sr, final = arrays['sr_temperature_c'], arrays['final_temperature_c']
        expected = alpha * sr + (1 - alpha) * arrays['temporal_history_before_c']
        if not np.array_equal(final, expected):
            raise ValueError('Saved final output disagrees with its weights and history')
        for i, frame in enumerate(deinterlaced):
            if not np.allclose(sr[i], self.session.predict(frame), atol=2e-5, rtol=0):
                raise ValueError('Saved CNN output disagrees with the repaired input')
        if self.native_filter is not None:
            expected_native = (native_alpha * deinterlaced.astype(np.float32) +
                               (1 - native_alpha) * arrays['native_temporal_history_before_c'])
            if not np.array_equal(arrays['native_temporal_c'], expected_native):
                raise ValueError('Pre-CNN filter disagrees with the shared weights and its native history')
            for i, frame in enumerate(arrays['native_temporal_c']):
                if not np.allclose(arrays['pre_cnn_temperature_c'][i], self.session.predict(frame), atol=2e-5, rtol=0):
                    raise ValueError('Lower-row CNN output disagrees with the temporally filtered input')
