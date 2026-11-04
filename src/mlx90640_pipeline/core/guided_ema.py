"""Apply shared native weights on either temperature grid."""
import numpy as np


class NativeGuidedEMA:
    """Apply one native alpha map to the same field of view before or after 2x SR."""
    def __init__(self, reset_gap_s=.75):
        if not np.isfinite(reset_gap_s) or reset_gap_s <= 0:
            raise ValueError('reset_gap_s must be positive')
        self.reset_gap_s = float(reset_gap_s)
        self.reset()

    def reset(self):
        self.state = None
        self.timestamp = None

    def update(self, temperature_c, timestamp_s, native_alpha):
        current = np.asarray(temperature_c, np.float32)
        weight = np.asarray(native_alpha, np.float32)
        if (current.shape not in ((24, 32), (48, 64)) or weight.shape != (24, 32) or
                not np.isfinite(current).all() or not np.isfinite(timestamp_s) or
                not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1))):
            raise ValueError('Expected finite native or 2x temperatures and native EMA weights in [0,1]')
        if current.shape == (48, 64):
            weight = weight.repeat(2, axis=0).repeat(2, axis=1)
        restarted = (self.state is None or current.shape != self.state.shape or
                     not 0 < timestamp_s - self.timestamp <= self.reset_gap_s)
        if restarted:
            weight = np.ones(current.shape, np.float32)
        if restarted or np.all(weight == 1):
            self.state = current.copy()
        else:
            self.state = weight * current + (1 - weight) * self.state
        self.timestamp = float(timestamp_s)
        return self.state.copy(), restarted, weight
