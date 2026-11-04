"""Causal temperature smoothing, separate from the frozen SR model."""
import numpy as np


class TemperatureEMA:
    def __init__(self, alpha=0.5, reset_gap_s=0.75):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alpha must be in (0, 1]')
        if not np.isfinite(reset_gap_s) or reset_gap_s <= 0:
            raise ValueError('reset_gap_s must be positive')
        self.alpha = float(alpha)
        self.reset_gap_s = float(reset_gap_s)
        self.reset()

    def reset(self):
        self.state = None
        self.timestamp = None

    def update(self, temperature_c, timestamp_s):
        current = np.asarray(temperature_c, dtype=np.float32)
        if current.ndim != 2 or not np.isfinite(current).all() or not np.isfinite(timestamp_s):
            raise ValueError('Expected a finite temperature matrix and timestamp')
        restarted = (self.state is None or self.state.shape != current.shape or
                     not 0 < timestamp_s - self.timestamp <= self.reset_gap_s)
        if restarted:
            self.state = current.copy()
        else:
            self.state = self.alpha * current + (1.0 - self.alpha) * self.state
        self.timestamp = float(timestamp_s)
        return self.state.copy(), restarted


class MotionAwareTemperatureEMA(TemperatureEMA):
    """Use current samples where spatially coherent temperature changes are large.

    This is a change detector, not optical flow or proof of physical motion.
    Low-contrast changes can still lag; strong noise can trigger the fast path.
    """
    def __init__(self, alpha=.5, low_c=.9, high_c=1.8, hold_frames=2, reset_gap_s=.75):
        if not np.isfinite([low_c,high_c]).all() or not 0 <= low_c < high_c or hold_frames < 1:
            raise ValueError('Invalid motion sensitivity or hold time')
        self.low_c = float(low_c)
        self.high_c = float(high_c)
        self.hold_frames = int(hold_frames)
        super().__init__(alpha,reset_gap_s)

    def reset(self):
        super().reset()
        self.hold = None

    def update(self, temperature_c, timestamp_s):
        import cv2
        current = np.asarray(temperature_c,dtype=np.float32)
        if current.ndim != 2 or not np.isfinite(current).all() or not np.isfinite(timestamp_s):
            raise ValueError('Expected a finite temperature matrix and timestamp')
        restarted = (self.state is None or self.state.shape != current.shape or
                     not 0 < timestamp_s-self.timestamp <= self.reset_gap_s)
        if restarted:
            self.state = current.copy()
            self.hold = np.zeros(current.shape,np.int16)
            weight = np.ones(current.shape,np.float32)
        else:
            # Average signed innovations to suppress isolated random noise;
            # dilate the detector by one SR pixel to cover changing edges.
            cue = cv2.dilate(np.abs(cv2.blur(current-self.state,(3,3))),np.ones((3,3),np.uint8))
            weight = self.alpha+(1-self.alpha)*np.clip((cue-self.low_c)/(self.high_c-self.low_c),0,1)
            self.hold = np.maximum(self.hold-1,np.where(cue >= self.high_c,self.hold_frames,0))
            weight = np.where(self.hold>0,1.,weight).astype(np.float32)
            self.state = weight*current+(1-weight)*self.state
        self.timestamp = float(timestamp_s)
        return self.state.copy(), restarted, weight


class MotionGatedTemperatureEMA(MotionAwareTemperatureEMA):
    """Bypass temporal history for the whole frame during substantial changes."""
    def __init__(self, alpha=.5, low_c=.9, high_c=1.8, change_c=1., change_fraction=.03, hold_frames=4):
        if not np.isfinite([change_c,change_fraction]).all() or change_c <= 0 or not 0 < change_fraction < 1 or hold_frames < 1:
            raise ValueError('Invalid global change gate')
        self.change_c = float(change_c)
        self.change_fraction = float(change_fraction)
        self.global_hold_frames = int(hold_frames)
        super().__init__(alpha,low_c,high_c)

    def reset(self):
        super().reset()
        self.previous_input = None
        self.global_hold = 0
        self.bypassed = False
        self.changed_fraction = 0.

    def update(self, temperature_c, timestamp_s):
        import cv2
        output, restarted, weight = super().update(temperature_c,timestamp_s)
        current = np.asarray(temperature_c,dtype=np.float32)
        if restarted:
            self.previous_input = None
            self.global_hold = 0
        self.changed_fraction = (float((np.abs(cv2.blur(current-self.previous_input,(5,5))) > self.change_c).mean())
                                 if self.previous_input is not None else 0.)
        if self.changed_fraction > self.change_fraction:
            self.global_hold = self.global_hold_frames
        self.bypassed = self.global_hold > 0
        if self.bypassed:
            # Equal to this frame's unfiltered SR, so this frame has no filter-added trail.
            self.state = current.copy()
            output = current.copy()
            weight = np.ones(current.shape,np.float32)
            self.global_hold -= 1
        self.previous_input = current.copy()
        return output, restarted, weight
