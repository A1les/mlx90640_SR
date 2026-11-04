"""Latest-subpage cross interpolation used by the motion detector."""
import numpy as np
import cv2

SHAPE = (24, 32)
PARITY = np.indices(SHAPE).sum(axis=0) % 2
CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)


def validate(frame, latest):
    a = np.asarray(frame, np.float32)
    if a.shape != SHAPE or not np.isfinite(a).all() or latest not in (0, 1):
        raise ValueError('Expected finite 24x32 Celsius frame and subpage 0 or 1')
    return a


def paper_subframe(frame, latest):
    a = validate(frame, latest)
    mask = (PARITY == latest).astype(np.float32)
    total = cv2.filter2D(a * mask, -1, CROSS, borderType=cv2.BORDER_CONSTANT)
    count = cv2.filter2D(mask, -1, CROSS, borderType=cv2.BORDER_CONSTANT)
    interpolated = total / np.maximum(count, 1)
    return np.where(mask != 0, a, interpolated).astype(np.float32)
