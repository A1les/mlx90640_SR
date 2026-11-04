"""Patent-derived research variant; no claim of equivalence to Melexis binaries.

Gaussian/box frequency broadening follows EP4192002A1, paragraphs 0027-0038.
Complement, normalization, boundary extension, numerical widths and partial
strength are explicit implementation choices. Strength < 1 is an engineering
extension (linear blend with input), not a published Melexis recommendation.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=256)
def transfer(shape=(24, 32), kernel='gaussian', width=.18):
    if len(shape) != 2 or any(n < 2 or n % 2 for n in shape):
        raise ValueError('Expected even dimensions')
    if not np.isfinite(width) or not 0 < width < .5:
        raise ValueError('Width must lie in (0,.5) cycles/pixel')
    # A binary chess mask has just one non-DC impulse, at the Nyquist corner.
    fy, fx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing='ij')
    # Equivalent periodic distances to -0.5, computed symmetrically to avoid
    # rounding differences at a hard box cutoff on conjugate frequency bins.
    dy = .5 - np.abs(fy)
    dx = .5 - np.abs(fx)
    if kernel == 'gaussian':
        rejection = np.exp(-.5 * (dx*dx + dy*dy) / width**2)
    elif kernel == 'box':
        rejection = ((dx <= width + 1e-12) & (dy <= width + 1e-12)).astype(float)
    else:
        raise ValueError(kernel)
    h = 1 - rejection / rejection.max()
    h[0, 0] = 1.
    h.setflags(write=False)
    return h


def filter_temperature(frames, kernel='gaussian', width=.18, strength=1., boundary='circular'):
    a = np.asarray(frames, np.float32)
    if a.ndim not in (2,3) or a.shape[-2:] != (24,32) or not np.isfinite(a).all():
        raise ValueError('Expected finite native Celsius matrices')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must lie in [0,1]')
    pads = {'circular':0, 'reflect8':8, 'reflect16':16}
    if boundary not in pads:
        raise ValueError(boundary)
    pad = pads[boundary]
    padding = [(0,0)]*(a.ndim-2) + [(pad,pad),(pad,pad)]
    x = np.pad(a, padding, mode='reflect') if pad else a
    h = 1 - strength * (1 - transfer(x.shape[-2:], kernel, width))
    y = np.fft.ifft2(np.fft.fft2(x.astype(np.float64), axes=(-2,-1))*h, axes=(-2,-1))
    if np.max(np.abs(y.imag)) > 1e-7:
        raise ArithmeticError('Transfer must be conjugate symmetric')
    y = y.real
    if pad:
        y = y[...,pad:-pad,pad:-pad]
    return y.astype(np.float32)
