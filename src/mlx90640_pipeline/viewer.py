"""Shared-scale 2x3 rendering; temperatures remain unchanged by display."""
from pathlib import Path
import time
import cv2
import numpy as np

ORDER_PANELS = ('deinterlaced_c', 'sr_temperature_c', 'final_temperature_c',
                'deinterlaced_c', 'native_temporal_c', 'pre_cnn_temperature_c')
STAGE_PANELS = ('raw_temperature_c', 'frequency_once_c', 'repair_mask',
                'deinterlaced_c', 'native_temporal_c', 'pre_cnn_temperature_c')


def render(arrays, low=24., high=38., scale=5, labels=False, layout='order'):
    if not np.isfinite([low, high]).all() or high <= low or not isinstance(scale, int) or scale < 1:
        raise ValueError('Require finite low < high and positive integer scale')
    if layout not in ('order', 'stages'):
        raise ValueError('Unknown layout')
    panels = ORDER_PANELS if layout == 'order' else STAGE_PANELS
    images = []
    for key in panels:
        a = np.asarray(arrays[key])
        if a.dtype == bool:
            image = np.repeat((a.astype(np.uint8)*255)[..., None], 3, axis=2)
        else:
            index = np.rint(np.clip((a-low)/(high-low), 0, 1)*255).astype(np.uint8)
            image = cv2.applyColorMap(index, cv2.COLORMAP_INFERNO)
        image = cv2.resize(image, (64*scale, 48*scale), interpolation=cv2.INTER_NEAREST)
        if labels:
            label = np.zeros((28, image.shape[1], 3), np.uint8)
            cv2.putText(label, key.replace('_temperature_c', '').replace('_c', ''), (5, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
            image = np.vstack((label, image))
        images.append(image)
    return np.vstack((np.hstack(images[:3]), np.hstack(images[3:])))


def save_preview(path, arrays, **kwargs):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # imencode supports non-ASCII Windows paths, unlike some imwrite builds.
    ok, data = cv2.imencode('.png', render(arrays, **kwargs))
    if not ok:
        raise OSError('PNG encoding failed')
    with path.open('xb') as stream:
        stream.write(data.tobytes())


def show_sequence(arrays, times, labels=False, layout='order', scale=5, low=24., high=38.):
    name = 'MLX90640 replay | Q quit | P pause | T labels | V layout'
    paused, index = False, 0
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = {k: v[index] for k, v in arrays.items()}
            cv2.imshow(name, render(frame, labels=labels, layout=layout, scale=scale, low=low, high=high))
            delay = max(1, int(1000*(times[index+1]-times[index]))) if index+1 < len(times) else 125
            key = cv2.waitKey(30 if paused else min(delay, 1000)) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('t'):
                labels = not labels
            if key == ord('v'):
                layout = 'stages' if layout == 'order' else 'order'
            if not paused:
                index = (index + 1) % len(times)
    finally:
        cv2.destroyAllWindows()
