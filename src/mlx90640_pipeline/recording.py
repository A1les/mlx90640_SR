"""Validated, pickle-free native recordings and atomic output writes."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import uuid
import numpy as np
from .core.motion import timely_pair

SCHEMA = 'mlx90640_temperature_recording_v1'


def write_npz(path, arrays, metadata, factory=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dict(arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    if factory is not None:
        values['factory_json'] = np.asarray(json.dumps(factory, sort_keys=True))
    if any(np.asarray(v).dtype.hasobject for v in values.values()):
        raise ValueError('Object arrays are not permitted')
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.part')
    try:
        # Fail immediately on denied directories; do not retry randomized names.
        with tmp.open('xb') as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        # Never overwrite an existing acquisition or analysis.
        # Linking a same-directory temporary file is atomic and fails if the
        # destination exists, including on POSIX where rename would overwrite.
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@dataclass
class Recording:
    arrays: dict
    metadata: dict
    factory: dict

    def __len__(self):
        return len(self.arrays['times_s'])

    @property
    def elapsed(self):
        t = self.arrays['times_s']
        return t - self.metadata.get('start_s', float(t[0]))

    def validate(self):
        a = self.arrays
        required = ('raw_temperature_c', 'times_s', 'raw_subpages_u16', 'latest_subpage', 'conversion_json')
        if any(k not in a for k in required):
            raise ValueError('Missing raw temperatures, subpages, timestamps or conversion provenance')
        t, raw, words, pages = (a['times_s'], a['raw_temperature_c'],
                                a['raw_subpages_u16'], a['latest_subpage'])
        n = len(t)
        if (not n or t.shape != (n,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
                or raw.shape != (n, 24, 32) or raw.dtype not in (np.float32, np.float64)
                or not np.isfinite(raw).all() or words.dtype != np.uint16
                or words.shape != (n, 2, 834) or pages.shape != (n,)
                or not np.array_equal(words[:, -1, 833], pages)
                or not np.all(np.sort(words[:, :, 833], axis=1) == [0, 1])
                or not np.all(words[:, :, 832] & 0x1000)
                or not np.all(((words[:, :, 832] >> 7) & 7) == 5)
                or a['conversion_json'].shape != (n,)):
            raise ValueError('Invalid native data; this release requires finite 16 Hz chess recordings')
        if any(k not in self.factory for k in ('sensor_serial_words', 'eeprom_sha256')):
            raise ValueError('Sensor identity and EEPROM digest are required')
        # Timing failures remain in the raw record, but disable repair for that frame.
        self.arrays['pair_ok'] = np.asarray([
            timely_pair(json.loads(str(c)), w) for c, w in zip(a['conversion_json'], words)
        ], dtype=bool)
        return self


def load_recording(path, factory_path=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k not in ('metadata_json', 'factory_json')}
        metadata = json.loads(str(z['metadata_json'])) if 'metadata_json' in z else {}
        factory = json.loads(str(z['factory_json'])) if 'factory_json' in z else None
    if factory is None:
        p = Path(factory_path) if factory_path else path.with_name('sensor_factory.json')
        factory = json.loads(p.read_text(encoding='utf-8'))
    return Recording(arrays, metadata, factory).validate()
