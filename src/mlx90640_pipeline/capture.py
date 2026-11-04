"""Raspberry Pi acquisition with untouched Adafruit temperature conversion.

Imports of hardware libraries are lazy so replay and training work on a PC.
Only 16 Hz chess mode is accepted by the calibrated v3 pipeline.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
import numpy as np
from .recording import Recording, SCHEMA, write_npz


def make_sensor_class(driver):
    class AuditedSensor(driver.MLX90640):
        def _CalculateTo(self, frame_data, emissivity, tr, result):
            self.raw_subpages.append([int(word) for word in frame_data])
            self.conversion_inputs.append({
                'subpage': int(frame_data[833]), 'emissivity': float(emissivity),
                'reflected_temperature_c': float(tr),
                'sensor_die_temperature_c': float(self._GetTa(frame_data)),
                'supply_voltage_v': float(self._GetVdd(frame_data)),
                'read_at_utc': datetime.now(timezone.utc).isoformat(),
                'read_monotonic_s': time.monotonic(),
                'control_register_0x800d': int(frame_data[832]),
            })
            return super()._CalculateTo(frame_data, emissivity, tr, result)

        def read_native(self):
            self.raw_subpages, self.conversion_inputs = [], []
            buffer = [float('nan')] * 768
            self.getFrame(buffer)
            if (len(self.raw_subpages) != 2 or
                    {c['subpage'] for c in self.conversion_inputs} != {0, 1}):
                raise RuntimeError('Incomplete subpage pair; frame discarded')
            result = np.asarray(buffer, np.float64).reshape(24, 32)
            if not np.isfinite(result).all():
                raise RuntimeError('Nonfinite temperature; frame discarded')
            return result
    return AuditedSensor


def factory_snapshot(sensor, driver):
    words = np.asarray(driver.eeData, dtype='<u2')
    return dict(sensor='MLX90640', sensor_serial_words=[int(x) for x in sensor.serial_number],
                eeprom_words=words.tolist(), eeprom_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
                driver_version=importlib.metadata.version('adafruit-circuitpython-mlx90640'),
                factory_broken_pixel_indices=[int(x) for x in sensor.brokenPixels],
                factory_outlier_pixel_indices=[int(x) for x in sensor.outlierPixels],
                lens_variant='not inferred from sensor serial',
                temperature_conversion='Adafruit _CalculateTo unchanged; per-subpage assumptions recorded')


class DeviceLock:
    """Cooperative lock plus detection of existing I2C readers owned by this user."""
    def __init__(self, path, device='/dev/i2c-1'):
        self.path, self.device, self.stream = Path(path), device, None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owners = set()
            for process in Path('/proc').iterdir():
                if not process.name.isdigit() or int(process.name) == os.getpid():
                    continue
                try:
                    for fd in (process/'fd').iterdir():
                        try:
                            if os.readlink(fd) == self.device:
                                owners.add(int(process.name))
                        except OSError:
                            pass
                except (OSError, PermissionError):
                    pass
            if owners:
                raise RuntimeError(f'I2C device already open by PID(s) {sorted(owners)}; stop that acquisition first')
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *_):
        self.stream.close()


def run_capture(args):
    if not sys.platform.startswith('linux'):
        raise ValueError('Live acquisition requires Linux and a connected MLX90640')
    if not np.isfinite(args.seconds) or args.seconds < 0 or args.chunk_frames < 2:
        raise ValueError('Require seconds >= 0 and chunk-frames >= 2')
    import board
    import cv2
    import adafruit_mlx90640 as driver
    if importlib.metadata.version('adafruit-circuitpython-mlx90640') != '1.3.9':
        raise ValueError('The audited acquisition adapter requires Adafruit MLX90640 1.3.9')
    from .cli import predictor_from_args
    from .core.motion import load_profile, timely_pair
    from .pipeline import ThermalPipeline
    from .viewer import render, save_preview
    live = args.command == 'live'
    predictor = predictor_from_args(args) if live else None
    profile = load_profile(args.profile) if live else None
    args.output.mkdir(parents=True, exist_ok=False)
    stop = [False]
    def stopped(*_):
        stop[0] = True
    def timeout(*_):
        raise TimeoutError('Sensor read exceeded five seconds')
    old_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGALRM, timeout)
    i2c, factory, pipeline = None, None, None
    records, processed, statuses = [], [], []
    chunk, count, rejected = 0, 0, 0
    start = time.monotonic()
    labels = getattr(args, 'labels', False)
    layout = getattr(args, 'layout', 'order')
    latest = None

    def flush():
        nonlocal chunk
        if not records:
            return
        arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
        if live:
            arrays.update({k: np.stack([r[k] for r in processed]) for k in processed[0]})
            arrays.update({k: np.asarray([r[k] for r in statuses]) for k in statuses[0]})
        metadata = dict(schema=SCHEMA, start_s=start, synthetic=False, subpage_hz=16,
                        pixel_order='row-major; no mirror/rotation', chunk=chunk,
                        history_at_chunk_start='continuous live history; standalone replay resets at its first frame',
                        pipeline=pipeline.config if pipeline else None)
        write_npz(args.output/f'capture_{chunk:04d}.npz', arrays, metadata, factory)
        chunk += 1
        records.clear(); processed.clear(); statuses.clear()

    try:
        with DeviceLock(args.lock_file):
            i2c = board.I2C()
            sensor = make_sensor_class(driver)(i2c)
            sensor.refresh_rate = driver.RefreshRate.REFRESH_16_HZ
            time.sleep(.2)
            factory = factory_snapshot(sensor, driver)
            if factory['factory_broken_pixel_indices'] or factory['factory_outlier_pixel_indices']:
                raise ValueError('This release requires an array without factory-flagged pixels; no silent filling is applied')
            (args.output/'sensor_factory.json').write_text(json.dumps(factory, indent=2), encoding='utf-8')
            pipeline = ThermalPipeline(profile, factory, predictor) if live else None
            cv2.setNumThreads(1)
            # Discard timing-change startup frames before establishing history.
            for _ in range(5):
                signal.alarm(5)
                try:
                    sensor.read_native()
                finally:
                    signal.alarm(0)
            start = time.monotonic()
            while not stop[0] and (args.seconds == 0 or time.monotonic()-start < args.seconds):
                try:
                    signal.alarm(5)
                    raw = sensor.read_native()
                except (RuntimeError, TimeoutError) as exc:
                    rejected += 1
                    if pipeline:
                        pipeline.reset()
                    if rejected >= 5:
                        raise RuntimeError('Five acquisition errors; stopping without repeated device resets') from exc
                    print(f'Rejected acquisition: {exc}', flush=True)
                    continue
                finally:
                    signal.alarm(0)
                now = time.monotonic()
                words = np.asarray(sensor.raw_subpages, np.uint16)
                record = dict(raw_temperature_c=raw, times_s=np.asarray(now), raw_subpages_u16=words,
                              latest_subpage=np.asarray(int(words[-1, 833])),
                              conversion_json=np.asarray(json.dumps(sensor.conversion_inputs)),
                              acquired_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()))
                # Validate raw mode even when the numerical pipeline is disabled.
                Recording({k: np.stack([v]) for k, v in record.items()}, {}, factory).validate()
                records.append(record)
                count += 1
                if live:
                    latest, state = pipeline.update(raw, now, int(words[-1, 833]), int(words[-1, 832]),
                                                    timely_pair(sensor.conversion_inputs, words))
                    processed.append(latest); statuses.append(state)
                    if args.show:
                        image = render(latest, low=args.low, high=args.high, scale=args.scale, labels=labels, layout=layout)
                        cv2.imshow('MLX90640 | Q quit | T labels | V layout | S snapshot', image)
                        key = cv2.waitKey(1) & 255
                        if (key in (27, ord('q')) or cv2.getWindowProperty(
                                'MLX90640 | Q quit | T labels | V layout | S snapshot',
                                cv2.WND_PROP_VISIBLE) < 1):
                            stop[0] = True
                        elif key == ord('t'):
                            labels = not labels
                        elif key == ord('v'):
                            layout = 'stages' if layout == 'order' else 'order'
                        elif key == ord('s'):
                            save_preview(args.output/f'snapshot_{count:06d}.png', latest, low=args.low,
                                         high=args.high, scale=args.scale, labels=labels, layout=layout)
                if len(records) >= args.chunk_frames:
                    flush()
    finally:
        signal.alarm(0)
        try:
            # A processing failure must not discard an otherwise complete raw read.
            if factory is not None:
                if live and len(records) != len(processed):
                    pending = records[len(processed):]
                    raw_arrays = {k: np.stack([r[k] for r in pending]) for k in pending[0]}
                    write_npz(args.output/f'unprocessed_raw_{chunk:04d}.npz', raw_arrays,
                              dict(schema=SCHEMA, start_s=start, synthetic=False,
                                   reason='Complete acquisitions retained after a processing failure'), factory)
                    records[:] = records[:len(processed)]
                flush()
        finally:
            try:
                if i2c is not None:
                    i2c.deinit()
                if live and getattr(args, 'show', False):
                    cv2.destroyAllWindows()
            finally:
                for s, handler in old_signals.items():
                    signal.signal(s, handler)
    print(json.dumps(dict(frames=count, rejected=rejected, chunks=chunk, output=str(args.output))))
    return 0
