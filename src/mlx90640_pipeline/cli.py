"""Command-line entry points for demo, calibration, replay and Raspberry Pi capture."""
import argparse
import json
from pathlib import Path
import numpy as np
from .calibration import fit_profile, save_profile
from .core.motion import load_profile
from .inference import BilinearBaseline, TorchPredictor, OnnxPredictor
from .pipeline import ThermalPipeline
from .recording import SCHEMA, load_recording, write_npz
from .viewer import save_preview, show_sequence


def predictor_from_args(args):
    if args.onnx:
        return OnnxPredictor(args.onnx, expected_sha256=args.model_sha256)
    if args.checkpoint:
        return TorchPredictor(args.checkpoint, expected_sha256=args.model_sha256)
    if not getattr(args, 'baseline', False) and args.command != 'demo':
        raise ValueError('Specify --onnx, --checkpoint, or explicitly select --baseline')
    print('Using BILINEAR BASELINE: these outputs are not CNN results.', flush=True)
    return BilinearBaseline()


def run_recording(recording, profile, predictor, output):
    pipeline = ThermalPipeline(profile, recording.factory, predictor)
    arrays, states = pipeline.sequence(recording)
    metadata = dict(schema=SCHEMA, start_s=recording.metadata.get('start_s', float(recording.arrays['times_s'][0])),
                    synthetic=bool(recording.metadata.get('synthetic', False)),
                    source_metadata=recording.metadata, pipeline=pipeline.config,
                    units='estimated_degC', observations_preserved_at='deinterlaced_c only')
    raw_keys = ('raw_temperature_c', 'times_s', 'raw_subpages_u16', 'latest_subpage',
                'conversion_json', 'acquired_at_utc', 'pair_ok', 'synthetic_truth_c', 'raw_ir_counts')
    native = {k: recording.arrays[k] for k in raw_keys if k in recording.arrays}
    write_npz(output, dict(native, **arrays, **states), metadata, recording.factory)
    return arrays, states


def add_model(p):
    group = p.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--onnx', type=Path)
    group.add_argument('--baseline', action='store_true', help='Explicit interpolation-only baseline, no CNN')
    p.add_argument('--model-sha256', help='Optional expected checkpoint or ONNX digest')


def add_display(p):
    p.add_argument('--show', action='store_true', help='Open the desktop GUI')
    p.add_argument('--labels', action='store_true', help='Default display is text-free')
    p.add_argument('--layout', choices=['order', 'stages'], default='order')
    p.add_argument('--scale', type=int, default=5)
    p.add_argument('--low', type=float, default=24.)
    p.add_argument('--high', type=float, default=38.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate and process a labeled synthetic recording')
    demo.add_argument('--output', type=Path, required=True)
    demo.add_argument('--frames', type=int, default=280)
    add_model(demo)
    add_display(demo)
    calibrate = sub.add_parser('calibrate', help='Fit a sensor-specific profile on a stationary interval')
    calibrate.add_argument('recording', type=Path)
    calibrate.add_argument('--factory', type=Path)
    calibrate.add_argument('--output', type=Path, required=True)
    calibrate.add_argument('--start', type=float, default=5.)
    calibrate.add_argument('--end', type=float, default=17.5)
    replay = sub.add_parser('replay', help='Process raw recorded inputs, ignoring cached derived results')
    replay.add_argument('recording', type=Path)
    replay.add_argument('--factory', type=Path)
    replay.add_argument('--profile', type=Path, required=True)
    replay.add_argument('--output', type=Path, required=True)
    add_model(replay)
    add_display(replay)
    for command in ('capture', 'live'):
        p = sub.add_parser(command, help='Raspberry Pi raw acquisition' if command == 'capture' else 'Raspberry Pi 2x3 live pipeline')
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--seconds', type=float, default=30. if command == 'capture' else 0.)
        p.add_argument('--lock-file', type=Path, default=Path('/tmp/mlx90640-thermal-pipeline.lock'))
        p.add_argument('--chunk-frames', type=int, default=256, help='Flush raw/derived data in bounded chunks')
        if command == 'live':
            p.add_argument('--profile', type=Path, required=True)
            add_model(p)
            add_display(p)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, 'low') and (not np.isfinite([args.low, args.high]).all()
                                   or args.high <= args.low or args.scale < 1):
            raise ValueError('Display requires finite low < high and scale >= 1')
        if args.command in ('live', 'capture'):
            from .capture import run_capture
            return run_capture(args)
        if args.command == 'calibrate':
            r = load_recording(args.recording, args.factory)
            profile = fit_profile(r, args.start, args.end)
            save_profile(args.output, profile, args.recording)
            print(json.dumps({'profile': str(args.output), 'fit_frames': profile[2]['fit_frames'], 'cutoff': profile[2]['cutoff']}))
            return 0
        predictor = predictor_from_args(args)
        if args.command == 'demo':
            from .demo import synthetic_recording
            args.output.mkdir(parents=True, exist_ok=False)
            static = synthetic_recording(180, moving=False)
            profile = fit_profile(static)
            save_profile(args.output/'synthetic_profile.npz', profile)
            r = synthetic_recording(args.frames)
            write_npz(args.output/'synthetic_raw.npz', r.arrays, r.metadata, r.factory)
            output = args.output/'processed.npz'
        else:
            r = load_recording(args.recording, args.factory)
            profile = load_profile(args.profile)
            output = args.output
        arrays, states = run_recording(r, profile, predictor, output)
        i = min(len(r)-1, len(r)//2)
        kwargs = dict(labels=args.labels, layout=args.layout, scale=args.scale, low=args.low, high=args.high)
        save_preview(output.with_suffix('.png'), {k: v[i] for k, v in arrays.items()}, **kwargs)
        print(json.dumps({'frames': len(r), 'output': str(output), 'model_kind': predictor.kind,
                          'synthetic': bool(r.metadata.get('synthetic', False)),
                          'repaired_samples': int(states['repair_pixel_count'].sum()),
                          'passthrough_frames': int(states['temporal_passthrough'].sum())}))
        if args.show:
            show_sequence(arrays, r.arrays['times_s'], **kwargs)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
