"""Compare packaged kernels with a supplied pre-release source tree on real recordings.

No recordings or sensor profiles are downloaded or redistributed by this tool.
Run from an environment with the package available on PYTHONPATH or installed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from mlx90640_pipeline.calibration import fit_profile
from mlx90640_pipeline.core.motion import load_profile
from mlx90640_pipeline.core.spatial import paper_subframe
from mlx90640_pipeline.core.frequency import filter_temperature
from mlx90640_pipeline.core.repair import PATENT
from mlx90640_pipeline.inference import TorchPredictor, OnnxPredictor
from mlx90640_pipeline.pipeline import ThermalPipeline
from mlx90640_pipeline.recording import load_recording


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--recording', type=Path, action='append', required=True)
    p.add_argument('--calibration-recording', type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--onnx', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.study), str(args.release)]
    old_repair = load_module('legacy_single_repair', args.study/'single_repair.py')
    old_post = load_module('legacy_postprocess', args.study/'postprocess_pipeline.py')
    predictor = OnnxPredictor(args.onnx) if args.onnx else TorchPredictor(args.checkpoint)
    # Use the SAME inference backend and weights to isolate numerical refactoring.
    old_post.metadata = lambda: {'onnx_sha256': predictor.model_sha256}
    old_post.predict = lambda session, x: session.predict(x)
    profile = load_profile(args.profile)
    report = dict(inference_backend=predictor.kind, model_sha256=predictor.model_sha256,
                  scope='Same-input numerical regression; not physical ground-truth validation', recordings=[])
    for path in args.recording:
        recording = load_recording(path)
        new = ThermalPipeline(profile, recording.factory, predictor)
        old = old_repair.MotionRepair(profile)
        post = old_post.PostprocessPipeline(predictor, compare_order=True)
        a = recording.arrays
        max_difference, written, native_outputs = 0., 0, []
        keys = set()
        for i, t in enumerate(a['times_s']):
            raw, page = a['raw_temperature_c'][i], int(a['latest_subpage'][i])
            control, pair_ok = int(a['raw_subpages_u16'][i,-1,832]), bool(a['pair_ok'][i])
            expected, flags = old.update(raw, float(t), page, control,
                                         paper_subframe(raw,page), filter_temperature(raw,**PATENT), pair_ok)
            outputs, states = post.update(expected['deinterlaced_c'], expected['repair_eligible'],
                                           float(t), flags['repair_history_reset'])
            expected.update(outputs); flags.update(states)
            actual, actual_flags = new.update(raw, float(t), page, control, pair_ok)
            if flags != actual_flags:
                raise AssertionError(f'States differ at frame {i}')
            for key, expected_array in expected.items():
                if actual[key].dtype != expected_array.dtype or actual[key].tobytes() != expected_array.tobytes():
                    raise AssertionError(f'Numerical difference at frame {i}: {key}')
                keys.add(key)
            written += int(flags['repair_pixel_count'])
            native_outputs.append(actual['deinterlaced_c'])
        report['recordings'].append(dict(name=path.name, frames=len(recording),
                recording_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checked_array_keys=sorted(keys), byte_identical=True, states_identical=True,
                repaired_samples=written, maximum_absolute_difference_c=max_difference,
                native_output_sha256=hashlib.sha256(np.stack(native_outputs).tobytes()).hexdigest()))
    if args.calibration_recording:
        fitted = fit_profile(load_recording(args.calibration_recording), 5., 17.5)
        if (not np.array_equal(fitted[0], profile[0]) or not np.array_equal(fitted[1], profile[1])
                or fitted[2]['cutoff'] != profile[2]['cutoff']):
            raise AssertionError('Re-fitted noise profile differs from the frozen original')
        report['calibration_reproduced'] = True
        report['calibration_frames'] = fitted[2]['fit_frames']
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
