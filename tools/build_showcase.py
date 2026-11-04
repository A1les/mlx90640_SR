"""Render a real-recording comparison and report its exact data provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mlx90640_pipeline.core.motion import load_profile
from mlx90640_pipeline.inference import TorchPredictor
from mlx90640_pipeline.pipeline import ThermalPipeline
from mlx90640_pipeline.recording import load_recording
from mlx90640_pipeline.viewer import ORDER_PANELS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(path, args, predictor):
    recording = load_recording(path, args.factory)
    pipeline = ThermalPipeline(load_profile(args.profile), recording.factory, predictor)
    arrays, states = pipeline.sequence(recording)
    return recording, arrays, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'static', 'factory', 'profile', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frame-time', type=float, default=8.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictor = TorchPredictor(args.checkpoint)
    recording, arrays, states = replay(args.motion, args, predictor)
    index = int(np.argmin(np.abs(recording.elapsed - args.frame_time)))
    low, high = 29.5, 36.5
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.15), layout='constrained')
    fig.set_facecolor('#ffffff')
    titles = ['Deinterlaced D · 32×24', 'CNN(D) · 64×48', 'EMA(CNN(D)) · 64×48',
              'Same deinterlaced D · 32×24', 'EMA(D) · 32×24', 'CNN(EMA(D)) · 64×48']
    for ax, key, title in zip(axes.flat, ORDER_PANELS, titles):
        im = ax.imshow(arrays[key][index], cmap='inferno', vmin=low, vmax=high,
                       interpolation='nearest', extent=(0, 32, 24, 0))
        ax.set_title(title, fontsize=11, loc='left', pad=8, color='#172e38')
        ax.set_axis_off()
    fig.suptitle('Real MLX90640 recording · identical input, weights and temperature scale',
                 fontsize=13, x=.015, ha='left', color='#172e38')
    bar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=.045, pad=.035)
    bar.set_label('Temperature / reconstructed temperature (°C)', fontsize=10)
    image = args.output / 'pipeline_comparison.png'
    fig.savefig(image, dpi=140, metadata={'Title': 'MLX90640 processing-order comparison'})
    plt.close(fig)
    static, filtered, static_states = replay(args.static, args, predictor)
    selected = (static.elapsed >= 5.) & (static.elapsed < 30.)
    statistics = {key: float(np.std(filtered[key][selected], axis=0, ddof=0).mean())
                  for key in ['raw_temperature_c', 'deinterlaced_c', 'native_temporal_c',
                              'sr_temperature_c', 'final_temperature_c', 'pre_cnn_temperature_c']}
    report = {
        'scope': 'Real recorded MLX temperature arrays, replayed offline; no high-resolution ground truth.',
        'checkpoint_sha256': digest(args.checkpoint),
        'noise_profile_sha256': digest(args.profile),
        'motion': {'file': args.motion.name, 'sha256': digest(args.motion), 'frames': len(recording),
                   'preview_frame_zero_based': index, 'preview_elapsed_s': float(recording.elapsed[index]),
                   'deinterlaced_array_sha256': hashlib.sha256(arrays['deinterlaced_c'].tobytes()).hexdigest(),
                   'latest_observation_contract_checked_every_frame': True,
                   'repair_samples': int(arrays['repair_mask'].sum())},
        'static': {'file': args.static.name, 'sha256': digest(args.static), 'frames': len(static),
                   'analysis_interval_s': [5., 30.], 'analysis_frames': int(selected.sum()),
                   'statistic': 'Mean per-pixel temporal population standard deviation; ddof=0.',
                   'mean_temporal_std_c': statistics,
                   'repaired_samples_entire_recording': int(filtered['repair_mask'].sum())},
        'display': {'color_limits_c': [low, high], 'colormap': 'inferno',
                    'enlargement': 'nearest-neighbor, same physical extent per panel',
                    'panel_order': list(ORDER_PANELS)},
        'image': {'file': image.name, 'sha256': digest(image)}
    }
    (args.output / 'showcase.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'preview_frame': index, 'static': report['static']}, indent=2))


if __name__ == '__main__':
    main()
