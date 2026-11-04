"""Build real-recording MP4 comparisons and native-cadence GIF previews.

Requires the existing train dependencies plus FFmpeg/ffprobe on PATH. Videos
hold the latest captured image until the next measurement: 30 fps is only the
container cadence, with no interpolated temperatures or motion synthesis.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from mlx90640_pipeline.core.motion import load_profile
from mlx90640_pipeline.inference import TorchPredictor
from mlx90640_pipeline.pipeline import ThermalPipeline
from mlx90640_pipeline.recording import load_recording
from mlx90640_pipeline.viewer import ORDER_PANELS

LOW, HIGH = 29.5, 36.5
PALETTE = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1]
LABELS = ['D | 32 x 24', 'CNN(D) | 64 x 48', 'EMA(CNN(D)) | 64 x 48',
          'Same D | 32 x 24', 'EMA(D) | 32 x 24', 'CNN(EMA(D)) | 64 x 48']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_frame(arrays, index, elapsed, scene, pair=False):
    columns, rows, width = (2, 1, 320) if pair else (3, 2, 256)
    height = width * 3 // 4
    margin, header, label, footer = 8, 30, 24, 38
    size = (columns * width + (columns + 1) * margin,
            header + rows * (height + label) + (rows + 1) * margin + footer)
    image = Image.new('P', size, 0)
    image.putpalette(PALETTE.ravel().tolist())
    draw = ImageDraw.Draw(image)
    draw.fontmode = '1'
    font_path = font_manager.findfont('DejaVu Sans')
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 13)
    draw.text((margin, 7), f'MLX90640 | {scene} | original-speed replay', font=font, fill=255)
    draw.text((size[0] - 111, 7), f't = {elapsed:5.2f} s', font=small, fill=235)
    keys = ('raw_temperature_c', 'deinterlaced_c') if pair else ORDER_PANELS
    labels = ['Raw subpages | 32 x 24', 'Deinterlaced | 32 x 24'] if pair else LABELS
    for panel, (key, title) in enumerate(zip(keys, labels)):
        x = margin + (panel % columns) * (width + margin)
        y = header + margin + (panel // columns) * (height + label + margin)
        draw.text((x + 2, y + 2), title, font=small, fill=255)
        values = arrays[key][index]
        indices = np.rint(np.clip((values - LOW) / (HIGH - LOW), 0, 1) * 255).astype(np.uint8)
        tile = Image.frombytes('P', (indices.shape[1], indices.shape[0]), indices.tobytes())
        tile.putpalette(PALETTE.ravel().tolist())
        image.paste(tile.resize((width, height), Image.Resampling.NEAREST), (x, y + label))
    bottom = size[1] - footer + 7
    for x in range(192):
        draw.line((margin + x, bottom, margin + x, bottom + 8), fill=int(round(x / 191 * 255)))
    draw.text((208, bottom - 2), f'{LOW:.1f} - {HIGH:.1f} C | shared scale', font=small, fill=255)
    if not pair:
        draw.text((margin, size[1] - 16), 'Top: CNN then EMA    Bottom: EMA then CNN', font=small, fill=235)
    return image


def mp4(path, frames, times, ffmpeg, ffprobe):
    if path.exists():
        raise FileExistsError(path)
    elapsed = times - times[0]
    last_hold = float(np.median(np.diff(times)))
    duration = float(elapsed[-1] + last_hold)
    fps = 30
    ticks = np.arange(math.ceil(duration * fps)) / fps
    indices = np.searchsorted(elapsed, ticks, side='right') - 1
    if len(np.unique(indices)) != len(frames):
        raise ValueError('30 Hz display cadence would skip a source frame')
    width, height = frames[0].size
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
               '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
               '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-movflags', '+faststart', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    previous, pixels = -1, None
    try:
        for index in indices:
            if index != previous:
                pixels = frames[index].convert('RGB').tobytes()
                previous = index
            process.stdin.write(pixels)
        process.stdin.close()
        error = process.stderr.read().decode('utf-8', errors='replace')
        if process.wait(timeout=90):
            raise RuntimeError(error)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                       '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,duration',
                       '-of', 'json', str(path)], text=True))['streams'][0]
    assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p'
    assert int(probe['nb_frames']) == len(indices)
    assert abs(float(probe['duration']) - len(indices) / fps) < .002
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
            'source_frames': len(frames), 'container_frames': len(indices),
            'source_span_s': float(elapsed[-1]), 'last_frame_hold_s': last_hold,
            'duration_s': float(probe['duration']), 'container_fps': fps,
            'display_rule': 'Hold last measured frame; no frame interpolation; every source frame included.',
            'maximum_presentation_delay_s': 1 / fps, 'codec': 'H.264', 'pixel_format': 'yuv420p',
            'size_px': [width, height], 'complete_decode_verified': True}


def gif(path, frames, elapsed, start, end):
    if path.exists():
        raise FileExistsError(path)
    selected = np.flatnonzero((elapsed >= start) & (elapsed < end))
    stops = np.append(elapsed[selected[1:]], end)
    cumulative = np.rint(np.cumsum(stops - elapsed[selected]) * 100).astype(int)
    durations = np.diff(np.r_[0, cumulative]) * 10
    images = [frames[i] for i in selected]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations.tolist(),
                   loop=0, optimize=False, disposal=2)
    with Image.open(path) as decoded:
        assert decoded.n_frames == len(selected)
        total_ms = 0
        for frame, original, expected_ms in zip(range(decoded.n_frames), images, durations):
            decoded.seek(frame)
            assert decoded.info['duration'] == expected_ms
            assert np.array_equal(np.asarray(decoded.convert('RGB')), np.asarray(original.convert('RGB')))
            total_ms += decoded.info['duration']
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
            'source_interval_s': [start, end], 'first_source_frame': int(selected[0]),
            'last_source_frame': int(selected[-1]), 'source_frames': len(selected),
            'duration_ms': total_ms, 'pixel_exact_decode_verified': True,
            'timing_quantization_ms': 10, 'playback_speed': 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'static', 'factory', 'profile', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preview-output', type=Path, help='Optional directory for GIF excerpts')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which(args.ffmpeg), shutil.which(args.ffprobe)
    if not ffmpeg or not ffprobe:
        raise RuntimeError('FFmpeg and ffprobe must already be installed')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview_output:
        args.preview_output.mkdir(parents=True, exist_ok=True)
    predictor = TorchPredictor(args.checkpoint)
    report = {'checkpoint_sha256': sha(args.checkpoint), 'profile_sha256': sha(args.profile),
              'color_limits_c': [LOW, HIGH], 'colormap': 'OpenCV INFERNO',
              'display_enlargement': 'nearest-neighbor', 'scope': 'Offline replay of real MLX recordings.',
              'order_panels': list(ORDER_PANELS), 'recordings': {}, 'videos': {}, 'previews': {}}
    for scene in ('motion', 'static'):
        path = getattr(args, scene)
        recording = load_recording(path, args.factory)
        pipeline = ThermalPipeline(load_profile(args.profile), recording.factory, predictor)
        arrays, states = pipeline.sequence(recording)
        report['recordings'][scene] = {'file': path.name, 'sha256': sha(path), 'frames': len(recording),
            'measured_mean_hz': float((len(recording) - 1) / np.ptp(recording.arrays['times_s'])),
            'first_displayed_timestamp_s': float(recording.elapsed[0]),
            'displayed_timestamp_origin': 'Recording start; video time zero is the first measurement.',
            'deinterlaced_array_sha256': hashlib.sha256(arrays['deinterlaced_c'].tobytes()).hexdigest(),
            'current_and_unselected_observation_contract_checked_every_frame': True}
        frames = [draw_frame(arrays, i, float(t), scene) for i, t in enumerate(recording.elapsed)]
        report['videos'][scene] = mp4(args.output / (scene + '_pipeline.mp4'), frames,
                                     recording.arrays['times_s'], ffmpeg, ffprobe)
        if args.preview_output:
            interval = (7.5, 20.5) if scene == 'motion' else (5., 13.)
            report['previews'][scene] = gif(args.preview_output / (scene + '_pipeline.gif'), frames,
                                            recording.elapsed, *interval)
        if scene == 'motion':
            frames = [draw_frame(arrays, i, float(t), 'deinterlacing', pair=True)
                      for i, t in enumerate(recording.elapsed)]
            report['videos']['deinterlacing'] = mp4(args.output / 'motion_deinterlacing.mp4', frames,
                                                    recording.arrays['times_s'], ffmpeg, ffprobe)
        print(json.dumps({'finished': scene, 'video': report['videos'][scene]}), flush=True)
    for directory in dict.fromkeys(p for p in (args.output, args.preview_output) if p is not None):
        with (directory / 'videos.json').open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
