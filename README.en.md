# MLX90640 Thermal Pipeline

Temperature-domain motion-adaptive deinterlacing, causal temporal denoising, and lightweight x2 CNN reconstruction for the 32×24 MLX90640. The two CNN/EMA orders share native motion weights and are available in a text-free 2×3 viewer.

The active deinterlacer is `confirmed_frequency_repair_v3`. Only confirmed older-subpage locations can change. Latest-subpage observations and all unselected values remain byte-identical at the deinterlaced output. Subsequent EMA/CNN outputs are separately stored estimates.

## Quick start

```bash
python -m pip install .
mlx90640-pipeline demo --output outputs/demo
python -m unittest discover -s tests -v
```

The standalone demo uses synthetic data and an explicitly identified bilinear baseline. It is not a pretrained CNN demonstration or a physical accuracy experiment. Add `--checkpoint /path/to/best.pt` or `--onnx /path/to/model.onnx` to run supplied weights. Add `--show` for a desktop replay, `--labels` for annotations, or `--layout stages` for intermediate stages.

Optional dependency groups are `train`, `onnx`, `export`, and `pi`. Reuse an existing PyTorch environment when available. The application never installs dependencies automatically.

## Recorded data

```bash
mlx90640-pipeline calibrate captures/static.npz --start 5 --end 17.5 --output profiles/noise.npz
mlx90640-pipeline replay captures/motion.npz --profile profiles/noise.npz --checkpoint /path/to/best.pt --output outputs/motion.npz
```

The current calibrated repair requires 16 Hz chess subpages. A profile is bound to the sensor serial, EEPROM digest and resolution setting. The portable recording format retains original Celsius arrays, both 834-word subpage records, timestamps and conversion assumptions. Legacy recordings with a separate `sensor_factory.json` are supported.

## Model and training

`ThermalSRx2` contains five 3×3 convolutions with 24 hidden channels, four ReLUs and a 2× PixelShuffle. The mean-centered residual branch is added to a Celsius-domain bilinear skip. It has 16,732 parameters. Training uses Lepton images from MlGesture to form synthetic LR/HR supervision, with subject-disjoint splits. The selected historical run trained for 24 epochs and selected epoch 22 by validation MAE. Its effective loss is temperature L1 plus 0.2 times edge L1; consistency weight is zero.

Training images and pretrained weights are not embedded in the source distribution. See the [training instructions](docs/training.md), [model card](docs/model-card.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

## Evidence boundaries

A 64×48 output is a reconstruction grid, not proof of recovered physical resolution. Synthetic Lepton error, static temporal variation and backend parity measure different properties. No real registered MLX/high-resolution supervision or absolute thermal accuracy certification is claimed.

The packaged numerical path was checked against the legacy implementation on 511 real recorded frames using the same frozen PyTorch backend: all 14 compared arrays and all state flags were identical. The 98-frame noise calibration was reproduced. This is offline regression evidence. The portable live-capture wrapper has adapter tests; its new entry point has not yet undergone a fresh physical Pi acceptance run. See [validation](docs/validation.md).

## License

Project code and original documentation are MIT licensed. External data and dependencies retain their own terms. Patent-related implementation provenance is described in [third-party notices](THIRD_PARTY_NOTICES.md). See the [Chinese documentation](README.md) for the complete implementation and operation guide.
