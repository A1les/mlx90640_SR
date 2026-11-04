# MLX90640 SR

**Motion-adaptive deinterlacing, temporal denoising and lightweight x2 CNN reconstruction for MLX90640 thermal images.**

The deinterlacer preserves latest-subpage observations and repairs selected older pixels in moving regions. A shared-input 2×3 viewer compares CNN-before-EMA and EMA-before-CNN processing.

[中文](README.md) · [Pretrained weights](https://github.com/A1les/mlx90640_SR/releases/tag/model-v0.1.0) · [Training data](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file) · [Usage](docs/usage.md) · [CI](https://github.com/A1les/mlx90640_SR/actions)

## Results

![Real MLX90640 motion: original assembled temperatures (left), adaptive deinterlacing (right)](docs/assets/motion_deinterlacing.gif)

Real hand-motion recording, 15.0–20.5 s at original speed. Both panels use a fixed 29.5–36.5°C scale and nearest-neighbor display enlargement. Latest-subpage and unselected observations remain unchanged.

![Shared-input CNN/EMA processing-order comparison](docs/assets/pipeline_comparison.png)

Top: deinterlacing → CNN → EMA. Bottom: deinterlacing → EMA → CNN. Same model, input and temperature scale. [Recording and rendering provenance](docs/assets/showcase.json).

## Measurements

On **11,742 held-out Lepton frames with synthetic degradation**, the matched-condition CNN reduces MAE by **27.31%** and edge error by **23.97%** relative to bicubic interpolation.

| Method | MAE (°C) ↓ | RMSE (°C) ↓ | Edge error (°C/pixel) ↓ |
|---|---:|---:|---:|
| Bilinear | 0.2310 | 0.3711 | 0.1978 |
| Bicubic | 0.1928 | 0.2989 | 0.1936 |
| **ThermalSRx2** | **0.1402** | **0.2119** | **0.1472** |

In a real static recording (196 frames, 5–30 s), mean per-pixel temporal standard deviation on the native grid decreases from **0.5843°C to 0.3688°C**. The deinterlacer changes **zero pixels** across all 236 frames. Final 64×48 variation is **0.3330°C** for CNN→EMA and **0.3369°C** for EMA→CNN.

[All three synthetic test conditions](docs/validation/synthetic_test_results.json) · [Real-recording statistics](docs/assets/showcase.json) · [Validation scope](docs/validation.md)

Synthetic reference error and temporal variation measure different properties. A 64×48 output is a reconstruction grid; independently verified physical resolution and absolute temperature accuracy improvements are not claimed.

## Data and pretrained model

**ThermalSRx2:** 16,732 parameters; five 3×3 convolutions; 24 hidden channels; PixelShuffle ×2; Celsius-domain residual reconstruction. Train/validation/test: **54,450 / 13,703 / 11,742** frames with disjoint subjects. AdamW, batch 128, initial learning rate 0.001, 24 epochs; **epoch 22** selected on validation data. Loss: temperature L1 + 0.2 × edge L1.

Training uses the Lepton modality of **MlGesture**, with synthetic LR/HR supervision. [Official dataset page](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/home) · [Official ZIP, 1.59 GB](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file) · [Dataset checksum](configs/dataset.json) · [Subject splits](configs/subject_splits.csv).

| Download | Size | Purpose |
|---|---:|---|
| [PyTorch checkpoint](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.pt) | 72,149 bytes | Original selected checkpoint |
| [ONNX model](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.onnx) | 73,060 bytes | Fixed `[1,1,24,32]` input; CPU/Pi inference |

[Model card](docs/model-card.md) · [Model SHA256](configs/pretrained.json) · [Training configuration](configs/thermal_sr_x2.json) · [Training code](src/mlx90640_pipeline/training) · [Reproduction](docs/training.md)

## Run the pretrained CNN

Python 3.10+. The downloader verifies both size and SHA256.

```bash
git clone https://github.com/A1les/mlx90640_SR.git
cd mlx90640_SR
python -m pip install '.[train]'
python tools/download_model.py --format pt --output local_assets/thermal_sr_x2_epoch22.pt
mlx90640-pipeline demo --checkpoint local_assets/thermal_sr_x2_epoch22.pt --output outputs/demo_cnn --show
```

This command uses synthetic demo inputs; the figures above use real recordings. See [usage](docs/usage.md), [Raspberry Pi capture](docs/raspberry-pi.md), [algorithms](docs/algorithms.md), and [offline technical documentation](docs/offline/index.html).

Project code and original documentation are [MIT licensed](LICENSE). Dataset provenance and citations are in [third-party notices](THIRD_PARTY_NOTICES.md).
