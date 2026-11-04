# MLX90640 SR

**面向 MLX90640 的运动自适应去交错、时间降噪与轻量 CNN 二倍重建。**

保留最新子页观测，只修补运动区域的较旧像元；在同一输入上比较两种 CNN／时间降噪顺序。提供核心算法、树莓派展示程序、训练代码与预训练权重。

[English](README.en.md) · [完整视频下载](https://github.com/A1les/mlx90640_SR/releases/tag/video-v0.1.0) · [下载权重](https://github.com/A1les/mlx90640_SR/releases/tag/model-v0.1.0) · [训练数据](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file) · [运行说明](docs/usage.md) · [自动检查](https://github.com/A1les/mlx90640_SR/actions)

## 实际效果

### 运动视频：两种处理顺序 · 35 秒

上排：去交错 → CNN → 时间降噪。下排：去交错 → 时间降噪 → CNN。两排从同一个去交错结果开始，使用同一 CNN 权重，连续展示各阶段的运动表现。

https://github.com/user-attachments/assets/6d720682-5bd0-4d5d-a89c-c866f8283202

[下载运动对比 MP4 · 8.16 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/motion_pipeline.mp4)

### 运动视频：去交错前后 · 35 秒

左：原始温度拼帧。右：运动自适应频域去交错。最新子页和修补掩膜之外的观测保持原值。

https://github.com/user-attachments/assets/160e32c7-00c2-4a24-ab0e-1812c9d2fbe1

[下载去交错对比 MP4 · 4.87 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/motion_deinterlacing.mp4)

### 静态视频：时间降噪 · 30 秒

保持相同的 2×3 布局，观察静止手掌和背景的时间波动，以及两种处理顺序的输出。

https://github.com/user-attachments/assets/54fe28d2-9d0b-457b-8236-0dfc008d0682

[下载静态对比 MP4 · 5.29 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/static_pipeline.mp4)

三段视频均由真实 MLX90640 录像完整回放生成，按原速播放，保留约 7.8 Hz 的实际更新节奏；所有面板固定使用 29.5–36.5°C 色标和最近邻显示放大。[录像来源、时序与 SHA256](docs/videos.md) · [静帧对照](docs/assets/pipeline_comparison.png)

## 量化结果

CNN 在 **11,742 帧、独立受试者的 Lepton 合成测试集**上，匹配训练退化条件时的结果如下。MAE 相对双三次插值降低 **27.31%**，边缘误差降低 **23.97%**。

| 方法 | MAE（°C）↓ | RMSE（°C）↓ | 边缘误差（°C/pixel）↓ |
|---|---:|---:|---:|
| 双线性插值 | 0.2310 | 0.3711 | 0.1978 |
| 双三次插值 | 0.1928 | 0.2989 | 0.1936 |
| **ThermalSRx2** | **0.1402** | **0.2119** | **0.1472** |

真实静态录像的 5–30 秒区间共 196 帧，32×24 原生网格的平均逐像元时间标准差由 **0.5843°C 降至 0.3688°C**；完整 236 帧中，去交错模块修补像元数为 **0**。两种顺序的 64×48 最终输出时间标准差分别为 **0.3330°C**（先 CNN）和 **0.3369°C**（先时间降噪）。

[三种退化条件的完整测试数据](docs/validation/synthetic_test_results.json) · [真实录像统计](docs/assets/showcase.json) · [验证记录](docs/validation.md)

这些指标分别衡量合成参考误差和指定录像的时间波动。64×48 为重建网格，尚未通过独立真值实验确认物理分辨率或绝对测温精度提升。

## CNN、训练数据与权重

| 项目 | 当前模型 |
|---|---|
| 网络 | ThermalSRx2；5 层 3×3 卷积、24 个隐藏通道、PixelShuffle ×2、双线性残差连接 |
| 参数量／输入输出 | **16,732**；摄氏度 32×24 → 64×48 |
| 数据 | KU Leuven **MlGesture** 的 Lepton 温度图；合成 LR/HR 监督 |
| 训练／验证／测试 | **54,450 / 13,703 / 11,742 帧**；受试者互不重叠 |
| 训练配置 | AdamW，batch 128，初始学习率 0.001，24 轮；选中第 **22** 轮 |
| 损失 | 温度 L1 + 0.2 × 边缘 L1 |

**训练数据：**[官方页面](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/home) · [MlGesture.zip 官方直链，1.59 GB](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file) · [数据哈希与说明](configs/dataset.json) · [受试者划分](configs/subject_splits.csv)

| 权重下载 | 大小 | 用途 |
|---|---:|---|
| [thermal_sr_x2_epoch22.pt](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.pt) | 72,149 bytes | PyTorch 推理和评价，原始选中检查点 |
| [thermal_sr_x2_epoch22.onnx](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.onnx) | 73,060 bytes | 固定输入 `[1,1,24,32]`，树莓派 ONNX Runtime 推理 |

[模型卡](docs/model-card.md) · [训练超参数](configs/thermal_sr_x2.json) · [权重 SHA256](configs/pretrained.json) · [训练与导出代码](src/mlx90640_pipeline/training) · [训练复现](docs/training.md)

## 运行预训练模型

Python 3.10+。下载工具自动核验权重的大小和 SHA256；已有深度学习环境可直接复用。

```bash
git clone https://github.com/A1les/mlx90640_SR.git
cd mlx90640_SR
python -m pip install '.[train]'
python tools/download_model.py --format pt --output local_assets/thermal_sr_x2_epoch22.pt
mlx90640-pipeline demo --checkpoint local_assets/thermal_sr_x2_epoch22.pt --output outputs/demo_cnn --show
```

此命令使用合成输入演示预训练 CNN；上方成果图来自真实录像。[真实录像回放与命令行](docs/usage.md)、[树莓派采集](docs/raspberry-pi.md)、[算法细节](docs/algorithms.md)和[离线技术文档](docs/offline/index.html)分别提供操作与实现说明。

## 核心代码

| 模块 | 内容 |
|---|---|
| [core/repair.py](src/mlx90640_pipeline/core/repair.py) | 运动区域确认、观测保护与约束频域去交错 |
| [core/temporal.py](src/mlx90640_pipeline/core/temporal.py) | 运动自适应时间滤波 |
| [model.py](src/mlx90640_pipeline/model.py) | ThermalSRx2 网络 |
| [training/](src/mlx90640_pipeline/training) | 数据准备、训练、评价与 ONNX 导出 |
| [capture.py](src/mlx90640_pipeline/capture.py) / [viewer.py](src/mlx90640_pipeline/viewer.py) | 原始子页采集与 2×3 对照展示 |

项目代码和原创说明采用 [MIT](LICENSE)。数据由原作者分发，引用和来源见[第三方说明](THIRD_PARTY_NOTICES.md)。
