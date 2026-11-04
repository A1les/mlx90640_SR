# 真实录像动态对比

[项目首页](../README.md)提供三段内嵌视频。[视频发布页](https://github.com/A1les/mlx90640_SR/releases/tag/video-v0.1.0)提供相同内容的 MP4 下载、来源清单与 SHA256。

| 视频 | 完整时长 | 原始帧数 | 展示内容 | 下载 |
|---|---:|---:|---|---|
| 运动顺序对比 | 35.100 s | 275 | 上排 D → CNN → EMA；下排 D → EMA → CNN | [MP4 · 8.16 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/motion_pipeline.mp4) |
| 运动去交错对比 | 35.100 s | 275 | 左侧原始温度拼帧；右侧运动自适应频域去交错 | [MP4 · 4.87 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/motion_deinterlacing.mp4) |
| 静态顺序对比 | 30.067 s | 236 | 静止场景的同一 2×3 处理布局 | [MP4 · 5.29 MB](https://github.com/A1les/mlx90640_SR/releases/download/video-v0.1.0/static_pipeline.mp4) |

D 表示运动自适应去交错输出，EMA 表示运动自适应时间滤波。2×3 布局的两排共享去交错结果和 CNN 权重，各阶段输出在同一时刻对齐。温度阵列经固定 29.5–36.5°C 色标映射，以最近邻方式放大显示；每个面板没有单独调整对比度。

## 录像与播放时序

运动录像为 `motion_20261103T140119_035916Z.npz`，静态录像为 `static_20261103T034021_165175Z.npz`。两段录像的平均温度拼帧更新率分别为 7.8373 Hz 和 7.8520 Hz。本次使用已保存的真实传感器数据完整离线回放，不是新一轮现场采集。

MP4 使用 H.264、yuv420p 和 30 fps 容器帧率。播放器在两次实测更新之间保持上一帧，每个原始帧均被呈现，不生成中间运动或温度值，也不加速录像。显示更新时刻相对原始时间最多延后 1/30 秒。最后一帧保持时长取该录像采样间隔的中位数，再按容器帧率量化，因此视频时长略长于首末实测帧的时间跨度。画面中的 `t` 相对于录像开始时间，播放器的零时刻对应第一帧实测值。

三段 MP4 均通过完整解码检查，并核对编码格式、帧数与时长。回放期间逐帧验证去交错阶段的最新子页和修补掩膜外观测保持原始位值。视频为有损编码的显示结果，数值评价仍使用编码前的温度阵列。

## 下载文件与来源

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `motion_pipeline.mp4` | 8,156,117 | `3464bd40bdd927a87ba38fe1c10524387fc52197f0c857a75a2c2b3bc718758a` |
| `motion_deinterlacing.mp4` | 4,867,778 | `501804465426a996fa455e3ce9407ca34edfe4efd2f8d915106f9d0eb5e130a0` |
| `static_pipeline.mp4` | 5,293,093 | `c5006e50eca8664f237aa8f7e2148561a982a2674c64d85c5d821da09816af34` |

[机器可读清单](assets/videos.json)记录原始录像哈希、去交错输出阵列哈希、模型与标定文件哈希、面板顺序、显示参数和视频校验信息。[数值验证](validation.md)说明评价范围；这些视频用于观察动态表现，不能单独证明物理分辨率或绝对测温精度提升。原始设备录像与工厂标定不包含在公开下载中。

## 生成方法

生成脚本为 [tools/build_videos.py](../tools/build_videos.py)。使用现有 `train` 依赖，以及已安装的 FFmpeg 和 ffprobe；算法调用同一个 `ThermalPipeline`，视频编码不参与温度计算。以下命令使用占位的本地输入文件，需要提供兼容录像和与该设备匹配的标定。

```bash
python tools/build_videos.py \
  --motion captures/motion.npz \
  --static captures/static.npz \
  --factory local_assets/sensor_factory.json \
  --profile profiles/noise.npz \
  --checkpoint local_assets/thermal_sr_x2_epoch22.pt \
  --output outputs/videos
```

`--ffmpeg` 和 `--ffprobe` 可指定工具路径；可选 `--preview-output outputs/previews` 额外生成原速 GIF 片段。脚本拒绝覆盖已有 MP4 或 GIF。不同 FFmpeg 版本可能生成不同的编码字节，已发布文件以本页校验值为准。
