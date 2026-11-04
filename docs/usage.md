# 安装、下载模型与运行

源码仓库为 `A1les/mlx90640_SR`，Python 分发名为 `mlx90640-thermal-pipeline`，命令行入口为 `mlx90640-pipeline`。需要 Python 3.10+。

## 下载预训练权重

模型位于 [GitHub Release](https://github.com/A1les/mlx90640_SR/releases/tag/model-v0.1.0)。下载工具按 [pretrained.json](../configs/pretrained.json) 检查精确字节数和 SHA256，拒绝覆盖已有文件。

```bash
python -m pip install '.[train]'
python tools/download_model.py --format pt --output local_assets/thermal_sr_x2_epoch22.pt
mlx90640-pipeline demo --checkpoint local_assets/thermal_sr_x2_epoch22.pt --output outputs/demo_cnn --show
```

树莓派或 CPU 的 ONNX 推理可使用：

```bash
python -m pip install '.[onnx]'
python tools/download_model.py --format onnx --output local_assets/thermal_sr_x2_epoch22.onnx
mlx90640-pipeline demo --onnx local_assets/thermal_sr_x2_epoch22.onnx --output outputs/demo_onnx --show
```

代码不自动安装依赖。已有依赖时可用 `PYTHONPATH=src python -m mlx90640_pipeline ...` 直接运行。Windows PowerShell 可先设置 `$env:PYTHONPATH = (Resolve-Path -LiteralPath './src').Path`，再调用现有 Python 解释器。

未提供 `--checkpoint` 或 `--onnx` 时，演示使用明确标注的双线性基线。演示的原始输入为合成数据，不能作为实测结果或 DN 换算证据。

## 真实录像与标定

```bash
mlx90640-pipeline calibrate captures/static.npz --start 5 --end 17.5 --output profiles/noise.npz
mlx90640-pipeline replay captures/motion.npz --profile profiles/noise.npz --checkpoint local_assets/thermal_sr_x2_epoch22.pt --output outputs/motion_processed.npz --show
```

当前去交错配置接受 16 Hz 棋盘子页。标定绑定传感器序列号、EEPROM 哈希和分辨率设置。旧录像可用同目录 `sensor_factory.json` 或 `--factory` 提供工厂信息。原始录像、标定和个人设备信息不随模型发布，详见[数据格式](data-format.md)与[树莓派运行](raspberry-pi.md)。

## 展示布局

| 位置 | 左 | 中 | 右 |
|---|---|---|---|
| 上排 | 去交错 D | CNN(D) | EMA(CNN(D)) |
| 下排 | 同一个 D | EMA(D) | CNN(EMA(D)) |

默认 2×3 窗口无文字。`--labels` 添加标签，`--layout stages` 展示原始温度、单次频域结果、写回掩膜、最终去交错、原生时间滤波和 CNN 最终结果。两种顺序共享运动权重；显示使用共同色标且不改写温度阵列。

## 训练与检查

数据准备、24 轮训练和导出说明见[训练文档](training.md)。数值测试、发布资产校验和 PyTorch/ONNX 一致性检查分别为：

```bash
python -m unittest discover -s tests -v
python tools/audit_release.py
python tools/verify_model_release.py
```

最后一项需要 `train` 和 `onnx` 依赖，并下载约 145 KB 的模型文件。检查记录见[验证文档](validation.md)。
