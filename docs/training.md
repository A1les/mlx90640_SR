# 数据准备、训练、评价与导出

## 数据来源与划分

从 [MlGesture 官方入口](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/home)或[官方 ZIP 直链](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file)取得归档，并遵守发布方使用条件。直链已于 2026-11-04 验证为可读取的 ZIP，大小为 1,591,541,835 bytes。官方页面的旧 `at_download/file` 链接当前返回 404，实际可用的是 `@@download/file`。仓库不内嵌训练图像。当前预处理针对已审查的 `MlGesture.zip` 版本，SHA256 为 `36020b89e29312ec3fed46b5ea85b9a407a538ba3f8387fea8f4dd153c0c93d8`，机器可读记录见 [dataset.json](../configs/dataset.json)。

程序审查 `train_val/frame_data` 中两个视角的 Lepton 和 MLX90640 TIFF。两者分别按 signed int16 /100、/10 转为摄氏度。检查形状、数值有效性和跨集合完全重复帧；边缘、端点及孤立跳变仅作为诊断标记，不任意删除真实温度结构。Lepton 160×120 按面积重采样为 64×48 参考目标；MLX90640 仅用于原生输入审查和无配对推理。

| 集合 | 受试者 | Lepton 帧数 |
|---|---|---:|
| 训练 | 1,2,4,5,6,7,8,11,12,13,14,15,18,20,22,23 | 54,450 |
| 验证 | 10,17,21,24 | 13,703 |
| 测试 | 3,9,16,19 | 11,742 |

这些是项目预先固定的受试者划分，并非将官方手势分类训练标签直接用于超分。源归档中 139,348 帧 MLX90640 的 51 帧未通过宽范围数值检查；这些帧没有进入监督训练。数据中不存在已验证的真实 MLX/Lepton 配准真值关系。

## 准备缓存

```bash
python -m mlx90640_pipeline.training.prepare --archive /path/to/MlGesture.zip --splits configs/subject_splits.csv --output data/mlgesture
```

输出 Lepton 参考缓存、MLX 原生缓存、逐帧索引、可用性掩膜、固定划分和完整 manifest。原始归档不修改。目标目录必须为空或尚不存在，程序不会覆盖既有缓存。缓存包含约 1.4 GB 数值数组，应预留额外空间。

缓存生成时间和路径表示属于 manifest 身份的一部分，因此重新生成的 manifest 哈希通常不同。新训练权重绑定此次 manifest；使用历史权重评价历史数据时需保留与其匹配的原 manifest，不能简单改写校验值来绕过验证。不同缓存内容是否相同应另外比较数组和索引。

## 监督样本与损失

对参考温度 H，使用 5×5 高斯模糊、2×2 平均降采样、高斯随机噪声和 0.1°C 量化得到 L。训练模糊标准差在 0、0.5、1.0 中随机选取，噪声标准差从 0–0.2°C 均匀选取。模糊参数和噪声幅值按批次抽取，噪声样本逐元素独立；水平翻转也按批次以 0.5 概率执行。0.1°C 是公开数据的存储步长，不能用作传感器精度声明。

损失为温度 L1 + 0.2×水平和垂直温差的平均 L1。代码计算并记录降采样一致性误差，但选中配置的权重为 0。实际训练没有对 CNN 输出施加原始 MLX 观测的硬约束。当前模糊/噪声模型为模拟假设，尚未辨识当前设备的真实 PSF 和噪声分布。

## 训练与评价

```bash
python -m mlx90640_pipeline.training.train --data data/mlgesture --config configs/thermal_sr_x2.json --run runs/thermal_sr_x2 --device cuda
python -m mlx90640_pipeline.training.evaluate --data data/mlgesture --checkpoint runs/thermal_sr_x2/best.pt --output outputs/test --device cuda
```

CPU 运行改为 `--device cpu`。训练使用 AdamW，初始学习率 0.001，余弦下降至 0.00005，权重衰减 1e−5，批次 128，梯度范数裁剪 1.0，共 24 轮。每轮遍历全部训练样本；按验证 MAE 保存 `best.pt`，末轮状态保存为 `last.pt`。继续同配置同数据训练可使用 `--resume`。

验证采用固定种子、模糊 0.5 和噪声 0.1°C。测试分别使用 matched(0.5,0.1)、stronger_blur_noise(1.0,0.2)、outside_training_degradation(1.5,0.3)。评价输出逐帧和按受试者/视角/手势分组统计。总体统计按帧加权，不能视作各受试者等权。测试用于最终评价，不参与梯度更新。

历史选中模型在第 22 轮达到最低验证 MAE，虽然训练计划为 24 轮。新训练受运行环境和算子版本影响，不承诺重新生成相同权重字节。`run_metadata.json` 记录配置、源文件哈希、数据 manifest 哈希和设备；`history.jsonl` 记录每轮指标。

## ONNX 导出

```bash
python -m pip install '.[export]'
python -m mlx90640_pipeline.training.export --checkpoint runs/thermal_sr_x2/best.pt --output outputs/thermal_sr_x2.onnx
```

导出固定形状 float32 摄氏度输入 `[1,1,24,32]` 和估计摄氏度输出 `[1,1,48,64]`，opset 18。导出工具执行模型检查和 16 个常量/随机输入的 PyTorch–ONNX 一致性验证，最大差异阈值为 0.001°C。导出成功不等于物理精度通过验证。

本次整理复核了原冻结权重和训练源代码，执行了数据准备与训练/评价的小样本功能检查；没有为发布整理重新完成一轮 24-epoch 模型选择。具体执行结果见验证目录。
