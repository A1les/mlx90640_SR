# ThermalSRx2 模型卡

## 身份

历史选中模型为 `seed20261101_centered_c0_e24`。模型从 24 轮训练中按验证集结果选择第 22 轮，参数量为 16,732。源码在 `src/mlx90640_pipeline/model.py`，配置在 `configs/thermal_sr_x2.json`。

| 文件 | SHA256 |
|---|---|
| `best.pt` | `e20cdc8ab3d5d50b6a68dc4bc6119b164e1de57348a3eaf2d2d8fd03330274be` |
| `selected_model.onnx` | `1f8b16619252d806a9511661eb985bb1eb4b63c73467fca8746ee823d79cc617` |

预训练权重通过 [GitHub Release](https://github.com/A1les/mlx90640_SR/releases/tag/model-v0.1.0) 独立提供，不嵌入 Git 源码。下载 [PyTorch 检查点](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.pt)或[ONNX 推理模型](https://github.com/A1les/mlx90640_SR/releases/download/model-v0.1.0/thermal_sr_x2_epoch22.onnx)，也可使用 `tools/download_model.py` 自动验证大小和 SHA256。完整下载清单见 [pretrained.json](../configs/pretrained.json)。

上表记录历史原件哈希。公开 PyTorch 文件与历史选中检查点逐字节一致。公开 ONNX 删除了节点调试元数据中的工作站路径，计算图与张量字段字节保持不变；公开文件 SHA256 为 `19a880fae99bfaf1d3dcdf4d670244ef3d8b9cf8fa8950a555adecca0c4ca265`，大小为 73,060 bytes。`tools/verify_model_release.py` 对公开下载文件执行 SHA256、输入输出形状及 16 个输入的 CPU PyTorch/ONNX 一致性检查。

## 输入、输出与结构

输入为摄氏度 float32 `[N,1,24,32]`，输出为估计摄氏度 `[N,1,48,64]`。ONNX 发行模型的 N 固定为 1。输入布局为原始行列排列，不旋转、不翻转、不做颜色归一化。

卷积通道为 1→24→24→24→24→4，全部 3×3、stride 1、padding 1，前四层后接 ReLU。输出通过 PixelShuffle(2) 重排。网络学习 `(T−mean(T))/10` 上的残差，并将残差乘 10 后加到原温度的双线性插值上。末层零初始化。该结构没有时序输入，过去帧只在独立 EMA 中使用。

## 数据与目标

训练来源为 MlGesture 的 Lepton 温度数据，54,450/13,703/11,742 帧分别用于训练/验证/测试，参与者互不重叠。训练目标为原生 Lepton 图按面积重采样至 64×48 的参考温度图，输入由模拟退化产生。没有真实配准 MLX90640/高分辨率监督，没有利用当前相机的运动录像进行 CNN 配对训练。

损失权重为温度 1、边缘 0.2、一致性 0。历史验证 MAE 为约 0.1426°C。matched 合成测试相对 bicubic 的 MAE 降低约 27.31%，边缘误差降低约 23.97%。这些值描述相对合成参考的误差，不是绝对测温误差。

## 使用范围

该模型用于研究轻量温度图重建和算法顺序。训练场景以车内近室温手部为主，无法仅凭均值中心化推断它对室外、火灾、高温、远距离微小目标均有效。模型不保证热点峰值、细小间隙或温度极值保持，也不对输入观测施加逐像元硬约束。64×48 输出需要通过独立目标、配准、噪声控制和空间分辨率测试才能获得物理性能解释。

## 验证与复现

源码迁移后模型结构、参数数量、初始化行为、温度平移性质与反向传播均有自动检查。真实录像回归使用同一个冻结 PyTorch 模型隔离代码整理造成的差异。历史树莓派 ONNX 一致性记录与本次包级回归是不同验证任务；新导出文件应重新执行数值校验和目标设备性能测试。
