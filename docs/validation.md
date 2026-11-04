# 验证与证据范围

## 本次代码整理的数值回归

使用同一冻结 PyTorch 模型，逐帧比较整理前源文件与包内实现。比较覆盖去交错温度、三个掩膜/来源数组、两条处理顺序、两种网格权重、历史和区域旁路等 14 项数组，以及全部状态标志。275 帧运动录像和 236 帧静态录像全部逐字节相同；原 98 帧标定区间的两个噪声尺度数组和阈值也精确复现。

完整输入哈希、数组清单和结果见 [legacy_regression.json](validation/legacy_regression.json)。本次输入文件以这些哈希为准；不能将其他同名录像或经过截取的 235 帧记录混作同一个输入。

可使用自己的历史工程与录像重新执行：

```bash
python tools/compare_legacy.py --study /path/to/legacy/study --release /path/to/legacy/release --profile /path/to/noise.npz --recording /path/to/motion.npz --recording /path/to/static.npz --calibration-recording /path/to/static.npz --checkpoint /path/to/best.pt --output outputs/legacy_regression.json
```

也可用 `--onnx` 替代 `--checkpoint`，但必须安装对应后端。该检查隔离代码整理造成的变化，不是独立算法优劣或真实精度实验。

## 自动检查

检查涵盖最新子页和未选中观测的逐字节保留、错误标定拒绝、子页时间异常、帧间断和逆序重置、静态 EMA 噪声下降、场景突变旁路、两类子页邻域释放、直流场保持、无 pickle 录像往返和不覆盖保存、渲染不修改温度、模型参数量/初始化/平移性质/有限梯度，以及采集适配器对父类转换的保留。

命令为 `python -m unittest discover -s tests -v`。基础环境缺少 PyTorch 时相关模型测试明确跳过；发布核查在已有深度学习环境中运行全部测试。检查记录与数据预处理比对见本目录 JSON 报告。

| 检查 | 本次结果 | 记录 |
|---|---|---|
| 原版本回归 | 511 帧、14 项数组与全部状态一致 | [回归记录](validation/legacy_regression.json) |
| 全量数据准备 | 219,243 帧审查，缓存与划分内容一致 | [数据记录](validation/data_preparation.json) |
| 自动测试 | 13 项通过、0 跳过 | [环境与测试](validation/local_checks.json) |
| 训练与评价入口 | 合成小样本 1 轮训练及 3 种评价条件通过 | [功能检查](validation/training_smoke.json) |
| 安装包与 CLI | 隔离目录导入、冻结 CNN 演示及真实录像回放通过 | [打包检查](validation/package_checks.json) |

完整预处理比较保留了数值字节级检查；发布用受试者 CSV 规范为无 BOM 的 UTF-8 与 LF，文本字节不同，但逐条分配记录相同。单轮小样本训练只验证程序连接，不替代历史 24 轮训练与测试结果。ONNX 导出和新采集入口的实体设备验收不计入本次已通过项目。

2026-11-04，首次公开上传提交 `c4c1b34ecde022f1d1b752c018575bd54277bcfc` 的 GitHub Actions 基础检查和 CPU 模型检查均通过，见[首次远程运行记录](https://github.com/A1les/mlx90640_SR/actions/runs/41799421371)。其中包括基础演示、源码审查、单元测试和合成小样本训练与评价。`local_checks.json` 中的 `ci_status_at_local_packaging` 保留上传前的历史状态；后续提交的检查结果以[对应远程运行](https://github.com/A1les/mlx90640_SR/actions)为准。

## 历史实验与本次检查的区别

历史选中 CNN 的合成 matched 测试相对 bicubic 的 MAE 降低约 27.31%，边缘误差降低约 23.97%。该结果相对于 Lepton 合成参考，不能解释为 MLX 绝对测温精度。历史静态录像中两个顺序的最终平均时间标准差约为 0.2848°C 和 0.2819°C，属于该指定录像的时间稳定性统计。

源程序曾在树莓派进行实时展示，当前整理后的包级采集入口尚未做新一轮实体设备验收。已完成的离线回归、模型结构检查和模拟驱动检查分别报告，不合并为“全部硬件验收通过”。

## 尚未建立的结论

当前没有经过严格同步、配准、辐射标定的 MLX/高分辨率真值对；没有以 ESF/MTF、已知温度目标或黑体试验确认真实分辨率和测温提升；没有以多设备、室外与高温场景证明全场景泛化。整帧 α=1 可以证明该帧不混入 EMA 历史，但无法证明运动检测从不漏检，也无法消除传感器本身的时间混合。
