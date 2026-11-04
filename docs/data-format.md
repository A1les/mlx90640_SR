# 温度录像格式与来源追踪

## NPZ 容器

读取使用 `allow_pickle=False`，禁止对象数组。温度数组始终保留摄氏度数值，显示用的 uint8 色彩图不作为数值输入。保存使用同目录临时文件、fsync 和原子硬链接发布，不覆盖已有目标。输出目录所在文件系统需要支持同目录硬链接，常用 ext4 和 NTFS 支持该操作；不支持时保存会明确失败。

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `raw_temperature_c` | N×24×32，float64 或 float32 | 未经本项目插值/滤波的换算温度；采集入口为 float64 |
| `times_s` | N，浮点 | 进程单调时钟，秒；用于帧间间隔，不是 UTC epoch |
| `raw_subpages_u16` | N×2×834，uint16 | 每次转换的 832 RAM 字、控制寄存器和子页号 |
| `latest_subpage` | N，整数 0/1 | 第二次转换对应的最新子页 |
| `conversion_json` | N，Unicode | 两次转换的发射率、反射温度、芯片温度、电源电压、读取 UTC 等 |
| `acquired_at_utc` | N，Unicode，可选 | 完整帧取得后的 UTC 时间 |
| `factory_json` | 标量 Unicode | 设备身份、EEPROM 哈希及工厂信息 |
| `metadata_json` | 标量 Unicode | 格式、单位、算法配置、合成/真实标记、会话起点 |

旧格式可以没有 `factory_json`，但必须提供旁置的 `sensor_factory.json`。原始两行 RAM 不能被当成两张完整同步热图；只有对应子页位置属于该次更新。读取时间是主机观察到的转换/读取时间，不是已标定的模拟积分中心。

## 派生数组

`deinterlaced_c` 保留原 dtype；其余 CNN/EMA 温度通常为 float32。`repair_proposed`、`repair_eligible`、`repair_mask` 分别表示候选、确认可修补位置和实际发生改写的位置。`repair_source` 为 0（保留测量）或 3（约束频域估计）。

`sr_temperature_c` 是 CNN(D)；`final_temperature_c` 是 EMA(CNN(D))；`native_temporal_c` 是 EMA(D)；`pre_cnn_temperature_c` 是 CNN(EMA(D))，此历史字段名表示“CNN 之前做时间滤波”的最终结果，不是 CNN 输入。`temporal_native_alpha` 为 24×32 权重，`temporal_alpha` 为复制后的 48×64 权重。两条顺序各自保存更新前历史，便于验证加权公式。

## 分块与回放

现场采集默认每 256 帧保存一块，避免无限累积内存；记录包括完整原始数据和启用时的派生结果。跨块实时滤波历史连续，但单独回放一块会从首帧重新建立历史。因此不能要求任意截取文件的首帧派生结果等于原实时状态。每块元数据明确标记这一点。

`replay` 总是从原始温度重算，不信任缓存中的去交错或 CNN 数组。真实输入必须同时具备两个合法子页、16 Hz 棋盘配置和设备信息。子页间隔不合法时保留原数据并禁用该帧修补。格式错误、非有限温度和不支持的模式会报错。

## 公开范围

原始录像可能含现场内容和设备身份，EEPROM 也属于设备专有标定数据。本仓库不将这些文件放入 Git 或公共源码 ZIP。示例使用确定性的合成数据；其 `synthetic=True`，RAM 数值为零占位，不能用来推导真实温度换算或宣称采集精度。
