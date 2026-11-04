# 树莓派安装与运行

## 适用环境

目标为具有 I2C 的 Raspberry Pi OS 64 位系统和 MLX90640。数值算法固定使用 16 Hz 子页设置、棋盘模式，理想完整帧率上限约 8 FPS。历史装置在 400 kHz I2C 下约 7.85 完整 FPS，这是已有部署记录，不是对所有设备的保证。

运行前在系统设置启用 I2C，确认用户有设备访问权限。总线速度由操作系统配置，不能依赖 Linux Blinka 的 Python `frequency` 参数改变总线。历史配置为 `dtparam=i2c_arm=on,i2c_arm_baudrate=400000`；是否适合当前布线、电容与系统需自行验证。本程序不修改启动配置、不重启系统。

```bash
python -m pip install '.[pi]'
```

采集适配器固定要求 `adafruit-circuitpython-mlx90640==1.3.9`，因为它记录该版本私有转换接口的输入。驱动升级需要重新验证。现有环境已满足依赖时直接使用，不必重复安装。

## 先记录静态场景并标定

```bash
mlx90640-pipeline capture --seconds 30 --output captures/static_session
mlx90640-pipeline calibrate captures/static_session/capture_0000.npz --start 5 --end 17.5 --output profiles/device_noise.npz
```

保持相机与场景静止，避免目标在标定区间变化。工厂标定和转换参数随录像保存，噪声配置绑定当前设备。代码不默认使用作者设备的配置。当前入口对工厂标记坏点的设备明确报错，没有悄悄插值填补测量值。

## 打开完整对照

```bash
mlx90640-pipeline live --profile profiles/device_noise.npz --onnx /path/to/selected_model.onnx --output captures/live_session --show
```

默认是无文字 2×3 顺序对照。`--layout stages` 改为六阶段视图；`--labels` 添加标签。`Q`/Esc 退出、`T` 切换文字、`V` 切换布局、`S` 保存截图。温度色标默认 24–38°C，可用 `--low`、`--high` 调整；该操作不截断保存的温度值。

此公开入口持续分块记录，`--seconds 35` 可指定一次运动录像长度；没有复制旧展示器的 M/B 按键状态机。推荐运动验证流程为静止 5 秒、慢移 10 秒、快移 10 秒、移出并静止 10 秒。截图对应当时画面，原始数据已在当前分块记录中。

## 单设备读取与资源清理

开始前关闭其他 MLX90640 采集进程。入口使用合作锁，并检查可见进程是否已打开 `/dev/i2c-1`；它不会自动停止其他程序。锁检测受进程可见性和权限限制，不能替代操作层面的单读者要求。`--lock-file` 可与现有项目统一合作锁路径。

每次读取有 5 秒超时。错误读取不进入处理历史，累计 5 次错误后停止，不循环重置设备。退出时保存完整帧并释放 I2C 资源。信号处理和 GUI 需要从主线程运行；无桌面时省略 `--show`。

## 本次验证边界

当前发布包数值算法已在真实录像上与原版本核对，采集适配器通过模拟驱动测试，温度换算继续委托固定版本 Adafruit 驱动。**新公开入口尚未在实体树莓派上完成一次独立现场验收**；既有窗口保持原样运行。本地发布整理不应被记作新的硬件性能实验。
