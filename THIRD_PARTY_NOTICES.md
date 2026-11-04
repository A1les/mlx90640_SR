# 第三方来源与权利范围

## 项目代码

仓库内项目原创代码和说明按根目录 MIT 许可发布。MIT 许可不改变外部数据、第三方依赖、论文和专利的权利范围。本仓库不附带第三方训练图像、厂商闭源程序或完整论文副本。

## MlGesture

数据来源为 [KU Leuven EAVISE MlGesture](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/home)。官方页面要求引用 Vandersteegen, Reusen, Van Beeck and Goedemé, *Low-Latency Hand Gesture Recognition With a Low-Resolution Thermal Imager*, CVPR Workshops 2020，[论文入口](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Vandersteegen_Low-Latency_Hand_Gesture_Recognition_With_a_Low-Resolution_Thermal_Imager_CVPRW_2020_paper.html)。

第三方训练图像由原作者分发，本仓库提供[官方下载入口](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/mlgesture_dataset/@@download/file)、预处理程序、固定划分与哈希，不镜像原始数据。官方页面要求使用数据时引用上述论文；已读取页面和归档 README 未列出独立的数据再分发许可证。项目训练所得模型另见 [预训练权重发布](https://github.com/A1les/mlx90640_SR/releases/tag/model-v0.1.0)，包含参数与配置，不包含训练图像。项目 MIT 声明不扩展到原数据集。

## 频域去交错

频率域构造参考 [EP4192002A1](https://patents.google.com/patent/EP4192002A1/en)，尤其第 0027–0038 段及图 3、图 6。周期频率距离、归一化、直流保留、反射边界、宽度 0.22、强度 0.75 和仅选中旧像元的 16 次约束循环均在代码中明确实现。其中数值预设、部分强度和观测约束为项目选择。文献引用不代表专利许可，项目不对专利有效性或适用范围给出法律结论。

## 子页插值与网络结构

十字邻域核的研究出处为 [Chidurala 学位论文](https://digital.library.unt.edu/ark:/67531/metadc1987151/) 第 50 页式 3.5；边界有效邻居归一化以及最新子页的选取是项目实现。在活动 v3 流程中，该核主要构造运动检测残差，最终修补统一使用约束频域方法。

PixelShuffle 使用 [PyTorch 的公开算子](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)，相关原理可参考 [Shi et al., CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html)。ThermalSRx2 是本项目的轻量温度域结构，不标为原论文网络的完整复现。

## 外部软件依赖

Adafruit CircuitPython MLX90640 1.3.9 作为安装依赖使用；[上游许可](https://github.com/adafruit/Adafruit_CircuitPython_MLX90640/blob/1.3.9/LICENSE) 为 MIT。采集适配层调用父类温度换算，仅记录输入，不重新分发修改后的驱动。NumPy、OpenCV、PyTorch、Pillow、Matplotlib、ONNX、ONNX Runtime、ONNX Script、Blinka 均为外部依赖，其发行包包含各自许可。源码包不包含这些依赖的二进制文件。
