# 第三方来源与权利范围

## 项目代码

仓库内项目原创代码和说明按根目录 MIT 许可发布。MIT 许可不改变外部数据、第三方依赖、论文和专利的权利范围。本仓库不附带第三方训练图像、厂商闭源程序或完整论文副本。

## MlGesture

数据来源为 [KU Leuven EAVISE MlGesture](https://iiw.kuleuven.be/onderzoek/eavise/mlgesture/home)。官方页面要求引用 Vandersteegen, Reusen, Van Beeck and Goedemé, *Low-Latency Hand Gesture Recognition With a Low-Resolution Thermal Imager*, CVPR Workshops 2020，[论文入口](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Vandersteegen_Low-Latency_Hand_Gesture_Recognition_With_a_Low-Resolution_Thermal_Imager_CVPRW_2020_paper.html)。

截至本次整理，已读取的官方页面和数据 README 未提供可直接核实的独立数据再分发许可。因此源码发行物只提供获取入口、预处理程序、划分和哈希，不重新分发数据。预训练权重也从公开源码发行物中分离，发布权重前应单独核对原数据的使用条件。该说明记录本次核查范围，不对作者未公开的授权作推断。

## 频域去交错

频率域构造参考 [EP4192002A1](https://patents.google.com/patent/EP4192002A1/en)，尤其第 0027–0038 段及图 3、图 6。周期频率距离、归一化、直流保留、反射边界、宽度 0.22、强度 0.75 和仅选中旧像元的 16 次约束循环均在代码中明确实现。其中数值预设、部分强度和观测约束为项目选择。文献引用不代表专利许可，项目不对专利有效性或适用范围给出法律结论。

## 子页插值与网络结构

十字邻域核的研究出处为 [Chidurala 学位论文](https://digital.library.unt.edu/ark:/67531/metadc1987151/) 第 50 页式 3.5；边界有效邻居归一化以及最新子页的选取是项目实现。在活动 v3 流程中，该核主要构造运动检测残差，最终修补统一使用约束频域方法。

PixelShuffle 使用 [PyTorch 的公开算子](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)，相关原理可参考 [Shi et al., CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html)。ThermalSRx2 是本项目的轻量温度域结构，不标为原论文网络的完整复现。

## 外部软件依赖

Adafruit CircuitPython MLX90640 1.3.9 作为安装依赖使用；[上游许可](https://github.com/adafruit/Adafruit_CircuitPython_MLX90640/blob/1.3.9/LICENSE) 为 MIT。采集适配层调用父类温度换算，仅记录输入，不重新分发修改后的驱动。NumPy、OpenCV、PyTorch、Pillow、Matplotlib、ONNX、ONNX Runtime、ONNX Script、Blinka 均为外部依赖，其发行包包含各自许可。源码包不包含这些依赖的二进制文件。
