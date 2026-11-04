# 本地发布与 GitHub 上传

源码目标仓库为 [A1les/mlx90640_SR](https://github.com/A1les/mlx90640_SR)，项目代码采用 MIT。Python 分发名称保持为 `mlx90640-thermal-pipeline`，导入路径为 `mlx90640_pipeline`。本地提交和打包不代表远程上传成功；上传状态应以远程分支和文件清单核验为准。

## 内容边界

源码 ZIP 包含包代码、训练程序、冻结配置、划分、测试、CI、文档、离线说明和公开验证摘要。不含环境、依赖二进制、训练集、现场录像、模型权重、设备 EEPROM、设备标定、个人 SSH 信息和历史工作目录。

作者的已有模型和设备配置可保存在 `local_assets/`；该目录明确被 Git 和发布工具排除。公共 ZIP 由显式文件清单生成，并附完整 SHA256 清单。不要把整个原研究工作区直接拖入 GitHub。

## 上传步骤

目标空仓库 `A1les/mlx90640_SR` 已由维护者创建。进入本地源码仓库后，确认文件清单与差异；首次配置时将 `origin` 指向该仓库：

```bash
git status --short
python tools/audit_release.py
git add .
git commit -m "Prepare MLX90640 thermal pipeline source release"
git remote add origin https://github.com/A1les/mlx90640_SR.git
git push -u origin main
```

如果本地已存在提交，无需再次创建相同提交；如果 `origin` 已配置为上述地址，也无需重复添加。用 `git remote -v` 核实目标，再执行 `git push -u origin main`。Git 凭据由维护者已有 GitHub 登录方式管理，项目不保存令牌或 SSH 私钥。

上传后检查远程 `main` 的提交、文件清单和 [GitHub Actions](https://github.com/A1les/mlx90640_SR/actions) 结果。基础与 CPU 模型工作流均成功后，才可报告该次远程检查通过；远程检查不包含实体传感器验收。

## 自检和打包

```bash
python -m unittest discover -s tests -v
python tools/audit_release.py
python tools/package_release.py --output /path/to/new-release-directory
```

测试通过后再执行打包。公共源码包中的 SHA256 清单描述包内文件内容，不把自身包含在自身哈希中。脚本拒绝向已有输出目录写入，避免覆盖上一版交付。

## 许可与模型

MIT 许可覆盖项目代码和原创说明。第三方资料仅通过引用和安装依赖使用。公开数据和预训练权重若另行发布，需要针对相应数据使用条件单独核实，不应直接把代码 MIT 声明扩展到整个原始数据集。
