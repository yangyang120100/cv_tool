# CV Tool 本地网页工作台

本版本把原始 CV 工具代码集中到一个本地 Web 页面中，提供按类别选择工具、参数表单、执行状态、日志、结果表格/CSV 下载等功能。应用使用 Streamlit，默认只监听 `127.0.0.1`，不需要数据库。

## 1. Windows 启动方式

建议使用 Python 3.10–3.12，并在项目目录创建独立虚拟环境：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

安装基础依赖：

```bat
python -m pip install -r requirements-web.txt
```

需要启用数据增强、YOLO 模型分析、COCO RLE、Kaggle 下载等高级功能时，再安装可选依赖：

```bat
python -m pip install -r requirements-optional.txt
```

如需一次性安装完整功能依赖，可直接执行：

```bat
python -m pip install -r requirements-web.txt -r requirements-optional.txt
```

然后双击 `start_web.bat`，或在项目目录运行：

```bat
python -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
```

浏览器访问：<http://127.0.0.1:8501>

## 2. 页面工具类别

网页菜单包括约 56 项工具入口，涵盖以下能力：

- **图像与视频**：视频抽帧、图像拼接、等比例缩放、Mask 伪彩色、按 Mask 类别提取原图区域、YOLO/LabelMe 标注可视化、OBB 可视化、双模型 OBB 对比、文件分类、语义分割与 OBB 数据增强。
- **Mask 与分割**：LabelMe JSON 生成 Mask、Mask 转 LabelMe JSON、Mask 像素值映射、Mask 类别统计、二值 Mask 处理、图像与 Mask 打包为 NPZ。
- **标注转换与检查**：LabelMe 与 YOLO TXT 互转、COCO 转 LabelMe、LabelMe OBB 转 COCO/YOLO HBB、像素框 TXT 转 YOLO HBB、VOC XML 转 YOLO/COCO、YOLO TXT 类别映射/筛选、COCO 无效图像清理、JSON 类别处理、YOLO 数据集转 COCO/划分、Panoptic 生成与完整性检查、实例 Mask 转 COCO Instances。
- **文件与数据集管理**：扩展名/名称重命名、删除文件名子串、按末字段分组、无对应文件清理、中文文件名处理、清除 TXT 首行、追加文件名前后缀、按类型分类、同名图像拼接、LabelMe JSON 类别映射与 `imagePath` 修复。
- **原始脚本目录**：列出仓库内 Python 源码。工具目录页可以下载 CSV 索引。低层辅助函数、测试脚本和实验脚本仍保留在项目中，但不会被网页无条件执行。

## 3. 路径与输出

- 表单路径由运行网页的本机 Python 进程访问；请填写运行服务这台电脑上的路径，例如 `D:\datasets\images`。
- 大多数转换功能写入表单指定的输出目录，不修改原输入数据。
- 多数目录工具支持递归子目录；未开启递归时通常只处理当前目录一级文件。
- 对删除、移动、重命名、原地写回等操作，先查看预览，再勾选确认。执行前建议备份重要数据。
- 图像路径带逗号时，不适合用于“图像拼接”单项工具的逗号分隔输入；此时可以使用“拼接图像目录文件”。

## 4. 注意事项

- 应用默认仅监听本机回环地址，不要把 Streamlit 服务暴露到公网。
- 高级工具的依赖是按需使用的；如果提示缺少模块，请安装 `requirements-optional.txt`。
- Kaggle 下载还需要本机已配置 Kaggle API 凭据，并且可以访问网络。
- 原始仓库中的部分历史脚本含硬编码路径或具有导入即执行行为。网页入口采用参数化实现；源码目录会展示这些脚本，但不会自动执行未知脚本。
- 网页处理的是用户指定的本地数据，生成文件可能占用大量磁盘空间，请确认输出目录和可用容量。
