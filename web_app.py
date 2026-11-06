"""cv_tool"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_TITLE = "CV Tool · 本地图像算法工具箱"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# 页面工具目录
TOOL_GROUPS = {
    "图像与视频": [
        "视频抽帧", "图像拼接", "图像等比例缩放", "Mask 伪彩色可视化", "根据 Mask 提取图像区域",
        "YOLO / LabelMe 标注可视化", "OBB 标注可视化", "双模型 OBB 结果对比", "批量图像分类",
        "语义分割数据增强", "OBB 数据增强"
    ],
    "Mask 与分割": [
        "LabelMe JSON 生成 Mask", "Mask 转 LabelMe JSON", "Mask 像素值映射", "统计 Mask 像素类别",
        "Mask 二值化处理", "Mask 转 NPZ 数据集"
    ],
    "标注转换与检查": [
        "统计 JSON 标注类别", "删除 JSON 指定类别", "合并 JSON 指定类别", "仅保留 JSON 指定类别",
        "LabelMe JSON 转 YOLO TXT", "YOLO TXT 转 LabelMe JSON", "COCO JSON 转 LabelMe", "LabelMe OBB 转 COCO", "VOC XML 转 YOLO", "VOC XML 转 COCO",
        "LabelMe OBB 转 YOLO HBB", "像素框 TXT 转 YOLO HBB", "YOLO TXT 类别映射/筛选", "LabelMe JSON 类别映射与 imagePath 修复",
        "COCO JSON 清理无效图像", "YOLO 数据集转 COCO", "生成 NPZ 名称列表", "YOLO 数据集划分", "YOLO 模型性能分析",
        "生成 Panoptic 数据集", "实例 Mask 转 COCO Instances", "语义+实例 Mask 生成 Panoptic", "删除空标签 TXT 文件"
    ],
    "文件与数据集管理": [
        "文件扩展名 / 名称批量重命名", "批量删除文件名子串", "按文件名末字段分组", "清理缺少对应图像的标签",
        "删除中文文件名", "批量修改文件名（去中文）", "清除 TXT 首行", "统一追加文件名前后缀",
        "保存指定类别 JSON", "YOLO 数据集文件分类", "拼接图像目录文件", "重命名 JSON 对应图像文件名"
    ],
    "高级与原始脚本": ["工具脚本目录", "Panoptic 标签检查", "实例 Mask 转 COCO Instances", "语义+实例 Mask 生成 Panoptic", "语义类别重映射", "Kaggle 数据集下载"],
}

TOOL_DESCRIPTIONS = {
    "视频抽帧": "从单个视频按帧间隔保存 JPG，输出到指定目录。",
    "图像拼接": "将多张图像按横向拼接，输出一张图像。",
    "图像等比例缩放": "批量缩放图像并保持宽高比，可按宽或高设定目标尺寸。",
    "Mask 伪彩色可视化": "将类别 ID Mask 转为稳定伪彩色图，便于查看类别分布。",
    "根据 Mask 提取图像区域": "保留指定类别对应的原图像素，背景可设透明、黑色或白色。",
    "YOLO / LabelMe 标注可视化": "读取 YOLO TXT 或 LabelMe JSON，并把框/多边形绘制到图像上。",
    "OBB 标注可视化": "读取 LabelMe JSON 或 OBB TXT，在图像上绘制旋转框并保存。",
    "双模型 OBB 结果对比": "在同一张图像上叠加两份 COCO/OBB 结果，使用不同颜色区分模型预测。",
    "批量图像分类": "按扩展名将文件复制/移动到不同目录。默认复制。",
    "LabelMe JSON 生成 Mask": "按类别 ID 将 LabelMe JSON 多边形坐标点化为单通道 Mask。",
    "Mask 转 LabelMe JSON": "把分割 Mask 轮廓转为 LabelMe polygon JSON。",
    "Mask 像素值映射": "批量将 Mask 中某些像素类别值映射为新值。",
    "统计 Mask 像素类别": "统计目录中每张 Mask 的唯一像素值和像素数量。",
    "Mask 二值化处理": "将输入 Mask 转为单通道二值 Mask：背景为 0，所有非零像素为 1。",
    "Mask 转 NPZ 数据集": "将同名图像和 Mask 保存为压缩 NPZ（image、mask 两个字段）。",
    "统计 JSON 标注类别": "汇总 LabelMe JSON 中各类别 shape 数量。",
    "删除 JSON 指定类别": "删除 LabelMe JSON 中指定类别的 shape，输出到新目录。",
    "合并 JSON 指定类别": "将另一个目录同名 JSON 中指定类别 shape 合并到基础 JSON，默认输出新目录。",
    "仅保留 JSON 指定类别": "仅保留指定类别的 shape，并将结果保存到新目录。",
    "COCO JSON 转 LabelMe": "按 COCO bbox 转换为 LabelMe rectangle JSON。",
    "LabelMe OBB 转 COCO": "将 LabelMe polygon/OBB 标注转换为 COCO JSON。",
    "VOC XML 转 YOLO": "把 Pascal VOC XML 检测框转换为 YOLO TXT，并生成 classes.txt。",
    "VOC XML 转 COCO": "把 Pascal VOC XML 转为 COCO instances JSON。",
    "LabelMe OBB 转 YOLO HBB": "把 LabelMe JSON 的四点框转成水平框 YOLO TXT。",
    "像素框 TXT 转 YOLO HBB": "把 class_id xmin ymin xmax ymax 像素坐标转换为 YOLO 归一化水平框；兼容原脚本的 6 列输入。",
    "YOLO TXT 类别映射/筛选": "对 YOLO 标签 ID 进行重映射，或仅保留/删除指定类别，结果保存到新目录。",
    "LabelMe JSON 类别映射与 imagePath 修复": "批量修改 LabelMe shape label，并按图像目录修复 imagePath，输出到新目录，不覆盖原标注。",
    "COCO JSON 清理无效图像": "删除 COCO JSON 中磁盘不存在的 image 记录，并同步过滤对应 annotations。",
    "生成 NPZ 名称列表": "把目录中的 NPZ 文件名写到 TXT。",
    "YOLO 数据集划分": "按 train/val/test 比例复制图像与标签，并生成 data.yaml。",
    "YOLO 模型性能分析": "统计 Ultralytics YOLO 模型参数量、权重大小、FLOPs 和显存等。",
    "生成 Panoptic 数据集": "组合语义 Mask、实例 Mask、类别映射文件，生成 Panoptic PNG/JSON。",
    "文件扩展名 / 名称批量重命名": "批量修改扩展名或按序号/时间戳重命名；执行前会显示预览。",
    "批量删除文件名子串": "移除文件名中的指定子串；默认仅预览。",
    "按文件名末字段分组": "按文件名最后一个下划线后的字段匹配后复制/移动/删除。",
    "清理缺少对应图像的标签": "对比参考目录与待检查目录，找出 stem 无对应关系的文件；默认仅预览。",
    "删除中文文件名": "扫描中文文件名；默认仅预览，实际删除需明确确认。",
    "批量修改文件名（去中文）": "将文件名中的中文字符移除，支持预览和确认执行。",
    "清除 TXT 首行": "批量删除 TXT 标签文件首行；输出到新目录，原始文件不覆盖。",
    "统一追加文件名前后缀": "给目录中的文件名添加前缀或后缀，默认输出到新目录。",
    "保存指定类别 JSON": "只保存包含目标类别的 JSON 标注文件到输出目录。",
    "YOLO 数据集文件分类": "从混合目录按扩展名分流图像、JSON、TXT 等文件。",
    "拼接图像目录文件": "按文件名/规则将不同目录图像拼接；请先确认图像尺寸与对应关系。",
    "重命名 JSON 对应图像文件名": "根据 LabelMe JSON 的 imagePath 和文件 stem 进行检查/修正建议。",
    "工具脚本目录": "查看压缩包中的原始 Python 脚本，包含网页工具未单独暴露的辅助/实验脚本。",
    "Panoptic 标签检查": "检查 Panoptic JSON 与 RGB 编码 mask 是否匹配。需要 Pillow/OpenCV/NumPy。",
    "语义类别重映射": "对语义 Mask 做可配置类别重映射；Thing 类可统一映射为 ignore_label。",
    "Kaggle 数据集下载": "使用 kagglehub 下载数据集；需要先配置 Kaggle API 凭据。",
    "语义分割数据增强": "对配对的图像和语义 Mask 做同步几何/颜色数据增强。",
    "OBB 数据增强": "调用仓库中的 OBB 增强算法，对图像和旋转框标注同步增强。",
    "LabelMe JSON 转 YOLO TXT": "将 LabelMe rectangle/polygon 转换为 YOLO 检测框/分割格式 TXT。",
    "YOLO TXT 转 LabelMe JSON": "将 YOLO bbox 或 polygon TXT 转为 LabelMe JSON，并读取原图尺寸。",
    "像素框 TXT 转 YOLO HBB": "把 class_id xmin ymin xmax ymax 像素坐标转换为 YOLO 归一化水平框；兼容原脚本的 6 列输入。",
    "YOLO TXT 类别映射/筛选": "对 YOLO 标签 ID 进行重映射，或仅保留/删除指定类别，结果保存到新目录。",
    "LabelMe JSON 类别映射与 imagePath 修复": "批量修改 LabelMe shape label，并按图像目录修复 imagePath，输出到新目录，不覆盖原标注。",
    "COCO JSON 清理无效图像": "删除 COCO JSON 中磁盘不存在的 image 记录，并同步过滤对应 annotations。",
    "YOLO 数据集转 COCO": "将包含 images、labels、classes.txt 的 YOLO 数据集转为 COCO JSON，可选择随机拆分或清单拆分。",
    "实例 Mask 转 COCO Instances": "根据实例 Mask 连通域与语义 Mask 多数投票，生成 COCO instances JSON（RLE 编码）。",
    "语义+实例 Mask 生成 Panoptic": "根据语义类别和实例 ID 生成 Panoptic RGB 编码 Mask 与 JSON。",
    "删除空标签 TXT 文件": "扫描空白 TXT 标签；默认预览，确认后才删除。",
}


def _path(p: str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(p).strip().strip('"')))
    if not raw:
        raise ValueError("路径不能为空，请填写绝对路径。")
    return Path(raw)


def _need_dir(value: str, name: str) -> Path:
    p = _path(value)
    if not p.is_dir():
        raise FileNotFoundError(f"{name}目录不存在：{p}")
    return p


def _need_file(value: str, name: str) -> Path:
    p = _path(value)
    if not p.is_file():
        raise FileNotFoundError(f"{name}文件不存在：{p}")
    return p


def _out_dir(value: str) -> Path:
    p = _path(value)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _imread(path: Path, flags=None):
    import cv2
    import numpy as np
    flags = cv2.IMREAD_COLOR if flags is None else flags
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(str(path), flags)


def _imwrite(path: Path, image) -> bool:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def _images(folder: Path, recursive=False):
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda x: str(x).lower())


def _json_files(folder: Path, recursive=False):
    iterator = folder.rglob("*.json") if recursive else folder.glob("*.json")
    return sorted((p for p in iterator if p.is_file()), key=lambda x: str(x).lower())


def _parse_labels(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,，\n;；]", text or "") if x.strip()]


def _parse_class_map(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in re.split(r"[,，\n;；]", text or ""):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"类别映射格式错误：{item!r}，正确格式为 class_name=class_id")
        name, value = item.rsplit("=", 1)
        result[name.strip()] = int(value.strip())
    if not result:
        raise ValueError("请至少填写一个类别映射，例如 background=0,cable=1")
    return result


def _parse_mapping(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in re.split(r"[,，\n;；]", text or ""):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"像素映射格式错误：{item!r}，正确格式为 old_id=new_id")
        a, b = item.rsplit("=", 1)
        result[int(a.strip())] = int(b.strip())
    if not result:
        raise ValueError("请至少填写一个像素映射，例如 1=2,255=0")
    return result


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根节点不是 object：{path}")
    return data


def _write_json(path: Path, data: Any, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _tool_catalog() -> list[dict]:
    entries = []
    for category, names in TOOL_GROUPS.items():
        for name in names:
            entries.append({"category": category, "name": name, "description": TOOL_DESCRIPTIONS.get(name, "原始脚本功能")})
    return entries


SCRIPT_TOOL_HINTS = {
    "video_process/video2images_save.py": "视频抽帧",
    "data_handle/append_images.py": "拼接图像目录文件",
    "data_handle/zoom_image.py": "图像等比例缩放",
    "drawer/mask/show_pseudo_color.py": "Mask 伪彩色可视化",
    "data_handle/mask_tools/mask2image.py": "根据 Mask 提取图像区域",
    "drawer/boxes/read_labels_to_images.py": "YOLO / LabelMe 标注可视化",
    "drawer/obb/show_bbox.py": "OBB 标注可视化",
    "drawer/draw/show_obb_coco2images.py": "双模型 OBB 结果对比",
    "FileClassification/images_labels_classification.py": "YOLO 数据集文件分类",
    "data_handle/data_augmentation-seg.py": "语义分割数据增强",
    "data_handle/data_augmentation-obb.py": "OBB 数据增强",
    "data_handle/mask_tools/make_mask.py": "LabelMe JSON 生成 Mask",
    "data_handle/mask_tools/mask2jsons.py": "Mask 转 LabelMe JSON",
    "data_handle/mask_tools/re_mask_pixel.py": "Mask 像素值映射",
    "data_handle/mask_tools/count_mask_labels.py": "统计 Mask 像素类别",
    "data_handle/mask_tools/mask_data_process.py": "Mask 二值化处理",
    "data_handle/make_npz_data.py": "Mask 转 NPZ 数据集",
    "data_handle/json_tools/count_json_label.py": "统计 JSON 标注类别",
    "data_handle/json_tools/remove_class_labels.py": "删除 JSON 指定类别",
    "data_handle/json_tools/remove_jsonlabel2newjson.py": "合并 JSON 指定类别",
    "data_handle/json_tools/json2txt.py": "LabelMe JSON 转 YOLO TXT",
    "data_handle/txt2json.py": "YOLO TXT 转 LabelMe JSON",
    "data_handle/split_coco_annotations.py": "COCO JSON 转 LabelMe",
    "data_handle/make_obb_coco_json.py": "LabelMe OBB 转 COCO",
    "data_handle/xml2yolo.py": "VOC XML 转 YOLO",
    "data_handle/xml2coco.py": "VOC XML 转 COCO",
    "data_handle/normal_boxes.py": "像素框 TXT 转 YOLO HBB",
    "data_handle/obb_to_yolo.py": "LabelMe OBB 转 YOLO HBB",
    "data_handle/yolodatas2cocodatas.py": "YOLO 数据集转 COCO",
    "data_handle/make_npz_train_dataset.py": "生成 NPZ 名称列表",
    "split_train_datas/split_yolo.py": "YOLO 数据集划分",
    "model_process/yolo_per.py": "YOLO 模型性能分析",
    "data_handle/goose_process/make_panoptic_mask_2_classmap_fixed.py": "生成 Panoptic 数据集",
    "data_handle/goose_process/make_instanceids_json.py": "实例 Mask 转 COCO Instances",
    "data_handle/goose_process/panoptic_json2instances_json.py": "语义+实例 Mask 生成 Panoptic",
    "data_handle/remove/remove_nothay_label_files.py": "删除空标签 TXT 文件",
    "data_handle/rename_files.py": "文件扩展名 / 名称批量重命名",
    "data_handle/delet_file_rename.py": "批量删除文件名子串",
    "data_handle/file_name_group.py": "按文件名末字段分组",
    "data_handle/remove/remove_not_same_name_files.py": "清理缺少对应图像的标签",
    "data_handle/remove/remove_chinese_name_file.py": "删除中文文件名",
    "data_handle/rename_same_chinese_files.py": "批量修改文件名（去中文）",
    "data_handle/remove/remove_the_first_line.py": "清除 TXT 首行",
    "data_handle/make_over_suffix.py": "统一追加文件名前后缀",
    "data_handle/save_specify_label_file.py": "保存指定类别 JSON",
    "data_handle/goose_process/rename_json_filename.py": "重命名 JSON 对应图像文件名",
    "data_handle/goose_process/re_semantic.py": "语义类别重映射",
    "data_handle/download.py": "Kaggle 数据集下载",
}


def _json_stats(folder: Path, recursive=False) -> dict[str, int]:
    counter: Counter[str] = Counter()
    errors = []
    files = _json_files(folder, recursive)
    for p in files:
        try:
            data = _load_json(p)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label is not None:
                    counter[str(label)] += 1
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    return {"files": len(files), "counts": dict(counter), "errors": errors}


def _preview_files(paths: list[Path], limit=60) -> str:
    if not paths:
        return "未发现匹配文件。"
    lines = [f"匹配文件：{len(paths)} 个", *[str(p) for p in paths[:limit]]]
    if len(paths) > limit:
        lines.append(f"……另有 {len(paths)-limit} 个文件未显示。")
    return "\n".join(lines)


def execute_tool(tool: str, p: dict[str, Any]) -> dict[str, Any]:
    """按工具名称执行后端任务。路径均由用户在表单中指定。"""
    result: dict[str, Any] = {"tool": tool, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    if tool == "视频抽帧":
        import cv2
        video = _need_file(p["video_file"], "视频")
        out = _out_dir(p["output_dir"])
        interval = int(p["frame_interval"])
        if interval < 1:
            raise ValueError("帧间隔必须大于等于 1")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video}")
        frame_index = 0
        saved = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % interval == 0:
                    target = out / f"{video.stem}_{frame_index:08d}.jpg"
                    if _imwrite(target, frame):
                        saved += 1
                frame_index += 1
        finally:
            cap.release()
        result.update(message=f"抽帧完成：读取 {frame_index} 帧，保存 {saved} 张。", output_dir=str(out), saved=saved)
        return result

    if tool == "图像拼接":
        import cv2
        import numpy as np
        paths = [_need_file(x, "图像") for x in _parse_labels(p["image_paths"])]
        if len(paths) < 2:
            raise ValueError("至少需要 2 张图像，路径之间用英文逗号分隔")
        imgs = [_imread(x) for x in paths]
        if any(x is None for x in imgs):
            raise RuntimeError("存在无法解码的图像")
        target_h = imgs[0].shape[0]
        resized = [cv2.resize(img, (max(1, round(img.shape[1] * target_h / img.shape[0])), target_h)) for img in imgs]
        merged = cv2.hconcat(resized)
        out = _path(p["output_file"])
        if not out.suffix:
            out = out.with_suffix(".jpg")
        if not _imwrite(out, merged):
            raise RuntimeError(f"保存失败：{out}")
        result.update(message=f"拼接完成：{len(paths)} 张图像，结果尺寸 {merged.shape[1]}×{merged.shape[0]}。", output_file=str(out))
        return result

    if tool == "图像等比例缩放":
        import cv2
        source = _need_dir(p["input_dir"], "输入")
        out = _out_dir(p["output_dir"])
        size = int(p["target_size"])
        if size < 1:
            raise ValueError("目标尺寸必须大于 0")
        by_width = bool(p["by_width"])
        files = _images(source, bool(p["recursive"]))
        done, skipped = 0, []
        for f in files:
            img = _imread(f)
            if img is None:
                skipped.append(f"无法读取：{f}"); continue
            h, w = img.shape[:2]
            scale = size / (w if by_width else h)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _imwrite(dest, cv2.resize(img, (new_w, new_h))): done += 1
            else: skipped.append(f"保存失败：{dest}")
        result.update(message=f"缩放完成：成功 {done}，失败/跳过 {len(skipped)}。", output_dir=str(out), errors=skipped[:30])
        return result

    if tool == "Mask 伪彩色可视化":
        import cv2
        import numpy as np
        source = _need_dir(p["mask_dir"], "Mask")
        out = _out_dir(p["output_dir"])
        files = _images(source, bool(p["recursive"]))
        done, errors = 0, []
        for f in files:
            mask = _imread(f, cv2.IMREAD_UNCHANGED)
            if mask is None:
                errors.append(f"读取失败：{f}"); continue
            if mask.ndim == 3:
                if mask.shape[2] == 4: mask = mask[:, :, :3]
                # 类别 mask 通常为单通道；对彩色 mask 使用首通道作为 ID
                mask = mask[:, :, 0]
            ids = np.unique(mask)
            canvas = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
            for val in ids:
                if int(val) == 0:
                    color = (0, 0, 0)
                else:
                    seed = int(val) * 2654435761 % (2**32)
                    rng = np.random.default_rng(seed)
                    color = tuple(int(x) for x in rng.integers(55, 256, 3))
                canvas[mask == val] = color
            dest = out / f"{f.stem}_pseudo.png"
            _imwrite(dest, canvas); done += 1
        result.update(message=f"伪彩色转换完成：{done} 张，失败 {len(errors)} 张。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "根据 Mask 提取图像区域":
        image_dir = _need_dir(p["image_dir"], "原图")
        mask_dir = _need_dir(p["mask_dir"], "Mask")
        out = _out_dir(p["output_dir"])
        labels = [int(x) for x in _parse_labels(p["labels"])]
        if not labels:
            raise ValueError("请填写要保留的类别 ID")
        from data_handle.mask_tools.mask2image import extract_labels_from_mask
        ext = p["mask_ext"].strip() or None
        ok, skipped, errors = extract_labels_from_mask(str(image_dir), str(mask_dir), labels, str(out), p["background"], p["output_ext"], ext)
        result.update(message=f"完成：成功 {ok}，跳过 {skipped}，错误 {errors}。", output_dir=str(out))
        return result

    if tool == "YOLO / LabelMe 标注可视化":
        from drawer.boxes.read_labels_to_images import process_folder
        images = _need_dir(p["image_dir"], "图像")
        labels = _need_dir(p["label_dir"], "标签")
        out = _out_dir(p["output_dir"])
        classes = p["classes_file"].strip()
        if classes:
            classes_path = _need_file(classes, "类别文件")
        else:
            # 允许没有 classes.txt；按标签 ID 显示数字类别。
            classes_path = out / "_generated_classes.txt"
            classes_path.write_text("\n".join(f"class_{i}" for i in range(200)), encoding="utf-8")
        ext_list = [x.strip().lower() for x in _parse_labels(p["extensions"])] if p["extensions"].strip() else [".jpg", ".jpeg", ".png", ".bmp"]
        process_folder(str(images), str(labels), str(out), str(classes_path), bool(p["save_undetected"]), ext_list, int(p["workers"]))
        result.update(message="标注可视化完成。", output_dir=str(out))
        return result

    if tool == "双模型 OBB 结果对比":
        json1 = _need_file(p["json1"], "模型 1 COCO JSON")
        json2 = _need_file(p["json2"], "模型 2 COCO JSON")
        images = _need_dir(p["image_dir"], "原图")
        out = _out_dir(p["output_dir"])
        from drawer.draw.show_obb_coco2images import visualize_combined_obb
        mapping = {}
        for line in p["category_map"].splitlines():
            if not line.strip(): continue
            if "=" not in line: raise ValueError(f"类别映射格式错误：{line}，正确格式为 id=name")
            k,v=line.split("=",1); mapping[int(k.strip())]=v.strip()
        color1 = tuple(int(x.strip()) for x in p["color1"].split(","))
        color2 = tuple(int(x.strip()) for x in p["color2"].split(","))
        if len(color1)!=3 or len(color2)!=3 or any(x<0 or x>255 for x in (*color1,*color2)):
            raise ValueError("模型颜色必须为三个 0–255 的 BGR 整数，例如 255,0,0")
        visualize_combined_obb(str(json1), str(json2), str(images), str(out), mapping,
            {"name":p["model1_name"],"color":color1}, {"name":p["model2_name"],"color":color2})
        result.update(message="双模型 OBB 对比可视化完成。", output_dir=str(out))
        return result

    if tool == "OBB 标注可视化":
        import cv2
        import numpy as np
        images_dir = _need_dir(p["image_dir"], "图像")
        labels_dir = _need_dir(p["label_dir"], "标签")
        out = _out_dir(p["output_dir"])
        label_ext = p["label_ext"].lower()
        done, no_label, errors = 0, 0, []
        for img_path in _images(images_dir, bool(p["recursive"])):
            label_path = labels_dir / f"{img_path.stem}{label_ext}"
            if not label_path.is_file(): no_label += 1; continue
            img = _imread(img_path)
            if img is None: errors.append(f"读取失败：{img_path}"); continue
            objects = []
            try:
                if label_ext == ".json":
                    data = _load_json(label_path)
                    for sh in data.get("shapes", []):
                        pts = sh.get("points", [])
                        if len(pts) >= 4: objects.append((str(sh.get("label", "?")), np.asarray(pts, dtype=np.int32)))
                else:
                    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                        vals = line.split()
                        if len(vals) >= 9:
                            objects.append((vals[0], np.asarray([float(x) for x in vals[1:9]], dtype=np.int32).reshape(4, 2)))
                for label, pts in objects:
                    cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                    x, y = pts[0]
                    cv2.putText(img, label, (int(x), max(15, int(y)-4)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                _imwrite(out / img_path.name, img); done += 1
            except Exception as e: errors.append(f"{label_path.name}: {e}")
        result.update(message=f"可视化完成：保存 {done} 张，无标签 {no_label} 张，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "批量图像分类":
        source = _need_dir(p["source_dir"], "源")
        folders = {}
        for line in (p["rules"] or "").splitlines():
            if not line.strip(): continue
            if "=" not in line: raise ValueError(f"分类规则格式错误：{line}，需要 类别名=扩展名列表")
            name, exts = line.split("=", 1)
            suffixes = [x.strip().lower() if x.strip().startswith(".") else "."+x.strip().lower() for x in exts.split(",") if x.strip()]
            folders[name.strip()] = (source.parent / name.strip(), suffixes)
        if not folders: raise ValueError("至少设置一条规则，例如 images=.jpg,.png")
        move = bool(p["move"])
        confirm = bool(p["confirm"])
        files = [x for x in source.rglob("*") if x.is_file()]
        matches = [x for x in files if any(x.suffix.lower() in exts for _, exts in folders.values())]
        if move and not confirm: raise ValueError("移动操作需要勾选确认；也可以先使用复制模式")
        done, errors = 0, []
        for f in matches:
            dest_dir = next((d for d, exts in folders.values() if f.suffix.lower() in exts), None)
            if dest_dir is None: continue
            dest_dir.mkdir(parents=True, exist_ok=True); dest = dest_dir / f.name
            if dest.exists(): errors.append(f"目标已存在，跳过：{dest}"); continue
            try:
                if move: shutil.move(str(f), str(dest))
                else: shutil.copy2(f, dest)
                done += 1
            except Exception as e: errors.append(f"{f}: {e}")
        result.update(message=f"分类完成：{done} 个文件，错误/跳过 {len(errors)}。", errors=errors[:30])
        return result

    if tool == "LabelMe JSON 生成 Mask":
        images = _need_dir(p["image_dir"], "图像")
        labels = _need_dir(p["label_dir"], "JSON 标签")
        out = _out_dir(p["output_dir"])
        cmap = _parse_class_map(p["class_map"])
        import cv2
        import numpy as np
        done, missing, errors = 0, 0, []
        for image in _images(images, bool(p["recursive"])):
            img = _imread(image)
            if img is None: errors.append(f"图像读取失败：{image.name}"); continue
            h, w = img.shape[:2]
            js = labels / f"{image.stem}.json"
            mask = np.zeros((h, w), dtype=np.uint8)
            if not js.exists():
                if not p["generate_empty"]: missing += 1; continue
            else:
                try:
                    data = _load_json(js)
                    for shape in data.get("shapes", []):
                        label = shape.get("label")
                        points = shape.get("points", [])
                        if label in cmap and len(points) >= 3:
                            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], int(cmap[label]))
                except Exception as e: errors.append(f"{js.name}: {e}"); continue
            if _imwrite(out / f"{image.stem}.png", mask): done += 1
        result.update(message=f"Mask 生成完成：{done} 张，缺少 JSON 跳过 {missing}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "Mask 转 LabelMe JSON":
        from data_handle.mask_tools.mask2jsons import batch_convert
        masks = _need_dir(p["mask_dir"], "Mask")
        images = _need_dir(p["image_dir"], "图像")
        out = _out_dir(p["output_dir"])
        batch_convert(str(masks), str(images), str(out), float(p["min_area"]), float(p["epsilon_ratio"]), int(p["workers"]), int(p["bridge_width"]), float(p["bridge_extension"]))
        result.update(message="Mask 到 LabelMe JSON 转换完成。", output_dir=str(out))
        return result

    if tool == "Mask 像素值映射":
        import cv2
        import numpy as np
        from data_handle.mask_tools.re_mask_pixel import remap_mask
        source = _need_dir(p["input_dir"], "输入")
        out = _out_dir(p["output_dir"])
        mapping = _parse_mapping(p["mapping"])
        files = _images(source, bool(p["recursive"]))
        done, errors = 0, []
        for f in files:
            m = _imread(f, cv2.IMREAD_UNCHANGED)
            if m is None: errors.append(f"读取失败：{f.name}"); continue
            try:
                remapped = remap_mask(m, mapping)
                if _imwrite(out / f.name, remapped): done += 1
            except Exception as e: errors.append(f"{f.name}: {e}")
        result.update(message=f"像素值映射完成：成功 {done}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "统计 Mask 像素类别":
        import numpy as np
        import cv2
        source = _need_dir(p["mask_dir"], "Mask")
        rows, errors = [], []
        for f in _images(source, bool(p["recursive"])):
            arr = _imread(f, cv2.IMREAD_UNCHANGED)
            if arr is None: errors.append(f"读取失败：{f.name}"); continue
            if arr.ndim == 3: arr = arr[:, :, 0]
            vals, counts = np.unique(arr, return_counts=True)
            rows.append({"文件": f.name, "像素总数": int(arr.size), "类别数": int(len(vals)), "类别ID及像素数": ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))})
        result.update(message=f"统计完成：{len(rows)} 张 Mask。", table=rows, errors=errors[:30])
        return result

    if tool == "Mask 二值化处理":
        import cv2
        import numpy as np
        source = _need_dir(p["input_dir"], "输入")
        out = _out_dir(p["output_dir"])
        files = _images(source, bool(p["recursive"]))
        done, errors = 0, []
        for f in files:
            try:
                mask = _imread(f, cv2.IMREAD_UNCHANGED)
                if mask is None: raise ValueError("图像无法读取")
                if mask.ndim == 3: mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
                binary = np.where(mask > 0, 1, 0).astype(np.uint8)
                dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name))
                if not _imwrite(dest, binary): raise IOError("写入失败")
                done += 1
            except Exception as e: errors.append(f"{f.name}: {e}")
        result.update(message=f"二值 Mask 处理完成：成功 {done}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "Mask 转 NPZ 数据集":
        import cv2
        import numpy as np
        images = _need_dir(p["image_dir"], "图像")
        masks = _need_dir(p["mask_dir"], "Mask")
        out = _out_dir(p["output_dir"])
        mask_ext = p["mask_ext"].strip() or ".png"
        done, skipped, errors = 0, 0, []
        for img_path in _images(images, bool(p["recursive"])):
            mask_path = masks / f"{img_path.stem}{mask_ext}"
            if not mask_path.exists(): skipped += 1; continue
            img, mask = _imread(img_path), _imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None: errors.append(f"读取失败：{img_path.name}"); continue
            if img.shape[:2] != mask.shape[:2]: errors.append(f"尺寸不一致：{img_path.name}"); continue
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            np.savez_compressed(out / f"{img_path.stem}.npz", image=image_rgb, mask=mask)
            done += 1
        result.update(message=f"NPZ 生成完成：{done} 个，缺少 Mask 跳过 {skipped}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "统计 JSON 标注类别":
        folder = _need_dir(p["json_dir"], "JSON")
        stats = _json_stats(folder, bool(p["recursive"]))
        rows = [{"类别": k, "标注数量": v} for k, v in sorted(stats["counts"].items(), key=lambda x: (-x[1], x[0]))]
        result.update(message=f"扫描 JSON {stats['files']} 个，类别 {len(rows)} 个，标注 {sum(stats['counts'].values())} 个。", table=rows, errors=stats["errors"][:30])
        return result

    if tool in {"删除 JSON 指定类别", "仅保留 JSON 指定类别", "保存指定类别 JSON"}:
        source = _need_dir(p["input_dir"], "输入 JSON")
        out = _out_dir(p["output_dir"])
        selected = set(_parse_labels(p["labels"]))
        if not selected: raise ValueError("请至少填写一个类别名称")
        keep_only = tool != "删除 JSON 指定类别"
        only_save_containing = tool == "保存指定类别 JSON"
        files = _json_files(source, bool(p["recursive"]))
        done, skipped, errors = 0, 0, []
        for f in files:
            try:
                data = _load_json(f)
                shapes = data.get("shapes", [])
                if only_save_containing and not any(s.get("label") in selected for s in shapes):
                    skipped += 1; continue
                if keep_only:
                    data["shapes"] = [s for s in shapes if s.get("label") in selected]
                else:
                    data["shapes"] = [s for s in shapes if s.get("label") not in selected]
                dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name))
                _write_json(dest, data, int(p["indent"]))
                done += 1
            except Exception as e: errors.append(f"{f.name}: {e}")
        action = "筛选" if keep_only else "删除指定类别"
        result.update(message=f"{action}完成：写入 {done} 个 JSON，按规则跳过 {skipped}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "合并 JSON 指定类别":
        base = _need_dir(p["base_dir"], "基础 JSON")
        add = _need_dir(p["add_dir"], "追加 JSON")
        out = _out_dir(p["output_dir"])
        labels = set(_parse_labels(p["labels"]))
        if not labels: raise ValueError("请填写需要合并的 label")
        if p["overwrite"] and not p["confirm"]: raise ValueError("原地覆盖需要勾选“确认覆盖基础 JSON”。默认建议输出到新目录。")
        base_files = {f.name: f for f in _json_files(base)}
        add_files = {f.name: f for f in _json_files(add)}
        common = sorted(set(base_files) & set(add_files))
        done, added_count, errors = 0, 0, []
        for name in common:
            try:
                b = _load_json(base_files[name]); a = _load_json(add_files[name])
                if not isinstance(b.get("shapes"), list) or not isinstance(a.get("shapes"), list): raise ValueError("shapes 字段不是列表")
                selected = [s for s in a["shapes"] if s.get(p["label_field"], s.get("label")) in labels]
                b["shapes"].extend(selected)
                dest = base_files[name] if p["overwrite"] else out / name
                _write_json(dest, b, int(p["indent"]))
                done += 1; added_count += len(selected)
            except Exception as e: errors.append(f"{name}: {e}")
        result.update(message=f"合并完成：成功 {done}/{len(common)} 个 JSON，追加 shape {added_count} 个。", output_dir=str(base if p["overwrite"] else out), errors=errors[:30])
        return result

    if tool == "COCO JSON 转 LabelMe":
        coco_file = _need_file(p["coco_json"], "COCO JSON")
        out = _out_dir(p["output_dir"])
        coco = json.loads(coco_file.read_text(encoding="utf-8-sig"))
        imgs = {x["id"]: x for x in coco.get("images", [])}
        ann_map: dict[Any, list] = {}
        for a in coco.get("annotations", []): ann_map.setdefault(a.get("image_id"), []).append(a)
        done = 0
        for image_id, info in imgs.items():
            shapes = []
            for ann in ann_map.get(image_id, []):
                if not ann.get("bbox") or len(ann["bbox"]) < 4: continue
                x, y, w, h = ann["bbox"][:4]
                shapes.append({"label": str(p["label_name"]), "score": None, "points": [[x,y],[x+w,y],[x+w,y+h],[x,y+h]], "group_id": None, "description": "", "difficult": False, "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []})
            labelme = {"version": "5.0.0", "flags": {}, "shapes": shapes, "imagePath": info.get("file_name", ""), "imageData": None, "imageHeight": info.get("height", 0), "imageWidth": info.get("width", 0)}
            _write_json(out / f"{Path(info.get('file_name','image')).stem}.json", labelme)
            done += 1
        result.update(message=f"转换完成：生成 {done} 个 LabelMe JSON。", output_dir=str(out))
        return result

    if tool == "LabelMe OBB 转 COCO":
        json_dir = _need_dir(p["json_dir"], "LabelMe JSON")
        image_dir = _need_dir(p["image_dir"], "图像")
        classes_file = _need_file(p["classes_file"], "类别文件")
        output_json = _path(p["output_json"])
        classes = [x.strip() for x in classes_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        cat_map = {name: i+1 for i, name in enumerate(classes)}
        images, annotations = [], []
        image_id = ann_id = 1
        for jf in _json_files(json_dir, bool(p["recursive"])):
            try:
                data = _load_json(jf)
                img_name = Path(data.get("imagePath", "")).name
                if not img_name or not (image_dir / img_name).exists():
                    match = next((x for x in image_dir.iterdir() if x.stem == jf.stem and x.suffix.lower() in IMAGE_EXTS), None)
                    if match is None: continue
                    img_name = match.name
                width, height = int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0)
                if not width or not height:
                    im = _imread(image_dir / img_name)
                    if im is None: continue
                    height, width = im.shape[:2]
                images.append({"id": image_id, "file_name": img_name, "width": width, "height": height})
                for s in data.get("shapes", []):
                    name = s.get("label")
                    points = s.get("points", [])
                    if name not in cat_map or len(points) < 3: continue
                    xs = [float(q[0]) for q in points]; ys = [float(q[1]) for q in points]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                    area = abs(sum(float(points[i][0])*float(points[(i+1)%len(points)][1]) - float(points[(i+1)%len(points)][0])*float(points[i][1]) for i in range(len(points))))/2
                    seg = [v for pt in points for v in [float(pt[0]), float(pt[1])]]
                    annotations.append({"id": ann_id, "image_id": image_id, "category_id": cat_map[name], "segmentation": [seg], "area": area, "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], "iscrowd": 0})
                    ann_id += 1
                image_id += 1
            except Exception as e:
                result.setdefault("errors", []).append(f"{jf.name}: {e}")
        coco = {"images": images, "annotations": annotations, "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in cat_map.items()]}
        _write_json(output_json, coco)
        result.update(message=f"转换完成：图像 {len(images)} 张，标注 {len(annotations)} 个。", output_file=str(output_json))
        return result

    if tool == "VOC XML 转 YOLO":
        from data_handle.xml2yolo import batch_convert_voc_to_yolo
        xml_dir = _need_dir(p["xml_dir"], "XML")
        out = _out_dir(p["output_dir"])
        classes = _parse_labels(p["classes"])
        if not classes: raise ValueError("请按类别 ID 顺序填写类别名，例如 person,hat")
        batch_convert_voc_to_yolo(str(xml_dir), str(out), classes)
        result.update(message="VOC XML 转 YOLO 完成。", output_dir=str(out))
        return result

    if tool == "VOC XML 转 COCO":
        from data_handle.xml2coco import voc_to_coco
        xml_dir = _need_dir(p["xml_dir"], "XML")
        output_json = _path(p["output_json"])
        classes = _parse_labels(p["classes"])
        if not classes: raise ValueError("请填写类别列表")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        # 兼容原脚本同时修正 category_id 为 1-based 的 COCO 标准。
        import xml.etree.ElementTree as ET
        data = {"info": {"description": "VOC to COCO", "year": datetime.now().year}, "images": [], "annotations": [], "categories": [{"id": i+1, "name": n, "supercategory": "none"} for i,n in enumerate(classes)]}
        ann_id = img_id = 1
        for xf in sorted(xml_dir.glob("*.xml")):
            root = ET.parse(xf).getroot(); size = root.find("size")
            if size is None: continue
            w, h = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
            filename = root.findtext("filename") or f"{xf.stem}.jpg"
            data["images"].append({"id": img_id, "file_name": filename, "width": w, "height": h})
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name not in classes: continue
                box = obj.find("bndbox")
                if box is None: continue
                x1,y1,x2,y2 = [float(box.findtext(k, "0")) for k in ("xmin","ymin","xmax","ymax")]
                bw,bh = max(0,x2-x1), max(0,y2-y1)
                data["annotations"].append({"id":ann_id,"image_id":img_id,"category_id":classes.index(name)+1,"bbox":[x1,y1,bw,bh],"area":bw*bh,"iscrowd":0,"segmentation":[]})
                ann_id += 1
            img_id += 1
        _write_json(output_json, data)
        result.update(message=f"COCO 生成完成：图像 {len(data['images'])} 张，标注 {len(data['annotations'])} 个。", output_file=str(output_json))
        return result

    if tool == "像素框 TXT 转 YOLO HBB":
        txt_dir = _need_dir(p["txt_dir"], "TXT 标签")
        image_dir = _need_dir(p["image_dir"], "图像")
        out = _out_dir(p["output_dir"])
        recursive = bool(p["recursive"])
        txt_files = sorted(txt_dir.rglob("*.txt") if recursive else txt_dir.glob("*.txt"))
        image_index = {}
        for image in _images(image_dir, recursive):
            image_index.setdefault(image.stem, image)
        done, skipped, errors = 0, 0, []
        for txt in txt_files:
            image = image_index.get(txt.stem)
            if image is None:
                skipped += 1; errors.append(f"找不到同名图像：{txt.name}"); continue
            arr = _imread(image)
            if arr is None:
                errors.append(f"图像读取失败：{image}"); continue
            height, width = arr.shape[:2]
            lines = []
            try:
                for line_no, line in enumerate(txt.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip(): continue
                    cols = line.split()
                    if len(cols) == 5:
                        cls, x1, y1, x2, y2 = cols
                    elif len(cols) == 6:
                        cls, x1, y1, x2, y2 = cols[0], *cols[2:6]
                    else:
                        raise ValueError(f"第 {line_no} 行应为 5 列或 6 列，实际 {len(cols)} 列")
                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                    if x2 < x1 or y2 < y1: raise ValueError(f"第 {line_no} 行坐标顺序错误")
                    xc = ((x1 + x2) / 2.0) / width
                    yc = ((y1 + y2) / 2.0) / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height
                    lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                dest = out / (txt.relative_to(txt_dir) if recursive else Path(txt.name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                done += 1
            except Exception as e:
                errors.append(f"{txt.name}: {e}")
        result.update(message=f"像素框转换完成：成功 {done}，缺少图像/跳过 {skipped}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:40])
        return result

    if tool == "YOLO TXT 类别映射/筛选":
        source = _need_dir(p["input_dir"], "YOLO TXT")
        out = _out_dir(p["output_dir"])
        mode = p["mode"]
        mapping = _parse_mapping(p["mapping"]) if mode == "映射类别 ID" else {}
        selected = {int(x) for x in _parse_labels(p["class_ids"])} if p["class_ids"].strip() else set()
        if mode != "映射类别 ID" and not selected:
            raise ValueError("筛选/删除模式必须填写类别 ID 列表")
        files = sorted(source.rglob("*.txt") if p["recursive"] else source.glob("*.txt"))
        done, skipped, errors = 0, 0, []
        for f in files:
            try:
                output_lines = []
                for line_no, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip(): continue
                    parts = line.split()
                    try: cls = int(parts[0])
                    except Exception: raise ValueError(f"第 {line_no} 行类别 ID 不是整数")
                    if mode == "映射类别 ID":
                        if cls in mapping: parts[0] = str(mapping[cls])
                        elif not p["keep_unmapped"]: continue
                        output_lines.append(" ".join(parts))
                    elif mode == "仅保留指定类别":
                        if cls in selected: output_lines.append(" ".join(parts))
                    elif mode == "删除指定类别":
                        if cls not in selected: output_lines.append(" ".join(parts))
                if not output_lines and p["drop_empty"]:
                    skipped += 1; continue
                dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
                done += 1
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        result.update(message=f"YOLO TXT 处理完成：写入 {done} 个，按空标签规则跳过 {skipped} 个，错误 {len(errors)}。", output_dir=str(out), errors=errors[:40])
        return result

    if tool == "LabelMe JSON 类别映射与 imagePath 修复":
        source = _need_dir(p["json_dir"], "LabelMe JSON")
        images_dir = _need_dir(p["image_dir"], "图像")
        out = _out_dir(p["output_dir"])
        label_map = {}
        for line in p["label_map"].splitlines():
            line = line.strip()
            if not line: continue
            if "=" not in line: raise ValueError(f"类别映射格式错误：{line}，正确格式为 old_label=new_label")
            old, new = line.split("=", 1); label_map[old.strip()] = new.strip()
        image_index = {}
        for image in _images(images_dir, True): image_index.setdefault(image.name.lower(), image)
        stem_index = {}
        for image in _images(images_dir, True): stem_index.setdefault(image.stem.lower(), image)
        files = _json_files(source, bool(p["recursive"]))
        done, changed_labels, fixed_paths, errors = 0, 0, 0, []
        for jf in files:
            try:
                data = _load_json(jf)
                for shape in data.get("shapes", []):
                    old = shape.get("label")
                    if old in label_map:
                        shape["label"] = label_map[old]; changed_labels += 1
                old_path = str(data.get("imagePath") or "").replace("\\", "/")
                image = image_index.get(Path(old_path).name.lower()) if old_path else None
                if image is None: image = stem_index.get(jf.stem.lower())
                dest = out / (jf.relative_to(source) if p["recursive"] else Path(jf.name))
                if image is not None:
                    rel = os.path.relpath(str(image), str(dest.parent)).replace("\\", "/")
                    if data.get("imagePath") != rel:
                        data["imagePath"] = rel; fixed_paths += 1
                else:
                    errors.append(f"{jf.name}: 找不到同名图像，imagePath 保持原值")
                _write_json(dest, data)
                done += 1
            except Exception as e: errors.append(f"{jf.name}: {e}")
        result.update(message=f"处理完成：输出 JSON {done} 个，修改 label {changed_labels} 处，修复 imagePath {fixed_paths} 个。", output_dir=str(out), errors=errors[:50])
        return result

    if tool == "COCO JSON 清理无效图像":
        coco_file = _need_file(p["coco_json"], "COCO JSON")
        image_dir = _need_dir(p["image_dir"], "图像")
        output_json = _path(p["output_json"])
        data = json.loads(coco_file.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("images"), list):
            raise ValueError("COCO JSON 格式错误：缺少 images 列表")
        valid_images, valid_ids, missing = [], set(), []
        for image in data["images"]:
            name = str(image.get("file_name", ""))
            image_path = (image_dir / name).resolve()
            try: image_path.relative_to(image_dir.resolve())
            except ValueError:
                missing.append(name + "（路径越界）"); continue
            if name and image_path.is_file():
                valid_images.append(image); valid_ids.add(image.get("id"))
            else:
                missing.append(name or "（空 file_name）")
        original_annotations = data.get("annotations", [])
        data["images"] = valid_images
        if isinstance(original_annotations, list):
            data["annotations"] = [a for a in original_annotations if a.get("image_id") in valid_ids]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_json, data)
        result.update(message=f"COCO 清理完成：保留图像 {len(valid_images)} 张，移除不存在图像 {len(missing)} 张，保留标注 {len(data.get('annotations', []))} 个。", output_file=str(output_json), errors=missing[:50])
        return result

    if tool == "LabelMe OBB 转 YOLO HBB":
        json_dir = _need_dir(p["json_dir"], "JSON")
        out = _out_dir(p["output_dir"])
        class_map = _parse_class_map(p["class_map"])
        done, errors = 0, []
        for jf in _json_files(json_dir, bool(p["recursive"])):
            try:
                data = _load_json(jf); w,h = int(data["imageWidth"]), int(data["imageHeight"])
                if w <= 0 or h <= 0: raise ValueError("imageWidth/imageHeight 不正确")
                lines = []
                for s in data.get("shapes", []):
                    name = s.get("label"); pts = s.get("points", [])
                    if name not in class_map or not pts: continue
                    xs = [float(q[0]) for q in pts]; ys = [float(q[1]) for q in pts]
                    x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
                    xc,yc,bw,bh = (x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h
                    lines.append(f"{class_map[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                (out / f"{jf.stem}.txt").write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
                done += 1
            except Exception as e: errors.append(f"{jf.name}: {e}")
        (out / "classes.txt").write_text("\n".join(name for name,_ in sorted(class_map.items(), key=lambda kv: kv[1])) + "\n", encoding="utf-8")
        result.update(message=f"转换完成：{done} 个 JSON，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "生成 NPZ 名称列表":
        source = _need_dir(p["npz_dir"], "NPZ")
        out = _path(p["output_txt"]); out.parent.mkdir(parents=True, exist_ok=True)
        files = sorted(source.glob("*.npz"), key=lambda x: x.name.lower())
        out.write_text("\n".join(x.stem for x in files) + ("\n" if files else ""), encoding="utf-8")
        result.update(message=f"已写入 {len(files)} 个 NPZ 文件名。", output_file=str(out))
        return result

    if tool == "YOLO 数据集划分":
        from split_train_datas.split_yolo import split_yolo_dataset
        images = _need_dir(p["images_dir"], "图像")
        labels = _need_dir(p["labels_dir"], "标签")
        out = _out_dir(p["output_dir"])
        ratios = [float(p["train_ratio"]), float(p["val_ratio"]), float(p["test_ratio"])]
        if any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-6: raise ValueError("train/val/test 比例必须非负且总和为 1")
        split_yolo_dataset(str(images), str(labels), str(out), *ratios, int(p["seed"]))
        result.update(message="YOLO 数据集划分完成。", output_dir=str(out))
        return result

    if tool == "YOLO 模型性能分析":
        from model_process.yolo_per import profile_yolo
        weights = _need_file(p["weights"], "模型权重")
        profile_yolo(str(weights), int(p["image_size"]), p["device"])
        result.update(message="模型分析完成。详细统计结果已输出到运行日志。")
        return result

    if tool == "生成 Panoptic 数据集":
        from data_handle.goose_process.make_panoptic_mask_2_classmap_fixed import generate_panoptic_dataset
        image_dir = _need_dir(p["image_dir"], "图像")
        semseg_dir = _need_dir(p["semseg_dir"], "语义 Mask")
        inst_dir = _need_dir(p["instance_dir"], "实例 Mask")
        out = _out_dir(p["output_dir"])
        cat = _need_file(p["categories_json"], "类别映射 JSON")
        split_file = p["split_file"].strip() or None
        generate_panoptic_dataset(str(image_dir), str(semseg_dir), str(inst_dir), str(out), p["split_name"], split_file, str(cat), int(p["label_divisor"]), int(p["min_instance_area"]), bool(p["keep_semantic_only"]), int(p["ignore_label"]), bool(p["zero_is_background"]), int(p["vis_n"]))
        result.update(message="Panoptic 数据集生成完成。", output_dir=str(out))
        return result

    if tool == "文件扩展名 / 名称批量重命名":
        source = _need_dir(p["directory"], "目标")
        old_ext = p["old_ext"].strip().lower(); new_ext = p["new_ext"].strip().lower()
        old_ext = old_ext if old_ext.startswith(".") else "."+old_ext
        new_ext = new_ext if new_ext.startswith(".") else "."+new_ext
        mode = p["mode"]
        files = sorted((x for x in (source.rglob("*") if p["recursive"] else source.iterdir()) if x.is_file() and x.suffix.lower() == old_ext), key=lambda x: str(x).lower())
        plans = []
        stamp = datetime.now().strftime(p["time_format"])
        for i, f in enumerate(files, start=int(p["start_num"])):
            if mode == "none": new_name = f.stem + new_ext
            elif mode == "timestamp": new_name = f"{stamp}_{i:0{max(1,int(p['num_digits']) or 3)}d}{new_ext}"
            else: new_name = f"{p['base_name']}{i:0{max(1,int(p['num_digits']) or len(str(int(p['start_num'])+len(files)-1)))}d}{new_ext}"
            plans.append((f, f.with_name(new_name)))
        if not p["confirm"]:
            result.update(message="预览模式：未修改任何文件。勾选确认后再次运行。", table=[{"原文件":str(a),"目标文件":str(b)} for a,b in plans[:100]], matched=len(plans))
            return result
        done, errors = 0, []
        # 防止重名覆盖：所有目标先检查
        planned_targets = [b for _,b in plans]
        collisions = [b for b in planned_targets if b.exists() and b not in [a for a,_ in plans]]
        if collisions: raise FileExistsError("目标文件已存在，避免覆盖而停止：" + ", ".join(str(x) for x in collisions[:10]))
        for a,b in plans:
            if a == b: continue
            try: a.rename(b); done += 1
            except Exception as e: errors.append(f"{a}: {e}")
        result.update(message=f"重命名完成：{done} 个，错误 {len(errors)}。", errors=errors[:30])
        return result

    if tool == "批量删除文件名子串":
        source = _need_dir(p["directory"], "目标")
        substring = p["substring"]
        if not substring: raise ValueError("子串不能为空")
        files = [x for x in (source.rglob("*") if p["recursive"] else source.iterdir()) if x.is_file() and substring in x.name]
        plans = [(f, f.with_name(f.name.replace(substring, ""))) for f in files]
        if not p["confirm"]:
            result.update(message="预览模式：未修改文件。勾选确认后再次运行。", table=[{"原文件":str(a),"目标文件":str(b)} for a,b in plans[:100]], matched=len(plans))
            return result
        conflicts = [b for a,b in plans if b.exists() and a != b]
        if conflicts: raise FileExistsError("目标重名，已停止避免覆盖："+", ".join(str(x) for x in conflicts[:10]))
        done = 0
        for a,b in plans:
            if a != b: a.rename(b); done += 1
        result.update(message=f"完成：重命名 {done} 个文件。")
        return result

    if tool == "按文件名末字段分组":
        source = _need_dir(p["directory"], "源")
        target = _path(p["target_dir"]) if p["target_dir"].strip() else None
        keywords = set(_parse_labels(p["keywords"]))
        action = p["action"]
        if action in {"copy", "move"} and target is None: raise ValueError("复制/移动需要指定目标目录")
        files = [x for x in (source.rglob("*") if p["recursive"] else source.iterdir()) if x.is_file()]
        matches = [(f, f.stem.rsplit("_",1)[-1] if "_" in f.stem else "") for f in files]
        matches = [(f, k) for f,k in matches if k in keywords]
        if not p["confirm"]:
            result.update(message="预览模式：没有执行复制、移动或删除。勾选确认后再次运行。", table=[{"文件":str(f),"字段":k,"操作":action} for f,k in matches[:100]], matched=len(matches))
            return result
        if action == "delete" and not p["confirm_delete"]: raise ValueError("删除操作必须勾选独立的删除确认")
        errors=[]; done=0
        for f,k in matches:
            try:
                if action == "delete": f.unlink()
                else:
                    dest_dir = target / k; dest_dir.mkdir(parents=True, exist_ok=True); dest=dest_dir/f.name
                    if dest.exists():
                        if not p["overwrite"]: continue
                        dest.unlink()
                    if action == "move": shutil.move(str(f),str(dest))
                    else: shutil.copy2(f,dest)
                done += 1
            except Exception as e: errors.append(f"{f}: {e}")
        result.update(message=f"分组操作完成：{done} 个文件，错误 {len(errors)}。", errors=errors[:30])
        return result

    if tool == "清理缺少对应图像的标签":
        ref = _need_dir(p["reference_dir"], "参考")
        check = _need_dir(p["check_dir"], "待检查")
        action = p["action"]
        target = _path(p["target_dir"]) if p["target_dir"].strip() else None
        if action in {"move", "copy"} and target is None: raise ValueError("移动/复制需指定目标目录")
        exts = {x.lower() if x.startswith(".") else "."+x.lower() for x in _parse_labels(p["extensions"])} if p["extensions"].strip() else set()
        suffixes = tuple(x.strip() for x in p["strip_suffixes"].split(",") if x.strip())
        def mapped(stem):
            for suf in suffixes:
                if stem.endswith(suf): return stem[:-len(suf)]
            return stem
        ref_stems = {mapped(x.stem) for x in (ref.rglob("*") if p["recursive"] else ref.iterdir()) if x.is_file()}
        candidates = [x for x in (check.rglob("*") if p["recursive"] else check.iterdir()) if x.is_file() and (not exts or x.suffix.lower() in exts) and mapped(x.stem) not in ref_stems]
        if action == "delete" and not p["confirm_delete"]: 
            result.update(message="预览模式：没有删除文件；确认删除后再次运行。", table=[{"待处理文件":str(x)} for x in candidates[:100]], matched=len(candidates)); return result
        if not p["confirm"]:
            result.update(message="预览模式：没有执行操作。勾选确认后再次运行。", table=[{"待处理文件":str(x),"操作":action} for x in candidates[:100]], matched=len(candidates)); return result
        errors=[]; done=0
        for f in candidates:
            try:
                if action == "ignore": continue
                if action == "delete": f.unlink()
                else:
                    target.mkdir(parents=True, exist_ok=True); dest=target/f.name
                    if dest.exists(): dest=target/f"{f.stem}_dup{f.suffix}"
                    if action == "move": shutil.move(str(f),str(dest))
                    else: shutil.copy2(f,dest)
                done += 1
            except Exception as e: errors.append(f"{f}: {e}")
        result.update(message=f"清理完成：处理 {done} 个不匹配文件，候选 {len(candidates)}，错误 {len(errors)}。", errors=errors[:30])
        return result

    if tool == "删除中文文件名":
        source = _need_dir(p["directory"], "目标")
        files = [x for x in source.rglob("*") if x.is_file() and re.search(r"[\u4e00-\u9fff]", x.name)]
        if not p["confirm"]:
            result.update(message="预览模式：未删除文件。勾选确认删除后再次运行。", table=[{"待删除文件":str(x)} for x in files[:100]], matched=len(files))
            return result
        errors=[]; done=0
        for f in files:
            try: f.unlink(); done+=1
            except Exception as e: errors.append(f"{f}: {e}")
        result.update(message=f"删除完成：删除 {done} 个文件，错误 {len(errors)}。", errors=errors[:30])
        return result

    if tool == "批量修改文件名（去中文）":
        # 图像和标签使用同一个 stem 映射，保证同名配对关系不会被破坏。
        dirs = [_need_dir(p["image_dir"], "图像"), _need_dir(p["label_dir"], "标签")]
        prefix = p["prefix"].strip()
        files_by_dir = [[f for f in folder.iterdir() if f.is_file()] for folder in dirs]
        stems = sorted({f.stem for files in files_by_dir for f in files if re.search(r"[\u4e00-\u9fff]", f.stem)})
        if prefix:
            stem_map = {stem: f"{prefix}{i:04d}" for i, stem in enumerate(stems, 1)}
        else:
            stem_map = {stem: (re.sub(r"[\u4e00-\u9fff]", "", stem) or f"file_{i:04d}") for i, stem in enumerate(stems, 1)}
        plans = []
        for files in files_by_dir:
            for f in files:
                if f.stem in stem_map:
                    plans.append((f, f.with_name(stem_map[f.stem] + f.suffix)))
        targets = [b for _, b in plans]
        duplicate_targets = sorted({str(x) for x in targets if targets.count(x) > 1})
        if duplicate_targets:
            raise FileExistsError("多个文件将重命名为同一路径，已停止：" + ", ".join(duplicate_targets[:10]))
        conflicts = [b for a, b in plans if b.exists() and a != b and all(b != src for src, _ in plans)]
        if conflicts:
            raise FileExistsError("目标文件已存在，已停止：" + ", ".join(str(x) for x in conflicts[:10]))
        if not p["confirm"]:
            result.update(message="预览模式：未重命名。图像与标签使用同一 stem 映射；请检查预览后勾选确认。", table=[{"原文件":str(a),"目标文件":str(b)} for a,b in plans[:200]], matched=len(plans)); return result
        # 先统一改为唯一临时名，再改成目标名，允许文件间名称交换。
        staged = []
        try:
            for i, (src, dst) in enumerate(plans):
                if src == dst:
                    continue
                temp = src.with_name(f".__cvtool_tmp_{os.getpid()}_{i}_{src.name}")
                if temp.exists(): raise FileExistsError(f"临时文件名冲突：{temp}")
                src.rename(temp)
                staged.append((temp, dst))
            for temp, dst in staged:
                temp.rename(dst)
        except Exception:
            # 尝试回滚尚未落到目标名的临时文件，最大限度避免留下临时文件。
            for temp, dst in reversed(staged):
                if temp.exists():
                    original = next((src for src, target in plans if target == dst), None)
                    if original is not None and not original.exists():
                        try: temp.rename(original)
                        except Exception: pass
            raise
        result.update(message=f"重命名完成：{len(staged)} 个文件。共映射 {len(stem_map)} 个原始 stem，图像/标签配对保持一致。")
        return result

    if tool == "清除 TXT 首行":
        source = _need_dir(p["input_dir"], "输入")
        out = _out_dir(p["output_dir"])
        files = sorted(source.rglob("*.txt") if p["recursive"] else source.glob("*.txt"))
        done, errors = 0, []
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8-sig").splitlines()
                dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name)); dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("\n".join(lines[1:]) + ("\n" if len(lines)>1 else ""), encoding="utf-8")
                done += 1
            except Exception as e: errors.append(f"{f.name}: {e}")
        result.update(message=f"处理完成：{done} 个 TXT 输出到新目录，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "统一追加文件名前后缀":
        source = _need_dir(p["input_dir"], "输入")
        out = _out_dir(p["output_dir"])
        prefix, suffix = p["prefix"], p["suffix"]
        files = [x for x in (source.rglob("*") if p["recursive"] else source.iterdir()) if x.is_file()]
        done, errors = 0, []
        for f in files:
            dest = out / (f.relative_to(source) if p["recursive"] else Path(f.name))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest = dest.with_name(prefix + dest.stem + suffix + dest.suffix)
            try: shutil.copy2(f,dest); done += 1
            except Exception as e: errors.append(f"{f.name}: {e}")
        result.update(message=f"完成：复制并重命名 {done} 个文件，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "YOLO 数据集文件分类":
        # 兼容老版本图像标签混合目录分类，规则通过表单定义
        source = _need_dir(p["source_dir"], "源")
        rules = []
        for line in p["rules"].splitlines():
            if not line.strip(): continue
            if "=" not in line: raise ValueError(f"规则格式错误：{line}")
            folder, ext_text = line.split("=",1)
            exts = {x.strip().lower() if x.strip().startswith(".") else "."+x.strip().lower() for x in ext_text.split(",") if x.strip()}
            rules.append((folder.strip(), exts))
        if not rules: raise ValueError("请至少设置一条规则，例如 images=.jpg,.png")
        outroot = _path(p["output_root"]); outroot.mkdir(parents=True,exist_ok=True)
        files = [x for x in source.rglob("*") if x.is_file()]
        done, skipped, errors = 0,0,[]
        for f in files:
            rule = next((r for r in rules if f.suffix.lower() in r[1]),None)
            if not rule: skipped+=1; continue
            destdir=outroot/rule[0]; destdir.mkdir(parents=True,exist_ok=True); dest=destdir/f.name
            try:
                if dest.exists(): errors.append(f"目标已存在：{dest}"); continue
                if p["move"]:
                    if not p["confirm"]: continue
                    shutil.move(str(f),str(dest))
                else: shutil.copy2(f,dest)
                done+=1
            except Exception as e: errors.append(f"{f.name}: {e}")
        if p["move"] and not p["confirm"]:
            result.update(message=f"移动模式预览：共匹配 {len(files)-skipped} 个文件，未执行。勾选确认后再次运行。", matched=len(files)-skipped)
        else:
            result.update(message=f"分类完成：处理 {done}，未匹配 {skipped}，错误/冲突 {len(errors)}。", output_dir=str(outroot), errors=errors[:30])
        return result

    if tool == "拼接图像目录文件":
        # 对两个目录中的同名图像执行横向拼接；不依赖硬编码路径。
        import cv2
        a = _need_dir(p["left_dir"], "左图目录"); b = _need_dir(p["right_dir"], "右图目录"); out = _out_dir(p["output_dir"])
        from PIL import Image
        idx_b = {x.stem: x for x in _images(b, bool(p["recursive"]))}
        done, skipped, errors = 0,0,[]
        for f in _images(a, bool(p["recursive"])):
            mate=idx_b.get(f.stem)
            if mate is None: skipped+=1; continue
            ia,ib=_imread(f),_imread(mate)
            if ia is None or ib is None: errors.append(f"读取失败：{f.name}"); continue
            h=max(ia.shape[0],ib.shape[0]); ia=cv2.resize(ia,(round(ia.shape[1]*h/ia.shape[0]),h)); ib=cv2.resize(ib,(round(ib.shape[1]*h/ib.shape[0]),h))
            dest=out/f.name
            if _imwrite(dest,cv2.hconcat([ia,ib])): done+=1
        result.update(message=f"目录拼接完成：{done} 张，缺少匹配 {skipped}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "重命名 JSON 对应图像文件名":
        source = _need_dir(p["json_dir"], "JSON")
        images = _need_dir(p["image_dir"], "图像")
        mode = p["mode"]
        rows=[]; errors=[]
        for jf in _json_files(source, bool(p["recursive"])):
            try:
                data=_load_json(jf); ip=data.get("imagePath",""); name=Path(ip).name if ip else ""
                candidates=[x for x in images.iterdir() if x.is_file() and x.stem==jf.stem and x.suffix.lower() in IMAGE_EXTS]
                expected=candidates[0].name if candidates else ""
                if expected and name != expected: rows.append({"JSON":str(jf),"原 imagePath":name,"建议 imagePath":expected})
            except Exception as e: errors.append(f"{jf.name}: {e}")
        if mode == "预览" or not p["confirm"]:
            result.update(message=f"检查完成：发现 {len(rows)} 个 imagePath 不一致。当前为预览模式，未修改 JSON。", table=rows[:200], errors=errors[:30]); return result
        # 执行写回前，确认仅修正同 stem 的 imagePath 字段。
        fixed=0
        for row in rows:
            jf=Path(row["JSON"]); d=_load_json(jf); d["imagePath"]=row["建议 imagePath"]; _write_json(jf,d); fixed+=1
        result.update(message=f"修正完成：更新 {fixed} 个 JSON 的 imagePath。", errors=errors[:30])
        return result

    if tool == "语义分割数据增强":
        image_dir = _need_dir(p["image_dir"], "图像")
        mask_dir = _need_dir(p["mask_dir"], "Mask")
        if int(p["aug_times"]) < 1 or int(p["workers"]) < 1: raise ValueError("增强倍数和线程数必须大于 0")
        script = ROOT / "data_handle" / "data_augmentation-seg.py"
        cmd = [sys.executable, str(script), "--image_dir", str(image_dir), "--mask_dir", str(mask_dir), "--aug_times", str(int(p["aug_times"])), "--num_workers", str(int(p["workers"]))]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode: raise RuntimeError(f"增强脚本退出码 {proc.returncode}。\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}")
        result.update(message="语义分割增强完成。", output_dir=str(image_dir.parent / f"aug_{image_dir.stem}"), log=(proc.stdout+"\n"+proc.stderr)[-20000:])
        return result

    if tool == "OBB 数据增强":
        image_dir = _need_dir(p["image_dir"], "图像")
        label_dir = _need_dir(p["label_dir"], "标签")
        if int(p["aug_times"]) < 1 or int(p["workers"]) < 1: raise ValueError("增强倍数和线程数必须大于 0")
        script = ROOT / "data_handle" / "data_augmentation-obb.py"
        cmd = [sys.executable, str(script), "--image_dir", str(image_dir), "--label_dir", str(label_dir), "--label_ext", p["label_ext"], "--aug_times", str(int(p["aug_times"])), "--num_workers", str(int(p["workers"]))]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode: raise RuntimeError(f"OBB 增强脚本退出码 {proc.returncode}。\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}")
        result.update(message="OBB 增强完成。", output_dir=str(image_dir.parent / "aug_images"), log=(proc.stdout+"\n"+proc.stderr)[-20000:])
        return result

    if tool == "LabelMe JSON 转 YOLO TXT":
        json_dir = _need_dir(p["json_dir"], "LabelMe JSON")
        out = _out_dir(p["output_dir"])
        classes = _parse_labels(p["classes"])
        if not classes: raise ValueError("请按类别 ID 顺序填写类别名，例如 cable,white_cable")
        class_to_id = {name:i for i,name in enumerate(classes)}
        files = _json_files(json_dir, bool(p["recursive"]))
        done, errors = 0, []
        for jf in files:
            try:
                data = _load_json(jf); w,h=int(data.get("imageWidth",0)),int(data.get("imageHeight",0))
                if w<=0 or h<=0: raise ValueError("缺少有效 imageWidth/imageHeight")
                lines=[]
                for sh in data.get("shapes",[]):
                    name=sh.get("label"); pts=sh.get("points",[]); typ=sh.get("shape_type","polygon")
                    if name not in class_to_id or len(pts)<2: continue
                    if typ == "rectangle" or len(pts)==2:
                        xs=[float(q[0]) for q in pts]; ys=[float(q[1]) for q in pts]
                        x1,x2,y1,y2=min(xs),max(xs),min(ys),max(ys)
                        lines.append(f"{class_to_id[name]} {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                    else:
                        coords=[]
                        for x,y in pts: coords.extend([float(x)/w,float(y)/h])
                        lines.append(str(class_to_id[name])+" "+" ".join(f"{v:.6f}" for v in coords))
                dest=out/f"{jf.stem}.txt"; dest.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8"); done+=1
            except Exception as e: errors.append(f"{jf.name}: {e}")
        (out/"classes.txt").write_text("\n".join(classes)+"\n",encoding="utf-8")
        result.update(message=f"转换完成：{done}/{len(files)} 个 JSON，错误 {len(errors)}。注意：polygon 输出是 YOLO-seg 格式。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "YOLO TXT 转 LabelMe JSON":
        txt_dir = _need_dir(p["txt_dir"], "YOLO TXT")
        image_dir = _need_dir(p["image_dir"], "图像")
        out = _out_dir(p["output_dir"])
        classes = _parse_labels(p["classes"])
        if not classes: raise ValueError("请按类别 ID 顺序填写类别名")
        imgs = {x.stem:x for x in _images(image_dir, bool(p["recursive"]))}
        done, errors, missing = 0, [], 0
        for tf in sorted(txt_dir.rglob("*.txt") if p["recursive"] else txt_dir.glob("*.txt")):
            try:
                imgp=imgs.get(tf.stem)
                if imgp is None: missing+=1; continue
                import cv2
                img=_imread(imgp)
                if img is None: raise ValueError(f"无法读取对应图像 {imgp.name}")
                h,w=img.shape[:2]; shapes=[]
                for line in tf.read_text(encoding="utf-8-sig").splitlines():
                    parts=line.strip().split()
                    if len(parts)<5: continue
                    vals=[float(x) for x in parts]; cid=int(vals[0])
                    if cid<0 or cid>=len(classes): continue
                    if len(vals)==5:
                        _,xc,yc,bw,bh=vals
                        pts=[[max(0,(xc-bw/2)*w),max(0,(yc-bh/2)*h)],[min(w,(xc+bw/2)*w),min(h,(yc+bh/2)*h)]]
                        typ="rectangle"
                    else:
                        coords=vals[1:]
                        if len(coords)<6 or len(coords)%2: continue
                        pts=[[coords[i]*w,coords[i+1]*h] for i in range(0,len(coords),2)]
                        typ="polygon"
                    shapes.append({"label":classes[cid],"points":pts,"group_id":None,"description":"","shape_type":typ,"flags":{}})
                data={"version":"5.2.1","flags":{},"shapes":shapes,"imagePath":imgp.name,"imageData":None,"imageHeight":h,"imageWidth":w}
                _write_json(out/f"{tf.stem}.json",data); done+=1
            except Exception as e: errors.append(f"{tf.name}: {e}")
        result.update(message=f"转换完成：生成 {done} 个 JSON，缺少对应图像 {missing}，错误 {len(errors)}。", output_dir=str(out), errors=errors[:30])
        return result

    if tool == "YOLO 数据集转 COCO":
        root = _need_dir(p["root_dir"], "YOLO 数据集根目录")
        images_dir=_need_dir(str(root/"images"),"YOLO images")
        labels_dir=_need_dir(str(root/"labels"),"YOLO labels")
        classes_file=_need_file(str(root/"classes.txt"),"classes.txt")
        classes=[x.strip() for x in classes_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        mode=p["split_mode"]
        if mode == "按 train.txt/val.txt/test.txt":
            for name in ("train.txt","val.txt","test.txt"):
                _need_file(str(root/name),name)
        script=ROOT/"data_handle"/"yolodatas2cocodatas.py"
        save=p["save_path"].strip() or "_annotations.coco.json"
        cmd=[sys.executable,str(script),"--root_dir",str(root),"--save_path",save]
        if mode=="随机划分 8:1:1": cmd.append("--random_split")
        elif mode=="按 train.txt/val.txt/test.txt": cmd.append("--split_by_file")
        proc=subprocess.run(cmd,cwd=str(ROOT),capture_output=True,text=True,encoding="utf-8",errors="replace")
        if proc.returncode: raise RuntimeError(f"YOLO 转 COCO 失败。\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}")
        result.update(message="YOLO 数据集转 COCO 完成。",output_dir=str(root/"annotations"),log=(proc.stdout+"\n"+proc.stderr)[-20000:])
        return result

    if tool == "删除空标签 TXT 文件":
        folder = _need_dir(p["txt_dir"], "TXT 标签")
        files = sorted(folder.rglob("*.txt") if p["recursive"] else folder.glob("*.txt"))
        empty=[]
        for f in files:
            try:
                if not f.read_text(encoding="utf-8-sig",errors="replace").strip(): empty.append(f)
            except Exception: pass
        if not p["confirm"]:
            result.update(message="预览模式：未删除文件。确认删除后再次运行。",table=[{"待删除空标签":str(x)} for x in empty[:200]],matched=len(empty)); return result
        errors=[]; done=0
        for f in empty:
            try: f.unlink(); done+=1
            except Exception as e: errors.append(f"{f}: {e}")
        result.update(message=f"空标签清理完成：删除 {done} 个，错误 {len(errors)}。",errors=errors[:30])
        return result

    if tool == "实例 Mask 转 COCO Instances":
        image_dir=_need_dir(p["image_dir"],"图像")
        instance_dir=_need_dir(p["instance_dir"],"实例 Mask")
        semantic_dir=_need_dir(p["semantic_dir"],"语义 Mask")
        categories_path=_need_file(p["categories_json"],"类别定义 JSON")
        output_json=_path(p["output_json"])
        output_json.parent.mkdir(parents=True,exist_ok=True)
        categories_raw=json.loads(categories_path.read_text(encoding="utf-8-sig"))
        categories_raw=categories_raw.get("categories",categories_raw) if isinstance(categories_raw,dict) else categories_raw
        categories=[c for c in categories_raw if isinstance(c,dict)]
        thing_ids={int(c["id"]) for c in categories if int(c.get("isthing",0))==1}
        cat_coco=[{"id":int(c["id"]),"name":str(c["name"]),"supercategory":"thing"} for c in categories if int(c.get("isthing",0))==1]
        if not cat_coco: raise ValueError("类别 JSON 必须包含类别列表，且 thing 类含 id/name/isthing=1")
        import cv2
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mask_utils
        inst_map={f.stem:f for f in instance_dir.glob("*.png")}; sem_map={f.stem:f for f in semantic_dir.glob("*.png")}
        images=[]; annotations=[]; ann_id=1; missing=0; mismatches=[]
        image_files=_images(image_dir,bool(p["recursive"]))
        min_area=int(p["min_area"]); min_dom=float(p["min_dominance"])
        for image_id,imgp in enumerate(image_files,start=1):
            ip=inst_map.get(imgp.stem); sp=sem_map.get(imgp.stem)
            if ip is None or sp is None: missing+=1; continue
            image=Image.open(imgp); w,h=image.size
            inst=np.asarray(Image.open(ip)); sem=np.asarray(Image.open(sp))
            if inst.ndim==3: inst=inst[:,:,0]
            if sem.ndim==3: sem=sem[:,:,0]
            if inst.shape!=sem.shape: mismatches.append(f"{imgp.name}: Mask 尺寸不一致"); continue
            images.append({"id":image_id,"width":w,"height":h,"file_name":imgp.name})
            uniq=np.unique(inst)
            if len(uniq)<=2 and set(int(x) for x in uniq).issubset({0,1}):
                n,labels=cv2.connectedComponents((inst>0).astype(np.uint8),connectivity=8)
                components=[labels==i for i in range(1,n)]
            else:
                components=[inst==v for v in uniq if int(v)!=0]
            for comp in components:
                area=int(comp.sum())
                if area<min_area: continue
                vals=sem[comp].astype(np.int64); vals=vals[vals>=0]
                if not vals.size: continue
                values,counts=np.unique(vals,return_counts=True); index=int(np.argmax(counts)); catid=int(values[index]); dominant=float(counts[index])/float(vals.size)
                if catid not in thing_ids or dominant<min_dom: continue
                ys,xs=np.where(comp)
                if not len(xs): continue
                rle=mask_utils.encode(np.asfortranarray(comp.astype(np.uint8))); rle["counts"]=rle["counts"].decode("utf-8")
                annotations.append({"id":ann_id,"image_id":image_id,"category_id":catid,"segmentation":rle,"area":area,"bbox":[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],"iscrowd":0}); ann_id+=1
        _write_json(output_json,{"images":images,"annotations":annotations,"categories":cat_coco},indent=2)
        result.update(message=f"COCO instances 生成完成：图像 {len(images)}，实例标注 {len(annotations)}，缺少匹配 Mask {missing}。",output_file=str(output_json),errors=mismatches[:30])
        return result

    if tool == "语义+实例 Mask 生成 Panoptic":
        image_dir=_need_dir(p["image_dir"],"图像"); semantic_dir=_need_dir(p["semantic_dir"],"语义 Mask"); instance_dir=_need_dir(p["instance_dir"],"实例 Mask")
        out=_out_dir(p["panoptic_dir"]); output_json=_path(p["output_json"]); output_json.parent.mkdir(parents=True,exist_ok=True)
        cat_path=_need_file(p["categories_json"],"类别定义 JSON")
        raw=json.loads(cat_path.read_text(encoding="utf-8-sig")); cats=raw.get("categories",raw) if isinstance(raw,dict) else raw
        cat_map={int(c["id"]):c for c in cats}; semmap={x.stem:x for x in semantic_dir.glob("*.png")}; instmap={x.stem:x for x in instance_dir.glob("*.png")}
        import numpy as np
        import cv2
        from PIL import Image
        images=[]; annotations=[]; segment_id=1; missing=0; errors=[]
        for image_id,imgp in enumerate(_images(image_dir,bool(p["recursive"])),start=1):
            sp=semmap.get(imgp.stem); ip=instmap.get(imgp.stem)
            if sp is None or ip is None: missing+=1; continue
            sem=np.asarray(Image.open(sp)); inst=np.asarray(Image.open(ip))
            if sem.ndim==3: sem=sem[:,:,0]
            if inst.ndim==3: inst=inst[:,:,0]
            if sem.shape!=inst.shape: errors.append(f"{imgp.name}: Mask 尺寸不一致"); continue
            h,w=sem.shape; rgb=np.zeros((h,w,3),np.uint8); segs=[]
            for class_id in np.unique(sem):
                class_id=int(class_id)
                if class_id not in cat_map or (class_id==int(p["ignore_label"])): continue
                cat=cat_map[class_id]; cmask=(sem==class_id)
                if int(cat.get("isthing",0))==1:
                    masks=[cmask & (inst==int(i)) for i in np.unique(inst[cmask]) if int(i)!=0]
                else:
                    n,cc=cv2.connectedComponents(cmask.astype(np.uint8),connectivity=8); masks=[cc==i for i in range(1,n)]
                for mask in masks:
                    area=int(mask.sum())
                    if area<int(p["min_area"]): continue
                    ys,xs=np.where(mask); sid=segment_id; segment_id+=1
                    rgb[mask,0]=sid%256; rgb[mask,1]=(sid//256)%256; rgb[mask,2]=(sid//65536)%256
                    segs.append({"id":sid,"category_id":class_id,"area":area,"bbox":[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],"iscrowd":0})
            pan_name=f"{imgp.stem}.png"; Image.fromarray(rgb).save(out/pan_name)
            images.append({"id":image_id,"file_name":imgp.name,"width":w,"height":h}); annotations.append({"image_id":image_id,"file_name":pan_name,"segments_info":segs})
        _write_json(output_json,{"images":images,"annotations":annotations,"categories":cats})
        result.update(message=f"Panoptic 生成完成：图像 {len(images)}，缺少 Mask {missing}，尺寸错误 {len(errors)}。",output_dir=str(out),output_file=str(output_json),errors=errors[:30])
        return result

    if tool == "语义类别重映射":
        import cv2
        import numpy as np
        src=_need_dir(p["input_dir"],"语义 Mask"); out=_out_dir(p["output_dir"])
        stuff=[int(x) for x in _parse_labels(p["stuff_ids"])]; things={int(x) for x in _parse_labels(p["thing_ids"])}
        ignore=int(p["ignore_label"]); mapping={old:new for new,old in enumerate(stuff)}
        if set(stuff)&things: raise ValueError("stuff_ids 与 thing_ids 不应交叉")
        done=0; errors=[]
        for f in _images(src,bool(p["recursive"])):
            mask=_imread(f,cv2.IMREAD_UNCHANGED)
            if mask is None: errors.append(f"读取失败：{f.name}"); continue
            outmask=np.full(mask.shape,ignore,dtype=np.uint8)
            for old,new in mapping.items(): outmask[mask==old]=new
            # things 保持 ignore_label，其他未知值也保持 ignore_label
            dest=out/(f.relative_to(src) if p["recursive"] else Path(f.name)); _imwrite(dest,outmask); done+=1
        result.update(message=f"语义类别映射完成：{done} 个 Mask，stuff 类别 {len(mapping)} 个，thing 类统一设为 {ignore}。",output_dir=str(out),errors=errors[:30])
        return result

    if tool == "工具脚本目录":
        all_py = sorted(ROOT.rglob("*.py"))
        rows=[]
        for f in all_py:
            if any(part in {".git", "__pycache__", ".venv"} for part in f.parts): continue
            rel=str(f.relative_to(ROOT)).replace("\\", "/")
            first=""
            try:
                for line in f.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:10]:
                    x=line.strip().strip('"# ')
                    if x and not x.startswith(("import ","from ","#", "def ")):
                        first=x[:110]; break
            except Exception: pass
            hint = SCRIPT_TOOL_HINTS.get(rel)
            rows.append({"脚本路径":rel,"说明":first,"网页入口":f"对应工具：{hint}" if hint else "原始辅助/实验脚本，保留源码"})
        result.update(message=f"源码目录包含 {len(rows)} 个 Python 文件。没有自动运行未知脚本，以免其硬编码路径触发误操作。", table=rows)
        return result

    if tool == "Panoptic 标签检查":
        import cv2
        import numpy as np
        from PIL import Image
        pan_json = _need_file(p["panoptic_json"], "Panoptic JSON")
        root = _need_dir(p["panoptic_root"], "Panoptic mask")
        data = json.loads(pan_json.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
            raise ValueError("Panoptic JSON 格式错误：缺少 annotations 列表")
        images_by_id = {x.get("id"): x for x in data.get("images", [])}
        categories = {int(c["id"]): c for c in data.get("categories", []) if "id" in c}
        rows, all_errors, all_warnings = [], [], []
        category_counts = Counter()
        checked, missing_masks = 0, 0
        for ann in data["annotations"]:
            fname = str(ann.get("file_name", ""))
            if not fname:
                all_errors.append("annotation 缺少 file_name"); continue
            mask_path = (root / fname).resolve()
            try:
                mask_path.relative_to(root.resolve())
            except ValueError:
                all_errors.append(f"{fname}: mask 路径越界"); continue
            if not mask_path.is_file():
                # 容忍 JSON 只记录 basename 的数据集布局
                fallback = root / Path(fname).name
                if fallback.is_file(): mask_path = fallback
                else:
                    missing_masks += 1; all_errors.append(f"{fname}: 缺少 Panoptic mask"); continue
            try:
                rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint32)
            except Exception as exc:
                all_errors.append(f"{fname}: 读取失败：{exc}"); continue
            ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
            actual_ids = {int(x) for x in np.unique(ids) if int(x) != 0}
            declared_ids, seen_ids = set(), set()
            image_errors, image_warnings = [], []
            image_info = images_by_id.get(ann.get("image_id"), {})
            if image_info and (int(image_info.get("width", rgb.shape[1])) != rgb.shape[1] or int(image_info.get("height", rgb.shape[0])) != rgb.shape[0]):
                image_errors.append(f"Mask 尺寸 {rgb.shape[1]}x{rgb.shape[0]} 与 images 尺寸不一致")
            for seg in ann.get("segments_info", []):
                try:
                    sid = int(seg["id"])
                    cid = int(seg["category_id"])
                except Exception:
                    image_errors.append(f"segments_info 存在无效 id/category_id：{seg}"); continue
                if sid in seen_ids: image_errors.append(f"JSON 中 segment id 重复：{sid}")
                seen_ids.add(sid); declared_ids.add(sid)
                if cid not in categories:
                    image_errors.append(f"segment {sid} 使用未知 category_id={cid}")
                else:
                    category_counts[(cid, str(categories[cid].get("name", cid)))] += 1
                seg_mask = ids == sid
                if not np.any(seg_mask):
                    image_errors.append(f"JSON 声明的 segment id={sid} 不存在于 PNG"); continue
                area = int(np.count_nonzero(seg_mask))
                if int(seg.get("area", -1)) != area:
                    image_errors.append(f"segment {sid} area 不一致：JSON={seg.get('area')}，PNG={area}")
                ys, xs = np.where(seg_mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
                try: declared_bbox = [int(round(float(v))) for v in seg.get("bbox", [])]
                except Exception: declared_bbox = []
                if declared_bbox != bbox:
                    image_warnings.append(f"segment {sid} bbox 不一致：JSON={seg.get('bbox')}，PNG={bbox}")
                x, y, bw, bh = bbox
                if x < 0 or y < 0 or x + bw > rgb.shape[1] or y + bh > rgb.shape[0]:
                    image_errors.append(f"segment {sid} bbox 超出图像范围")
                if cid in categories and bool(categories[cid].get("isthing", 0)):
                    cc_count = int(cv2.connectedComponents(seg_mask.astype(np.uint8), connectivity=8)[0] - 1)
                    if cc_count > 1:
                        image_warnings.append(f"thing segment {sid} 分裂为 {cc_count} 个连通域")
            extra_ids, missing_ids = actual_ids - declared_ids, declared_ids - actual_ids
            if extra_ids: image_errors.append(f"PNG 存在 JSON 未声明的 ID：{sorted(extra_ids)[:12]}")
            if missing_ids: image_errors.append(f"JSON ID 在 PNG 缺失：{sorted(missing_ids)[:12]}")
            checked += 1
            all_errors.extend(f"{fname}: {e}" for e in image_errors)
            all_warnings.extend(f"{fname}: {w}" for w in image_warnings)
            rows.append({"Panoptic Mask": str(mask_path.relative_to(root.resolve())).replace("\\", "/"), "PNG segment 数": len(actual_ids), "JSON segment 数": len(declared_ids), "错误数": len(image_errors), "警告数": len(image_warnings), "状态": "异常" if image_errors else ("警告" if image_warnings else "通过"), "错误摘要": "；".join(image_errors[:3]), "警告摘要": "；".join(image_warnings[:3])})
        category_rows = [{"category_id": cid, "类别": name, "segment 数": n} for (cid, name), n in sorted(category_counts.items())]
        result.update(
            message=f"Panoptic 检查完成：成功检查 {checked} 张，缺失 Mask {missing_masks} 张，错误 {len(all_errors)} 条，警告 {len(all_warnings)} 条。",
            table=rows,
            errors=(all_errors + ["[警告] " + x for x in all_warnings])[:100],
            counts={"检查图像数": checked, "缺失 Mask 数": missing_masks, "错误数": len(all_errors), "警告数": len(all_warnings), "类别统计": category_rows},
        )
        return result

    if tool == "Kaggle 数据集下载":
        import kagglehub
        dataset = p["dataset_id"].strip()
        if not dataset: raise ValueError("请输入 Kaggle 数据集 ID，例如 owner/dataset-name")
        path = kagglehub.dataset_download(dataset)
        result.update(message="Kaggle 数据集下载完成。", output_dir=str(path))
        return result

    raise ValueError(f"未实现的工具：{tool}")


# ---------- UI schema ----------
COMMON = {"recursive": ("checkbox", False, "递归子目录")}
FIELDS: dict[str, list[tuple]] = {
    "视频抽帧": [("video_file","path","视频文件路径", ""),("output_dir","path","输出目录", ""),("frame_interval","int","每隔多少帧保存一张", 25)],
    "图像拼接": [("image_paths","text","图像路径（英文逗号分隔）", ""),("output_file","path","输出图像文件", "")],
    "图像等比例缩放": [("input_dir","path","输入图像目录", ""),("output_dir","path","输出目录", ""),("target_size","int","目标宽/高像素",1024),("by_width","checkbox","按宽度缩放（不勾选则按高度）",True),("recursive","checkbox","递归子目录",False)],
    "Mask 伪彩色可视化": [("mask_dir","path","Mask 目录", ""),("output_dir","path","输出目录", ""),("recursive","checkbox","递归子目录",False)],
    "根据 Mask 提取图像区域": [("image_dir","path","原图目录", ""),("mask_dir","path","Mask 目录", ""),("output_dir","path","输出目录", ""),("labels","text","保留类别 ID（逗号分隔）","1,2"),("background","select","背景模式",["transparent","black","white"]),("output_ext","select","输出扩展名",[".png",".tif"]),("mask_ext","text","Mask 扩展名（留空自动匹配）","" )],
    "YOLO / LabelMe 标注可视化": [("image_dir","path","图像目录", ""),("label_dir","path","标签目录", ""),("output_dir","path","输出目录", ""),("classes_file","path","classes.txt 路径（可空）", ""),("save_undetected","checkbox","无标签图像也保存",False),("extensions","text","图像扩展名（逗号分隔）",".jpg,.jpeg,.png,.bmp"),("workers","int","线程数",8)],
    "OBB 标注可视化": [("image_dir","path","图像目录", ""),("label_dir","path","标签目录", ""),("output_dir","path","输出目录", ""),("label_ext","select","标签格式",[".json",".txt"]),("recursive","checkbox","递归子目录",False)],
    "双模型 OBB 结果对比": [("json1","path","模型 1 COCO OBB JSON", ""),("json2","path","模型 2 COCO OBB JSON", ""),("image_dir","path","原图目录", ""),("output_dir","path","输出目录", ""),("category_map","textarea","类别映射（每行 id=name）","1=class_1\n2=class_2"),("model1_name","text","模型 1 名称","model_A"),("color1","text","模型 1 BGR 颜色，例如 255,0,0","255,0,0"),("model2_name","text","模型 2 名称","model_B"),("color2","text","模型 2 BGR 颜色，例如 0,0,255","0,0,255")],
    "语义分割数据增强": [("image_dir","path","图像目录", ""),("mask_dir","path","Mask 目录", ""),("aug_times","int","每张图像增强次数",2),("workers","int","线程数",8)],
    "OBB 数据增强": [("image_dir","path","图像目录", ""),("label_dir","path","OBB 标签目录", ""),("label_ext","select","标签格式",[".json",".txt"]),("aug_times","int","每张图像增强次数",2),("workers","int","线程数",8)],
    "批量图像分类": [("source_dir","path","源目录", ""),("rules","textarea","分类规则（每行：类别名=.jpg,.png）","images=.jpg,.jpeg,.png\njson_labels=.json\ntxt_labels=.txt"),("move","checkbox","移动而非复制（不可撤销）",False),("confirm","checkbox","确认移动操作",False)],
    "LabelMe JSON 生成 Mask": [("image_dir","path","图像目录", ""),("label_dir","path","LabelMe JSON 目录", ""),("output_dir","path","Mask 输出目录", ""),("class_map","textarea","类别映射（每行 name=id）","background=0\ncable=1"),("generate_empty","checkbox","缺少 JSON 时生成全黑 Mask",False),("recursive","checkbox","递归图像子目录",False)],
    "Mask 转 LabelMe JSON": [("mask_dir","path","Mask 目录", ""),("image_dir","path","对应图像目录", ""),("output_dir","path","JSON 输出目录", ""),("min_area","float","最小轮廓面积",10.0),("epsilon_ratio","float","轮廓简化比例",0.002),("workers","int","线程数",8),("bridge_width","int","孔洞桥接宽度",2),("bridge_extension","float","桥接延伸像素",2.0)],
    "Mask 像素值映射": [("input_dir","path","输入 Mask 目录", ""),("output_dir","path","输出目录", ""),("mapping","text","映射 old_id=new_id", "1=2,255=0"),("recursive","checkbox","递归子目录",False)],
    "统计 Mask 像素类别": [("mask_dir","path","Mask 目录", ""),("recursive","checkbox","递归子目录",False)],
    "Mask 二值化处理": [("input_dir","path","输入 Mask 目录", ""),("output_dir","path","输出目录", ""),("recursive","checkbox","递归子目录",False)],
    "Mask 转 NPZ 数据集": [("image_dir","path","图像目录", ""),("mask_dir","path","Mask 目录", ""),("output_dir","path","NPZ 输出目录", ""),("mask_ext","text","Mask 扩展名", ".png"),("recursive","checkbox","递归子目录",False)],
    "统计 JSON 标注类别": [("json_dir","path","LabelMe JSON 目录", ""),("recursive","checkbox","递归子目录",False)],
    "LabelMe JSON 转 YOLO TXT": [("json_dir","path","LabelMe JSON 目录", ""),("output_dir","path","YOLO TXT 输出目录", ""),("classes","text","类别名（按 ID 顺序，逗号分隔）","cable,white_cable"),("recursive","checkbox","递归子目录",False)],
    "YOLO TXT 转 LabelMe JSON": [("txt_dir","path","YOLO TXT 标签目录", ""),("image_dir","path","对应图像目录", ""),("output_dir","path","LabelMe JSON 输出目录", ""),("classes","text","类别名（按 ID 顺序，逗号分隔）","cable"),("recursive","checkbox","递归子目录",False)],
    "删除 JSON 指定类别": [("input_dir","path","输入 JSON 目录", ""),("output_dir","path","输出目录（不会覆盖原文件）", ""),("labels","text","要删除的 label（逗号分隔）", ""),("recursive","checkbox","递归子目录",False),("indent","int","JSON 缩进",2)],
    "合并 JSON 指定类别": [("base_dir","path","基础 JSON 目录", ""),("add_dir","path","追加 JSON 目录", ""),("output_dir","path","输出目录（非覆盖模式）", ""),("labels","text","要追加的 label（逗号分隔）", ""),("label_field","text","类别字段名","label"),("indent","int","JSON 缩进",2),("overwrite","checkbox","覆盖基础目录的同名 JSON",False),("confirm","checkbox","确认覆盖基础 JSON（仅覆盖模式需要）",False)],
    "仅保留 JSON 指定类别": [("input_dir","path","输入 JSON 目录", ""),("output_dir","path","输出目录", ""),("labels","text","要保留的 label（逗号分隔）", ""),("recursive","checkbox","递归子目录",False),("indent","int","JSON 缩进",2)],
    "COCO JSON 转 LabelMe": [("coco_json","path","COCO JSON 文件", ""),("output_dir","path","输出目录", ""),("label_name","text","输出 LabelMe 统一类别名","object")],
    "LabelMe OBB 转 COCO": [("json_dir","path","LabelMe JSON 目录", ""),("image_dir","path","图像目录", ""),("classes_file","path","classes.txt 路径", ""),("output_json","path","输出 COCO JSON 文件", ""),("recursive","checkbox","递归子目录",False)],
    "VOC XML 转 YOLO": [("xml_dir","path","VOC XML 目录", ""),("output_dir","path","YOLO TXT 输出目录", ""),("classes","text","类别名（按 ID 顺序，逗号分隔）", "person,hat")],
    "VOC XML 转 COCO": [("xml_dir","path","VOC XML 目录", ""),("output_json","path","输出 COCO JSON 文件", ""),("classes","text","类别名（逗号分隔）", "person,hat")],
    "LabelMe OBB 转 YOLO HBB": [("json_dir","path","LabelMe JSON 目录", ""),("output_dir","path","TXT 输出目录", ""),("class_map","textarea","类别映射（每行 name=id）", "class_a=0\nclass_b=1"),("recursive","checkbox","递归子目录",False)],
    "像素框 TXT 转 YOLO HBB": [("txt_dir","path","像素坐标 TXT 目录", ""),("image_dir","path","原图目录", ""),("output_dir","path","YOLO TXT 输出目录", ""),("recursive","checkbox","递归子目录",False)],
    "YOLO TXT 类别映射/筛选": [("input_dir","path","YOLO TXT 输入目录", ""),("output_dir","path","输出目录", ""),("mode","select","处理模式",["映射类别 ID","仅保留指定类别","删除指定类别"]),("mapping","text","类别 ID 映射 old=new（映射模式使用）", "2=0,3=1,4=2"),("class_ids","text","类别 ID 列表（逗号分隔）", "0,1"),("keep_unmapped","checkbox","映射模式保留未映射类别",True),("drop_empty","checkbox","过滤后为空时不输出该 TXT",False),("recursive","checkbox","递归子目录",False)],
    "LabelMe JSON 类别映射与 imagePath 修复": [("json_dir","path","LabelMe JSON 输入目录", ""),("image_dir","path","真实图像目录", ""),("output_dir","path","修正后 JSON 输出目录", ""),("label_map","textarea","类别映射（每行 old_label=new_label）", "lead_wire=conductor_wire"),("recursive","checkbox","递归 JSON 子目录",False)],
    "COCO JSON 清理无效图像": [("coco_json","path","COCO JSON 文件", ""),("image_dir","path","图像根目录", ""),("output_json","path","清理后 COCO JSON 输出路径", "")],
    "YOLO 数据集转 COCO": [("root_dir","path","YOLO 数据集根目录（含 images/labels/classes.txt）", ""),("save_path","text","COCO 输出文件名（保存在 annotations/ 下）","_annotations.coco.json"),("split_mode","select","数据集输出方式",["单个 COCO JSON","随机划分 8:1:1","按 train.txt/val.txt/test.txt"])],
    "生成 NPZ 名称列表": [("npz_dir","path","NPZ 文件夹", ""),("output_txt","path","输出 TXT 文件", "")],
    "YOLO 数据集划分": [("images_dir","path","图像目录", ""),("labels_dir","path","标签目录", ""),("output_dir","path","输出数据集目录", ""),("train_ratio","float","Train 比例",0.7),("val_ratio","float","Val 比例",0.2),("test_ratio","float","Test 比例",0.1),("seed","int","随机种子",42)],
    "YOLO 模型性能分析": [("weights","path","YOLO 权重路径", ""),("image_size","int","输入尺寸",640),("device","select","设备",["cuda:0","cpu"])],
    "生成 Panoptic 数据集": [("image_dir","path","图像目录", ""),("semseg_dir","path","语义 Mask 目录", ""),("instance_dir","path","实例 Mask 目录", ""),("output_dir","path","Panoptic 输出根目录", ""),("categories_json","path","类别映射 JSON", ""),("split_name","text","数据集 split 名称","train"),("split_file","path","样本清单（可空）",""),("label_divisor","int","label_divisor",1000),("min_instance_area","int","实例最小面积",0),("keep_semantic_only","checkbox","保留纯语义区域",True),("ignore_label","int","忽略类别 ID",255),("zero_is_background","checkbox","0 表示背景",False),("vis_n","int","生成可视化数量",10)],
    "实例 Mask 转 COCO Instances": [("image_dir","path","原图目录", ""),("instance_dir","path","实例 Mask 目录", ""),("semantic_dir","path","语义 Mask 目录", ""),("categories_json","path","类别定义 JSON（categories 列表含 id/name/isthing）", ""),("output_json","path","输出 instances JSON", ""),("min_area","int","实例最小面积",1),("min_dominance","float","语义多数投票最小占比",0.5),("recursive","checkbox","递归原图目录",False)],
    "语义+实例 Mask 生成 Panoptic": [("image_dir","path","原图目录", ""),("semantic_dir","path","语义 Mask 目录", ""),("instance_dir","path","实例 Mask 目录", ""),("panoptic_dir","path","Panoptic RGB Mask 输出目录", ""),("output_json","path","输出 Panoptic JSON", ""),("categories_json","path","类别定义 JSON（categories 列表含 id/name/isthing）", ""),("min_area","int","片段最小面积",1),("ignore_label","int","忽略类别 ID",255),("recursive","checkbox","递归原图目录",False)],
    "删除空标签 TXT 文件": [("txt_dir","path","TXT 标签目录", ""),("recursive","checkbox","递归子目录",False),("confirm","checkbox","确认永久删除空白 TXT 文件",False)],
    "文件扩展名 / 名称批量重命名": [("directory","path","目标目录", ""),("old_ext","text","原扩展名", ".jpg"),("new_ext","text","新扩展名", ".png"),("recursive","checkbox","递归子目录",True),("mode","select","文件名模式",["none","counter","timestamp"]),("base_name","text","counter 模式基础名","image_"),("time_format","text","时间格式","%Y%m%d_%H%M%S"),("start_num","int","序号起始值",1),("num_digits","int","序号位数（0 自动）",0),("confirm","checkbox","确认执行重命名",False)],
    "批量删除文件名子串": [("directory","path","目标目录", ""),("substring","text","需要删除的子串", ""),("recursive","checkbox","递归子目录",False),("confirm","checkbox","确认执行重命名",False)],
    "按文件名末字段分组": [("directory","path","源目录", ""),("keywords","text","匹配字段（逗号分隔）", "vis,nir"),("action","select","操作",["copy","move","delete"]),("target_dir","path","目标根目录（复制/移动需要）", ""),("recursive","checkbox","递归子目录",False),("overwrite","checkbox","覆盖目标同名文件",False),("confirm","checkbox","确认执行操作",False),("confirm_delete","checkbox","确认永久删除（仅删除模式）",False)],
    "清理缺少对应图像的标签": [("reference_dir","path","参考目录（例如 images）", ""),("check_dir","path","待检查目录（例如 labels）", ""),("target_dir","path","移动/复制目标目录", ""),("extensions","text","待检查扩展名（逗号分隔；留空全部）", ".json,.txt,.png"),("strip_suffixes","text","去除的 stem 后缀", "_mask,_label,_gt,_seg"),("action","select","不匹配文件操作",["ignore","move","copy","delete"]),("recursive","checkbox","递归子目录",True),("confirm","checkbox","确认执行操作",False),("confirm_delete","checkbox","确认永久删除",False)],
    "删除中文文件名": [("directory","path","扫描目录", ""),("confirm","checkbox","确认永久删除中文文件名文件",False)],
    "批量修改文件名（去中文）": [("image_dir","path","图像目录", ""),("label_dir","path","标签目录", ""),("prefix","text","统一前缀（留空则删除中文）", ""),("confirm","checkbox","确认执行重命名",False)],
    "清除 TXT 首行": [("input_dir","path","TXT 输入目录", ""),("output_dir","path","输出目录", ""),("recursive","checkbox","递归子目录",False)],
    "统一追加文件名前后缀": [("input_dir","path","输入目录", ""),("output_dir","path","输出目录", ""),("prefix","text","文件名前缀", ""),("suffix","text","文件名后缀", "_new"),("recursive","checkbox","递归子目录",False)],
    "保存指定类别 JSON": [("input_dir","path","输入 JSON 目录", ""),("output_dir","path","输出目录", ""),("labels","text","包含任一类别即保存（逗号分隔）", ""),("recursive","checkbox","递归子目录",False),("indent","int","JSON 缩进",2)],
    "YOLO 数据集文件分类": [("source_dir","path","混合源目录", ""),("output_root","path","输出根目录", ""),("rules","textarea","分类规则（每行：images=.jpg,.png）","images=.jpg,.jpeg,.png,.bmp\nlabels=.txt\njsons=.json"),("move","checkbox","移动而非复制",False),("confirm","checkbox","确认移动",False)],
    "拼接图像目录文件": [("left_dir","path","左侧图像目录", ""),("right_dir","path","右侧图像目录", ""),("output_dir","path","输出目录", ""),("recursive","checkbox","递归子目录",False)],
    "重命名 JSON 对应图像文件名": [("json_dir","path","LabelMe JSON 目录", ""),("image_dir","path","图像目录", ""),("mode","select","模式",["预览","写回修正"]),("recursive","checkbox","递归子目录",False),("confirm","checkbox","确认写回 JSON",False)],
    "工具脚本目录": [],
    "Panoptic 标签检查": [("panoptic_json","path","Panoptic JSON 文件", ""),("panoptic_root","path","Panoptic PNG 目录", "")],
    "语义类别重映射": [("input_dir","path","原始语义 Mask 目录", ""),("output_dir","path","输出目录", ""),("stuff_ids","text","Stuff 类别 ID（按输出顺序，逗号分隔）", "0,2,3,4,5,7"),("thing_ids","text","Thing 类别 ID（统一变为 ignore_label）", "1,6,10,12,13,14,15"),("ignore_label","int","Thing/未知像素输出值",255),("recursive","checkbox","递归子目录",False)],
    "Kaggle 数据集下载": [("dataset_id","text","Kaggle 数据集 ID（owner/dataset）", "")],
}


def _render_field(field):
    key, kind, label, default = field
    if kind == "path" or kind == "text": return st.text_input(label, value=default, key=f"field_{key}")
    if kind == "textarea": return st.text_area(label, value=default, height=105, key=f"field_{key}")
    if kind == "int": return st.number_input(label, value=int(default), step=1, key=f"field_{key}")
    if kind == "float": return st.number_input(label, value=float(default), step=0.01, format="%.4f", key=f"field_{key}")
    if kind == "checkbox": return st.checkbox(label, value=bool(default), key=f"field_{key}")
    if kind == "select": return st.selectbox(label, options=default, key=f"field_{key}")
    return default


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧰", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 14px; margin-bottom: 1rem;}
      .small-muted {color: #777; font-size: .88rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.18); padding: 12px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("CV Tool")
    st.sidebar.caption("本地运行 · 无数据库 · 文件保存在指定目录")
    category = st.sidebar.selectbox("工具类别", list(TOOL_GROUPS.keys()))
    tool = st.sidebar.selectbox("选择功能", TOOL_GROUPS[category])
    st.sidebar.divider()
    st.sidebar.markdown("**本地使用说明**")
    st.sidebar.caption("路径请填写运行此服务的电脑上的绝对路径。输出目录不会自动删除已有文件。")

    st.markdown(f'<div class="hero"><h1 style="margin:0 0 .4rem 0">{tool}</h1><div>{TOOL_DESCRIPTIONS.get(tool, "本地算法辅助工具")}</div><div class="small-muted" style="margin-top:.7rem">工具箱路径：{ROOT}</div></div>', unsafe_allow_html=True)

    if tool == "工具脚本目录":
        st.info("此处列出压缩包内所有 Python 源码。部分脚本是测试、实验或含固定路径的历史版本；网页不会盲目执行这些脚本，以免误删/覆盖本机数据。")
        try:
            catalog = execute_tool("工具脚本目录", {})
            st.success(catalog.get("message", "源码目录已扫描"))
            st.dataframe(catalog.get("table", []), use_container_width=True, hide_index=True)
            if catalog.get("table"):
                import pandas as pd
                csv_data = pd.DataFrame(catalog["table"]).to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载脚本目录 CSV", csv_data, file_name="cv_tool_scripts.csv", mime="text/csv")
        except Exception as exc:
            st.error(str(exc))
    elif tool in FIELDS and FIELDS[tool]:
        st.subheader("参数配置")
        st.caption("建议先使用输出到新目录的方式。删除、移动、原地写回或重命名操作有二次确认选项。")
        values = {}
        with st.form(f"form_{tool}", clear_on_submit=False):
            columns = st.columns(2)
            for idx, field in enumerate(FIELDS[tool]):
                with columns[idx % 2]:
                    values[field[0]] = _render_field(field)
            submitted = st.form_submit_button("运行工具", type="primary", use_container_width=True)
        if submitted:
            with st.status("正在执行…", expanded=True) as status:
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        output = execute_tool(tool, values)
                    status.update(label="执行完成", state="complete", expanded=False)
                    st.success(output.get("message", "执行完成"))
                    if output.get("output_dir"): st.code(f"输出目录：{output['output_dir']}")
                    if output.get("output_file"): st.code(f"输出文件：{output['output_file']}")
                    if output.get("matched") is not None: st.metric("匹配文件数", output["matched"])
                    if output.get("table"):
                        st.dataframe(output["table"], use_container_width=True, hide_index=True)
                        try:
                            csv_bytes = io.StringIO(); output["table"] and __import__('pandas').DataFrame(output["table"]).to_csv(csv_bytes, index=False, encoding="utf-8-sig")
                            st.download_button("下载结果表 CSV", csv_bytes.getvalue().encode("utf-8-sig"), file_name="cv_tool_result.csv", mime="text/csv")
                        except Exception:
                            pass
                    if output.get("errors"):
                        with st.expander(f"查看警告/错误（{len(output['errors'])} 条）"):
                            st.code("\n".join(map(str, output["errors"])))
                    if output.get("log"):
                        with st.expander("外部脚本日志", expanded=False): st.code(output["log"])
                    if output.get("counts") is not None:
                        st.json(output["counts"])
                    captured = buf.getvalue().strip()
                    if captured:
                        with st.expander("运行日志", expanded=False): st.code(captured[-25000:])
                    st.session_state["last_result"] = output
                except Exception as e:
                    status.update(label="执行失败", state="error", expanded=True)
                    st.error(f"{type(e).__name__}: {e}")
                    captured = buf.getvalue().strip()
                    if captured: st.code(captured[-15000:])
                    with st.expander("错误堆栈"):
                        st.code(traceback.format_exc())
    else:
        st.info("该工具没有网页参数面板。请查看脚本目录页与 README_WEB.md 中的运行说明。")

    st.divider()
    st.caption("提示：涉及删除/移动的操作，请先查看预览结果和备份数据。")


if __name__ == "__main__":
    main()
