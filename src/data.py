"""八分类图像扫描、质量检查、来源分组和 CSV 清单读取。原图只读。"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .utils import load_config, load_labels, save_json

MANIFEST_COLUMNS = ("image_path", "label", "label_index", "group_id", "group_method",
                    "dataset", "original_split", "sha256", "pixel_sha256")
PARTITIONS = ("pv_train", "pv_val", "pv_test", "plantdoc_external_train",
              "plantdoc_external_test", "plantdoc_external_all")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _csv_rows(path, required=()):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} 缺少列：{sorted(missing)}")
        return list(reader)


def _write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative_image_path(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError(f"图片路径必须是项目内 data/... 相对路径：{value!r}")
    return path.as_posix()


def read_manifest(path):
    rows = _csv_rows(path, MANIFEST_COLUMNS)
    if not rows:
        raise ValueError(f"{path} 为空；请先运行 01 数据准备。")
    seen = set()
    for row in rows:
        row["image_path"] = _relative_image_path(row["image_path"])
        row["label_index"] = int(row["label_index"])
        if row["image_path"] in seen or not 0 <= row["label_index"] < 8:
            raise ValueError(f"重复路径或无效的八分类编号：{row['image_path']}")
        if not row["group_id"] or not row["sha256"] or not row["pixel_sha256"]:
            raise ValueError("清单缺少来源分组或图像指纹。")
        seen.add(row["image_path"])
    return rows


class ManifestDataset:
    """使用统一标签清单，返回 RGB 张量和 0—7 的类别编号。"""

    def __init__(self, manifest_path, project_root, transform=None):
        self.project_root = Path(project_root).resolve()
        path = Path(manifest_path)
        self.rows = read_manifest(path if path.is_absolute() else self.project_root / path)
        self.transform = transform if transform is not None else build_transforms()
        labels = [item["label"] for item in load_labels(self.project_root)]
        for row in self.rows:
            if row["label"] != labels[row["label_index"]]:
                raise ValueError("清单标签顺序与 configs/labels.yaml 不一致。")
            image = (self.project_root / row["image_path"]).resolve()
            if self.project_root not in image.parents or not image.is_file():
                raise ValueError(f"图像不存在或超出项目目录：{image}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(self.project_root / row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), row["label_index"]


def build_transforms(augmentation="none", image_size=224, mean=None, std=None):
    """验证/测试使用 none；basic 为几何增强，combined 再加颜色扰动。"""
    from torchvision import transforms
    if augmentation not in ("none", "basic", "combined"):
        raise ValueError("augmentation 必须为 none、basic 或 combined。")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size 必须为正整数。")
    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    steps = [transforms.Resize((image_size, image_size), antialias=True)]
    if augmentation != "none":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15)]
    if augmentation == "combined":
        steps += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def _seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(manifest_path, project_root, transform, batch_size=16, shuffle=False, num_workers=0, seed=42):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(ManifestDataset(manifest_path, project_root, transform), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, worker_init_fn=_seed_worker,
                      generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available())


def dataset_inventory(project_root):
    """直接扫描目标目录计数，可在首次 prepare 之前运行。"""
    root = Path(project_root)
    rows = []
    for item in load_labels(root):
        sources = [("plantvillage", "color", item["plantvillage_class"])]
        sources += [("plantdoc", split, name) for split in ("train", "test") for name in item["plantdoc_classes"]]
        for dataset, split, name in sources:
            folder = root / "data" / dataset / split / name
            if not folder.is_dir():
                raise FileNotFoundError(f"缺少目标类别目录：{folder}")
            rows.append({"dataset": dataset, "original_split": split, "class_name": name,
                         "label": item["label"], "label_index": item["label_index"],
                         "image_count": sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*"))})
    return rows


def dataset_fingerprint(project_root):
    root = Path(project_root)
    digest = hashlib.sha256()
    for relative in ["configs/labels.yaml"] + [f"data/splits/{name}.csv" for name in PARTITIONS]:
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def validate_partitions(project_root):
    """核查清单内路径、指纹、已知组及标签；磁盘解码/哈希由 prepare 完成。"""
    root = Path(project_root)
    classes = [item["label"] for item in load_labels(root)]
    partitions = {name: read_manifest(root / f"data/splits/{name}.csv") for name in PARTITIONS}
    for name, rows in partitions.items():
        for row in rows:
            if row["label"] != classes[row["label_index"]]:
                raise ValueError(f"{name} 标签顺序不一致。")
            expected = "plantvillage" if name.startswith("pv_") else "plantdoc"
            if row["dataset"] != expected:
                raise ValueError(f"{name} 混入其他数据集。")
        for key in ("image_path", "sha256", "pixel_sha256"):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError(f"{name} 存在重复 {key}。")
        if name.startswith("pv_") and {r["label_index"] for r in rows} != set(range(8)):
            raise ValueError(f"{name} 未包含全部八类。")
    disjoint = PARTITIONS[:-1]
    for i, name in enumerate(disjoint):
        for other in disjoint[i + 1:]:
            for key in ("image_path", "sha256", "pixel_sha256", "group_id"):
                if {r[key] for r in partitions[name]} & {r[key] for r in partitions[other]}:
                    raise ValueError(f"{name} 与 {other} 发生 {key} 重叠。")
    all_paths = {r["image_path"] for r in partitions["plantdoc_external_all"]}
    expected = {r["image_path"] for name in PARTITIONS[3:5] for r in partitions[name]}
    if all_paths != expected:
        raise ValueError("PlantDoc 合并清单与来源清单不一致。")
    for source in ("train", "test"):
        if any(r["original_split"] != source for r in partitions[f"plantdoc_external_{source}"]):
            raise ValueError("PlantDoc 原始来源标识错误。")
    return {"status": "passed", "counts": {k: len(v) for k, v in partitions.items()},
            "checks": ["labels", "paths", "byte_hash", "pixel_hash", "known_source_groups", "external_union"],
            "limitation": "此处核查清单一致性；磁盘内容由 prepare_manifests 解码和计算指纹。仅核查已知来源组，仍可能有未识别同源图。"}


def _group_split(rows, seed, ratios):
    """逐类按样本数逼近目标比例，同时保持可靠来源组完整。"""
    result = [[], [], []]
    rng = random.Random(seed)
    for label in range(8):
        groups = defaultdict(list)
        for row in rows:
            if row["label_index"] == label:
                groups[row["group_id"]].append(row)
        if len(groups) < 3:
            raise ValueError(f"类别 {label} 少于三个独立来源组，无法三分。")
        values = list(groups.values())
        rng.shuffle(values)
        values.sort(key=len, reverse=True)
        targets = [sum(map(len, values)) * ratio for ratio in ratios]
        counts = [0, 0, 0]
        for group in values:
            # 最大绝对缺口优先；大组先分配使样本比例接近 70/15/15。
            index = max(range(3), key=lambda i: targets[i] - counts[i])
            result[index].extend(group)
            counts[index] += len(group)
        if any(count == 0 for count in counts):
            raise ValueError(f"类别 {label} 来源组过大，产生空划分。")
    return result


def prepare_manifests(project_root, label_config=None, seed=42, ratios=(0.7, 0.15, 0.15)):
    """全量解码目标图像、去重、复用可验证来源信息，再固定分层划分。

    生成文件可重复运行；已有实验与数据指纹绑定，不会把变化后的划分当成旧实验数据。
    无法匹配官方叶片分组的文件保留为来源未知的单文件组，明确记录这一限制。
    """
    from PIL import Image
    root = Path(project_root).resolve()
    classes = load_labels(root)
    if label_config is not None and label_config["classes"] != classes:
        raise ValueError("只允许使用 configs/labels.yaml 的正式八分类配置。")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("划分比例必须为三个正数且总和为 1。")
    metadata = root / "data/metadata"
    source_manifest = metadata / "file_manifest.csv"
    provenance = {}
    if source_manifest.exists():
        names = {item["plantvillage_class"] for item in classes}
        with source_manifest.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["dataset"] == "plantvillage" and row["variant"] == "color" and row["class_name"] in names:
                    provenance["data/" + row["new_path"]] = row
    leaf_path = metadata / "plantvillage_leaf_map.json"
    leaf_map = json.loads(leaf_path.read_text()) if leaf_path.exists() else {}
    annotation_path = metadata / "source_exclusions.csv"
    annotations = {r["image_path"]: r["reason"] for r in _csv_rows(annotation_path)} if annotation_path.exists() else {}
    candidates, exclusions, inventory = [], [], []
    for item in classes:
        sources = [("plantvillage", "color", item["plantvillage_class"])]
        sources += [("plantdoc", split, name) for split in ("train", "test") for name in item["plantdoc_classes"]]
        for dataset, original_split, source_name in sources:
            folder = root / "data" / dataset / original_split / source_name
            if not folder.is_dir():
                raise FileNotFoundError(f"缺少目标类别目录：{folder}")
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise ValueError(f"目标类别目录没有图像：{folder}")
            inventory.append({"dataset": dataset, "original_split": original_split, "class_name": source_name,
                              "label": item["label"], "label_index": item["label_index"], "image_count": len(paths)})
            for path in paths:
                relative = path.relative_to(root).as_posix()
                try:
                    byte_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        rgb.load()
                        size = rgb.size
                        pixel_hash = hashlib.sha256(f"{size[0]}x{size[1]}:RGB:".encode() + rgb.tobytes()).hexdigest()
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                    exclusions.append({"image_path": relative, "reason": "decode_error", "detail": str(error)})
                    continue
                group_id, method = f"{dataset}_file:{byte_hash}", "source_unknown_file_only"
                source = provenance.get(relative)
                if dataset == "plantvillage" and source and source["sha256"] == byte_hash:
                    key = Path(source["original_path"]).stem.split("___", 1)[-1].strip().lower()
                    matches = sorted({value for value in leaf_map.get(key, []) if value.split(":::")[0] == source["original_class"]})
                    if len(matches) == 1:
                        group_id, method = "pv_leaf:" + matches[0], "official_leaf_map_exact_stem_and_class"
                candidates.append({"image_path": relative, "label": item["label"], "label_index": item["label_index"],
                                   "group_id": group_id, "group_method": method, "dataset": dataset,
                                   "original_split": original_split, "sha256": byte_hash, "pixel_sha256": pixel_hash})
    # 将字节相同、像素相同和已有人工确认的同照片版本合并为连通分量。
    parent = {row["image_path"]: row["image_path"] for row in candidates}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for key in ("sha256", "pixel_sha256"):
        seen = {}
        for row in candidates:
            value = row[key]
            if value in seen:
                union(row["image_path"], seen[value])
            else:
                seen[value] = row["image_path"]
    pair_path = metadata / "confirmed_same_photo_pairs.json"
    pairs = json.loads(pair_path.read_text()) if pair_path.exists() else []
    for pair in pairs:
        if pair["a"] in parent and pair["b"] in parent:
            union(pair["a"], pair["b"])
    components = defaultdict(list)
    for row in candidates:
        components[find(row["image_path"])].append(row)
    kept = []
    for component in components.values():
        # 跨标签冲突整组排除；跨数据集同图保留 PlantVillage，避免训练—外测重叠。
        component.sort(key=lambda r: (r["dataset"] != "plantvillage",
                                      not r["group_method"].startswith("official_"),
                                      r["original_split"] != "test", r["image_path"]))
        if len({r["label_index"] for r in component}) > 1:
            exclusions.extend({"image_path": r["image_path"], "reason": "duplicate_label_conflict", "detail": "同图存在不同源标签，整组排除"} for r in component)
            continue
        annotation = next((annotations[r["image_path"]] for r in component if r["image_path"] in annotations), None)
        if annotation:
            exclusions.extend({"image_path": r["image_path"], "reason": "source_annotation", "detail": annotation} for r in component)
            continue
        kept.append(component[0])
        exclusions.extend({"image_path": r["image_path"], "reason": "duplicate_content_or_confirmed_photo", "detail": component[0]["image_path"]} for r in component[1:])
    pv = [row for row in kept if row["dataset"] == "plantvillage"]
    pv_parts = _group_split(pv, seed, ratios)
    partitions = dict(zip(PARTITIONS[:3], pv_parts))
    external = [row for row in kept if row["dataset"] == "plantdoc"]
    for split in ("train", "test"):
        partitions[f"plantdoc_external_{split}"] = [r for r in external if r["original_split"] == split]
    partitions["plantdoc_external_all"] = external
    for name, rows in partitions.items():
        rows.sort(key=lambda row: (row["label_index"], row["image_path"]))
        _write_csv(root / f"data/splits/{name}.csv", rows, MANIFEST_COLUMNS)
    _write_csv(metadata / "target_class_counts.csv", inventory,
               ("dataset", "original_split", "class_name", "label", "label_index", "image_count"))
    _write_csv(metadata / "excluded_samples.csv", sorted(exclusions, key=lambda r: r["image_path"]), ("image_path", "reason", "detail"))
    known = sum(row["group_method"].startswith("official_") for row in pv)
    quality = {"raw_target_images": sum(row["image_count"] for row in inventory),
               "excluded": len(exclusions), "exclusion_reasons": dict(Counter(row["reason"] for row in exclusions)),
               "pv_known_leaf_group_images": known, "pv_source_unknown_images": len(pv) - known,
               "plantdoc_raw_by_source": {split: sum(r["image_count"] for r in inventory if r["dataset"] == "plantdoc" and r["original_split"] == split) for split in ("train", "test")},
               "source_annotation_scope": "复用旧人工初筛中有具体内容/标注依据的排除；不沿用缺少分组即排除的旧政策，也不代表八类均已人工审核。",
               "leaf_map_sha256": hashlib.sha256(leaf_path.read_bytes()).hexdigest() if leaf_path.exists() else None,
               "limitations": ["未匹配官方来源组的图像按单文件分组，不能据此宣称不存在同叶片泄漏。",
                               "未重新执行全量近重复或专家病理审核，源标签错误和未识别同源图仍可能存在。",
                               "PlantDoc 健康及黄化病毒映射为暂定语义对应；原始 train/test 均仅作最终外部评估。"]}
    summary = {"status": "ready", "seed": seed, "ratios": list(ratios),
               "counts": {name: len(rows) for name, rows in partitions.items()},
               "per_class": [{"label_index": item["label_index"], "label": item["label"],
                              **{name: sum(r["label_index"] == item["label_index"] for r in rows) for name, rows in partitions.items()}} for item in classes],
               "quality": quality, "validation": validate_partitions(root), "fingerprint": dataset_fingerprint(root)}
    save_json(metadata / "data_quality.json", quality)
    save_json(root / "data/splits/split_summary.json", summary)
    return summary


if __name__ == "__main__":
    from .utils import find_project_root
    result = prepare_manifests(find_project_root())
    print(json.dumps({"counts": result["counts"], "quality": result["quality"]}, ensure_ascii=False, indent=2))
