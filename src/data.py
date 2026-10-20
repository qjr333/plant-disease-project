"""CSV 清单驱动的数据读取、增强与分组划分。

原始图片保持一份。正式清单只在类别、样本审核和来源分组完成后生成。
清单中的 image_path 均相对于项目根目录，例如 data/plantdoc/train/...。
"""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

MANIFEST_COLUMNS = ("image_path", "label", "label_index", "group_id", "dataset", "original_split")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _csv_rows(path, required=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
        return [dict(row) for row in reader]


def _relative_image_path(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError(f"图片路径必须是项目根目录下的 data/... 相对路径：{value!r}")
    return path.as_posix()


def read_manifest(path) -> list:
    """检查清单列、标签映射和重复路径；label_index 转为整数。"""
    rows = _csv_rows(path, MANIFEST_COLUMNS)
    if not rows:
        raise ValueError(f"{path} 尚未包含样本，请先完成数据审核并生成正式划分。")
    seen, label_to_idx, idx_to_label = set(), {}, {}
    for row in rows:
        row["image_path"] = _relative_image_path(row["image_path"])
        if row["image_path"] in seen:
            raise ValueError(f"清单中存在重复图片路径：{row['image_path']}")
        seen.add(row["image_path"])
        try:
            row["label_index"] = int(row["label_index"])
        except (ValueError, TypeError) as exc:
            raise ValueError("label_index 必须是非负整数。") from exc
        if row["label_index"] < 0 or not all(row[key].strip() for key in ("label", "group_id", "dataset")):
            raise ValueError("每个样本必须包含有效的 label、label_index、group_id 和 dataset。")
        label, index = row["label"], row["label_index"]
        if label_to_idx.setdefault(label, index) != index or idx_to_label.setdefault(index, label) != label:
            raise ValueError("清单中的类别名称与编号不是一一对应关系。")
    return rows


class ManifestDataset:
    """读取固定清单，返回 (RGB 图像张量, 类别编号)。"""

    def __init__(self, manifest_path, project_root, transform=None):
        self.project_root = Path(project_root).resolve()
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = self.project_root / manifest_path
        self.rows = read_manifest(manifest_path)
        self.transform = transform if transform is not None else build_transforms()
        for row in self.rows:
            path = (self.project_root / row["image_path"]).resolve()
            if self.project_root not in path.parents:
                raise ValueError(f"图片路径超出项目目录：{path}")
            if not path.is_file():
                raise FileNotFoundError(f"清单中的图片不存在：{path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.project_root / row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), row["label_index"]


def build_transforms(augmentation="none", image_size=224, mean=None, std=None):
    """统一缩放；basic 加几何增强，combined 再加颜色增强。

    验证、测试和 Grad-CAM 一律使用 augmentation='none'。
    所有方案均固定 Resize((image_size, image_size))，不进行中心裁剪。
    """
    from torchvision import transforms

    if augmentation not in ("none", "basic", "combined"):
        raise ValueError("augmentation 只能是 none、basic 或 combined。")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size 必须是正整数。")
    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean/std 必须各有三个值，且 std 均为正数。")
    steps = [transforms.Resize((image_size, image_size), antialias=True)]
    if augmentation != "none":
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])
    if augmentation == "combined":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _seed_worker(worker_id):
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(manifest_path, project_root, transform, batch_size=32, shuffle=False, num_workers=0, seed=42):
    """Notebook 默认使用 num_workers=0，避免多进程启动问题。"""
    import torch
    from torch.utils.data import DataLoader

    dataset = ManifestDataset(manifest_path, project_root, transform)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def dataset_inventory(project_root) -> list:
    rows = _csv_rows(Path(project_root) / "data/metadata/class_counts.csv", ("dataset", "class_name", "image_count"))
    for row in rows:
        row["image_count"] = int(row["image_count"])
    return rows


def _review_table(path, required):
    rows = _csv_rows(path, required)
    indexed = {}
    for row in rows:
        key = _relative_image_path(row["image_path"])
        if key in indexed:
            raise ValueError(f"{Path(path).name} 包含重复路径：{key}")
        indexed[key] = row
    return indexed


def _group_counts(count, ratios):
    targets = [count * ratio for ratio in ratios]
    numbers = [max(1, math.floor(target)) for target in targets]
    while sum(numbers) > count:
        candidates = [index for index, value in enumerate(numbers) if value > 1]
        numbers[max(candidates, key=lambda index: numbers[index] - targets[index])] -= 1
    while sum(numbers) < count:
        numbers[max(range(3), key=lambda index: targets[index] - numbers[index])] += 1
    return numbers


def prepare_manifests(project_root, label_config, seed=42, ratios=(0.7, 0.15, 0.15)) -> dict:
    """在显式审核通过后生成固定划分；不会替用户选择类别或审核样本。

    label_config 格式：
      {'status': 'approved', 'classes': [
        {'label': '统一类别名', 'label_index': 0,
         'plantvillage_class': '来源class_name',
         'plantdoc_classes': ['来源class_name']}, ...]}
    label_index 从 0 连续。review_decisions.csv 要求所有候选样本
    有 keep/exclude；sample_groups.csv 要求所有保留样本分组且 reviewed=true。
    同一来源组不会跨划分，重复内容不得保留多份，跨域来源组不得重叠。
    已有四份清单如内容不一致则拒绝覆盖，防止悄然改变实验数据。
    """
    root = Path(project_root).resolve()
    classes = label_config.get("classes", [])
    if not classes:
        return {"status": "blocked", "reason": "请先在 configs/labels.yaml 确定共同类别并完成数据审核；尚未写入正式划分。"}
    if label_config.get("status") != "approved":
        raise ValueError("类别映射尚未审核：完成核对后将 labels.status 设置为 approved。")
    if len(classes) < 2:
        raise ValueError("至少需要两个共同类别。")
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("ratios 必须是三个正数，且总和为 1。")
    mappings, names, indices = {}, set(), set()
    for item in classes:
        name = item.get("label", "").strip()
        index = item.get("label_index")
        if not name or name in names:
            raise ValueError("每个统一类别必须包含非空、唯一的 label。")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in indices:
            raise ValueError("label_index 必须是从 0 开始的连续且唯一的整数。")
        names.add(name)
        indices.add(index)
        for dataset in ("plantvillage", "plantdoc"):
            sources = [item.get("plantvillage_class", "")] if dataset == "plantvillage" else item.get("plantdoc_classes", [])
            if not isinstance(sources, list) or not sources or not all(isinstance(value, str) and value for value in sources):
                raise ValueError(f"{name} 必须为 {dataset} 提供来源 class_name 列表。")
            for source in sources:
                key = (dataset, source)
                if key in mappings:
                    raise ValueError(f"来源类别被重复映射：{key}")
                mappings[key] = (name, index)
    if indices != set(range(len(classes))):
        raise ValueError("label_index 必须从 0 连续编号。")
    metadata = root / "data/metadata"
    decisions = _review_table(metadata / "review_decisions.csv", ("image_path", "decision", "reason"))
    groups = _review_table(metadata / "sample_groups.csv", ("image_path", "group_id", "reviewed", "group_method"))
    candidates = []
    found_sources = set()
    for row in _csv_rows(metadata / "file_manifest.csv", ("new_path", "dataset", "variant", "class_name", "sha256")):
        key = (row["dataset"], row["class_name"])
        if key not in mappings or (row["dataset"] == "plantvillage" and row["variant"] != "color"):
            continue
        found_sources.add(key)
        row["image_path"] = _relative_image_path("data/" + row["new_path"])
        candidates.append(row)
    if set(mappings) - found_sources:
        raise ValueError(f"以下来源类别不存在：{sorted(set(mappings) - found_sources)}")
    unresolved = [row["image_path"] for row in candidates if decisions.get(row["image_path"], {}).get("decision", "").strip().lower() not in ("keep", "exclude")]
    if unresolved:
        raise ValueError(f"还有 {len(unresolved)} 个候选样本未填写 keep/exclude 审核结论，例如：{unresolved[0]}")
    kept, hashes, group_labels, group_datasets = [], {}, {}, {}
    for row in candidates:
        path = row["image_path"]
        decision = decisions[path]
        if not decision["reason"].strip():
            raise ValueError(f"审核记录需要填写 reason：{path}")
        if decision["decision"].strip().lower() == "exclude":
            continue
        group = groups.get(path, {})
        if group.get("reviewed", "").strip().lower() != "true" or not group.get("group_id", "").strip() or not group.get("group_method", "").strip():
            raise ValueError(f"保留样本尚未完成来源分组审核（reviewed=true，group_id/group_method 非空）：{path}")
        if not (root / path).is_file():
            raise FileNotFoundError(f"图片不存在：{root / path}")
        digest = row["sha256"]
        if not digest or digest in hashes:
            raise ValueError(f"保留样本的校验值为空或存在重复内容；请审核排除重复样本：{path}")
        hashes[digest] = path
        name, index = mappings[(row["dataset"], row["class_name"])]
        group_id = group["group_id"].strip()
        if group_labels.setdefault(group_id, name) != name:
            raise ValueError(f"同一来源组出现不同类别，请审核标签冲突：{group_id}")
        if group_datasets.setdefault(group_id, row["dataset"]) != row["dataset"]:
            raise ValueError(f"PlantVillage 与 PlantDoc 包含同一来源组，需排除跨域重叠：{group_id}")
        kept.append(dict(zip(MANIFEST_COLUMNS, (path, name, index, group_id, row["dataset"], row.get("split") or "unspecified"))))
    partitions = {"pv_train": [], "pv_val": [], "pv_test": [], "plantdoc_external_test": []}
    rng = random.Random(seed)
    for name in (item["label"] for item in sorted(classes, key=lambda item: item["label_index"])):
        pv = [row for row in kept if row["label"] == name and row["dataset"] == "plantvillage"]
        pd = [row for row in kept if row["label"] == name and row["dataset"] == "plantdoc"]
        if not pd:
            raise ValueError(f"类别 {name} 未保留任何 PlantDoc 外部测试样本。")
        partitions["plantdoc_external_test"].extend(pd)
        by_group = defaultdict(list)
        for row in pv:
            by_group[row["group_id"]].append(row)
        ordered_groups = sorted(by_group)
        if len(ordered_groups) < 3:
            raise ValueError(f"类别 {name} 至少需要 3 个经审核的 PlantVillage 独立来源组。")
        rng.shuffle(ordered_groups)
        sizes, start = _group_counts(len(ordered_groups), ratios), 0
        for key, size in zip(("pv_train", "pv_val", "pv_test"), sizes):
            for group_id in ordered_groups[start:start + size]:
                partitions[key].extend(by_group[group_id])
            start += size
    split_dir = root / "data/splits"
    for key, rows in partitions.items():
        rows.sort(key=lambda row: row["image_path"])
        target = split_dir / f"{key}.csv"
        if target.exists():
            existing = read_manifest(target)
            if existing != rows:
                raise FileExistsError(f"正式划分已存在且内容不同，请先归档既有实验与清单：{target}")
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in partitions.items():
        target = split_dir / f"{key}.csv"
        if not target.exists():
            with target.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
    summary = {"status": "ready", "seed": seed, "ratios_by_group": list(ratios), "counts": {key: len(rows) for key, rows in partitions.items()}}
    (split_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
