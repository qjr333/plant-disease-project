"""项目路径、唯一标签配置、运行环境及随机种子。"""
from __future__ import annotations

import json
import os
import platform
import random
import warnings
from importlib import metadata
from pathlib import Path

# MPS 遇到尚未实现的算子时允许 PyTorch 回退到 CPU；须在导入 torch 前设置。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def find_project_root(start=None):
    for origin in [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]:
        if origin.is_file():
            origin = origin.parent
        for candidate in [origin, *origin.parents]:
            for root in [candidate, candidate / "plant-disease-project"]:
                if (root / "configs/base.yaml").is_file() and (root / "data").is_dir():
                    return root
    raise FileNotFoundError("请在 VSCode 中打开 plant-disease-project 文件夹。")


def load_config(path):
    import yaml
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"配置顶层必须为键值对象：{path}")
    return result


def load_base_config(project_root=None):
    return load_config(Path(project_root or find_project_root()) / "configs/base.yaml")


def load_labels(project_root):
    config = load_config(Path(project_root) / "configs/labels.yaml")
    classes = config.get("classes", [])
    if len(classes) != 8 or [item["label_index"] for item in classes] != list(range(8)):
        raise ValueError("labels.yaml 必须依次定义 0—7 共八类。")
    for key in ("label", "plantvillage_class", "display_name"):
        if len({item[key] for item in classes}) != 8:
            raise ValueError(f"八分类 {key} 必须唯一。")
    sources = [source for item in classes for source in item["plantdoc_classes"]]
    if len(sources) != 8 or len(set(sources)) != 8:
        raise ValueError("PlantDoc 八类映射必须唯一。")
    return classes


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def get_device(preference="auto"):
    import torch
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("mps" if mps else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if (device.type == "mps" and not mps) or (device.type == "cuda" and not torch.cuda.is_available()):
        warnings.warn(f"{preference} 不可用，本次使用 CPU。", RuntimeWarning)
        return torch.device("cpu")
    return device


def set_seed(seed=42):
    import numpy as np
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def environment_info(device):
    packages = {}
    for name in ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "PyYAML", "matplotlib", "nbformat"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "packages": packages}
