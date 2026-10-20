"""Shared configuration, paths and reproducibility helpers for the notebooks."""
from __future__ import annotations

import json
import os
import platform
import random
from importlib import metadata
from pathlib import Path


def find_project_root(start=None):
    """Find the project from a notebook, a terminal, or this installed module."""
    starts = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for origin in starts:
        if origin.is_file():
            origin = origin.parent
        for candidate in [origin, *origin.parents]:
            for root in [candidate, candidate / 'plant-disease-project']:
                if (root / 'configs/base.yaml').is_file() and (root / 'data').is_dir():
                    return root
    raise FileNotFoundError('未找到项目根目录，请从项目文件夹或 notebooks 文件夹运行。')


def load_config(path):
    """Read JSON-compatible YAML, or ordinary YAML when PyYAML is installed."""
    path = Path(path)
    content = path.read_text(encoding='utf-8-sig')
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ImportError('此配置使用 YAML 语法，请先安装 requirements.txt 中的 PyYAML。') from error
        result = yaml.safe_load(content)
    if not isinstance(result, dict):
        raise ValueError(f'配置文件顶层必须为键值对象：{path}')
    return result


def load_base_config(project_root=None):
    root = Path(project_root) if project_root is not None else find_project_root()
    return load_config(root / 'configs/base.yaml')


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path


def get_device(preference='auto'):
    import torch
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    device = torch.device(preference)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境没有可用的 CUDA 设备，请使用 auto 或 cpu。')
    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('当前环境没有可用的 MPS 设备，请使用 auto 或 cpu。')
    return device


def set_seed(seed):
    import numpy as np
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Hardware and library differences can still affect exact reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def environment_info(device):
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'gradio', 'grad-cam']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'device': str(device), 'packages': packages}


def approved_labels(project_root):
    """Return the final shared label order; source-specific indices are unsuitable."""
    config = load_config(Path(project_root) / 'configs/labels.yaml')
    classes = config.get('classes', [])
    if config.get('status') != 'approved' or len(classes) < 2:
        raise ValueError('请先在 01 中审核共同类别，并将 configs/labels.yaml 的 status 设为 approved。')
    ordered = sorted(classes, key=lambda item: item['label_index'])
    indices = [item['label_index'] for item in ordered]
    names = [item['label'] for item in ordered]
    if indices != list(range(len(ordered))) or len(set(names)) != len(names):
        raise ValueError('共同类别的 label_index 必须从 0 连续编号，label 不得重复。')
    return ordered
