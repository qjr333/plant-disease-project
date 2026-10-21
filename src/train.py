"""Notebook 共用训练流程；模型选择只读取 PlantVillage 验证指标。"""
from __future__ import annotations

import copy
import csv
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .utils import environment_info, get_device, load_labels, save_json, set_seed

import torch
from sklearn.metrics import f1_score

from .data import build_transforms, dataset_fingerprint, make_loader, read_manifest, validate_partitions
from .models import build_model, keep_frozen_batchnorm_eval, parameter_counts


def _check_labels(rows, class_names):
    for row in rows:
        index = int(row["label_index"])
        if not 0 <= index < 8 or row["label"] != class_names[index]:
            raise ValueError(f"清单与八分类配置不一致：{row.get('image_path')}")


def _epoch(model, loader, criterion, device, num_classes=8, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_batchnorm_eval(model)
    total_loss, count, truth, predictions = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现非有限值，请检查学习率和输入。")
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not count:
        raise ValueError("训练或验证集为空。")
    return {
        "loss": total_loss / count,
        "accuracy": sum(t == p for t, p in zip(truth, predictions)) / count,
        "macro_f1": float(f1_score(truth, predictions, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def _smoke_rows(rows, per_class):
    counts, selected = {}, []
    for row in rows:
        index = int(row["label_index"])
        if counts.get(index, 0) < per_class:
            selected.append(row)
            counts[index] = counts.get(index, 0) + 1
    return selected


def _save_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run_experiment(project_root, config, run_id=None, smoke_limit=None):
    """创建独立运行。smoke_limit=2 表示训练/验证每类取 2/1 张，禁止参与正式选择。"""
    root = Path(project_root).resolve()
    config = copy.deepcopy(config)
    run_id = run_id or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id 只能包含英文字母、数字、下划线和连字符。")
    if smoke_limit is not None:
        if not isinstance(smoke_limit, int) or isinstance(smoke_limit, bool) or smoke_limit < 1:
            raise ValueError("smoke_limit 必须是每类训练样本数的正整数。")
        config["smoke_test"] = True
        config["smoke_limit"] = {"train_per_class": smoke_limit, "val_per_class": max(1, smoke_limit // 2)}
    config.setdefault("smoke_test", False)
    run_dir = root / "outputs/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started, history = time.perf_counter(), []
    status = {"run_id": run_id, "status": "initializing", "smoke_test": bool(config["smoke_test"]),
              "started_at": datetime.now(timezone.utc).isoformat(), "epochs_completed": 0}
    save_json(run_dir / "status.json", status)
    save_json(run_dir / "config.yaml", config)
    try:
        classes = load_labels(root)
        class_names = [item["label"] for item in classes]
        class_to_idx = {label: i for i, label in enumerate(class_names)}
        status["dataset_fingerprint"] = dataset_fingerprint(root)
        validate_partitions(root)
        save_json(run_dir / "data_identity.json", {"dataset_fingerprint": status["dataset_fingerprint"]})
        save_json(run_dir / "label_mapping.json", {"class_names": class_names, "class_to_idx": class_to_idx, "classes": classes})
        data_config, training, model_config = config["data"], config["training"], config["model"]
        if int(training["epochs"]) < 1 or int(training["patience"]) < 1:
            raise ValueError("epochs 和 patience 必须为正整数。")
        if training.get("monitor", "val_macro_f1") != "val_macro_f1":
            raise ValueError("最佳模型统一按 val_macro_f1 选择。")
        weights = model_config.get("weights", "IMAGENET1K_V1")
        if not config["smoke_test"] and weights not in ("DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError("正式实验必须使用 ImageNet 预训练权重。")
        seed = set_seed(config.get("seed", 42))
        device = get_device(config.get("device", "auto"))
        save_json(run_dir / "environment.json", environment_info(device))
        train_path, val_path = root / data_config["train_manifest"], root / data_config["val_manifest"]
        if (train_path.resolve() != (root / "data/splits/pv_train.csv").resolve()
                or val_path.resolve() != (root / "data/splits/pv_val.csv").resolve()):
            raise ValueError("训练/验证必须使用固定的 pv_train.csv / pv_val.csv，测试集仅供最终评价。")
        train_rows, val_rows = read_manifest(train_path), read_manifest(val_path)
        _check_labels(train_rows + val_rows, class_names)
        if any(row["dataset"] != "plantvillage" for row in train_rows + val_rows):
            raise ValueError("模型训练与选择只能使用 PlantVillage。")
        if {row["image_path"] for row in train_rows} & {row["image_path"] for row in val_rows}:
            raise ValueError("训练和验证存在重复路径。")
        if {row["group_id"] for row in train_rows} & {row["group_id"] for row in val_rows}:
            raise ValueError("训练和验证存在重叠来源组。")
        transform_args = {key: data_config[key] for key in ("image_size", "mean", "std") if key in data_config}
        loader_args = {"project_root": root, "batch_size": int(training["batch_size"]),
                       "num_workers": int(training.get("num_workers", 0)), "seed": seed}
        train_loader = make_loader(train_path, transform=build_transforms(config.get("augmentation", "basic"), **transform_args),
                                   shuffle=True, **loader_args)
        val_loader = make_loader(val_path, transform=build_transforms("none", **transform_args), shuffle=False, **loader_args)
        if smoke_limit is not None:
            train_loader.dataset.rows = _smoke_rows(train_rows, smoke_limit)
            val_loader.dataset.rows = _smoke_rows(val_rows, max(1, smoke_limit // 2))
        status["sample_counts"] = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
        save_json(run_dir / "status.json", status)
        model = build_model(model_config["name"], 8, weights=weights,
                            finetuning=model_config.get("finetuning", "freeze")).to(device)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                     lr=float(training["lr"]), weight_decay=float(training.get("weight_decay", 0)))
        criterion = torch.nn.CrossEntropyLoss()
        status.update(status="running", parameters=parameter_counts(model), device=str(device))
        save_json(run_dir / "status.json", status)
        best_score, stale, best_epoch, best_metrics = -math.inf, 0, None, None
        for epoch in range(1, int(training["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_metrics = _epoch(model, train_loader, criterion, device, optimizer=optimizer)
            val_metrics = _epoch(model, val_loader, criterion, device)
            row = {"epoch": epoch, **{"train_" + key: value for key, value in train_metrics.items()},
                   **{"val_" + key: value for key, value in val_metrics.items()},
                   "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started,
                   "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            temporary = run_dir / "history.csv.tmp"
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            temporary.replace(run_dir / "history.csv")
            if val_metrics["macro_f1"] > best_score:
                best_score, stale, best_epoch, best_metrics = val_metrics["macro_f1"], 0, epoch, val_metrics
                _save_checkpoint(run_dir / "best.pt", {
                    "model_name": model_config["name"], "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "class_names": class_names, "class_to_idx": class_to_idx, "config": config, "epoch": epoch,
                    "monitor": "val_macro_f1", "validation_metrics": val_metrics,
                    "dataset_fingerprint": status["dataset_fingerprint"], "smoke_test": bool(config["smoke_test"]),
                })
            else:
                stale += 1
            status.update(epochs_completed=len(history), best_score=best_score, best_epoch=best_epoch,
                          validation_metrics=best_metrics, seconds=time.perf_counter() - started)
            save_json(run_dir / "status.json", status)
            print(f"{run_id} | epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}")
            if training.get("early_stopping", True) and stale >= int(training["patience"]):
                break
        status.update(status="completed", monitor="val_macro_f1", finished_at=datetime.now(timezone.utc).isoformat(),
                      seconds=time.perf_counter() - started)
        save_json(run_dir / "status.json", status)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error),
                      error_type=type(error).__name__, epochs_completed=len(history), seconds=time.perf_counter() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save_json(run_dir / "status.json", status)
        raise
    return run_dir
