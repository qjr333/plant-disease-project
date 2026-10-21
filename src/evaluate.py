"""八分类评价与推理；最终测试使用 E3 验证集选择所得模型。"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .utils import get_device, load_config, load_labels, save_json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .data import build_transforms, dataset_fingerprint, make_loader
from .models import build_model, parameter_counts


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def classification_metrics(truth, predictions, class_names):
    if len(class_names) != 8 or len(set(class_names)) != 8:
        raise ValueError("评价必须提供有序、唯一的八分类标签。")
    if len(truth) == 0 or len(truth) != len(predictions):
        raise ValueError("真实标签与预测必须非空且等长。")
    indices = list(range(8))
    if any(int(value) not in indices for value in list(truth) + list(predictions)):
        raise ValueError("预测或真实标签超出八分类编号范围。")
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=indices, average="macro", zero_division=0)
    return {
        "sample_count": len(truth), "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
        "classification_report": classification_report(truth, predictions, labels=indices, target_names=class_names,
                                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=indices).tolist(),
        "class_names": class_names,
    }


def load_trained_model(run_dir, device="cpu"):
    """加载本地 checkpoint，不再下载预训练权重。"""
    device = get_device(str(device))
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    for key in ("model_name", "model_state_dict", "class_names", "config", "dataset_fingerprint"):
        if key not in checkpoint:
            raise ValueError(f"模型缺少 {key}，请使用新的八分类训练流程生成权重。")
    classes = checkpoint["class_names"]
    if len(classes) != 8 or len(set(classes)) != 8:
        raise ValueError("checkpoint 的标签不是唯一的八分类。")
    model = build_model(checkpoint["model_name"], 8, weights=None,
                        finetuning=checkpoint["config"]["model"].get("finetuning", "freeze"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_image(model, image, transform, class_names, device="cpu"):
    device = get_device(str(device))
    if len(class_names) != 8:
        raise ValueError("推理标签必须为八分类。")
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.inference_mode():
        scores = model(inputs).softmax(1)[0].cpu().tolist()
    index = int(np.argmax(scores))
    return {"label": class_names[index], "label_index": index, "confidence": float(scores[index]),
            "probabilities": dict(zip(class_names, map(float, scores)))}


def evaluate_checkpoint(project_root, run_dir, manifest_path, dataset_name="test"):
    """在固定清单上评价，逐图预测保存原始 train/test 来源标识。"""
    root, run_dir = Path(project_root).resolve(), Path(run_dir)
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    manifest_path = Path(manifest_path)
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    config = load_config(run_dir / "config.yaml")
    device = get_device(config.get("device", "auto"))
    model, checkpoint = load_trained_model(run_dir, device)
    classes, data_config = checkpoint["class_names"], checkpoint["config"]["data"]
    if classes != [item["label"] for item in load_labels(root)]:
        raise ValueError("模型八分类标签与当前 labels.yaml 不一致。")
    if checkpoint["dataset_fingerprint"] != dataset_fingerprint(root):
        raise ValueError("当前数据清单或标签已改变，不能混用该 checkpoint 的结果。")
    transform = build_transforms("none", **{key: data_config[key] for key in ("image_size", "mean", "std") if key in data_config})
    loader = make_loader(manifest_path, root, transform, batch_size=int(config["training"]["batch_size"]),
                         shuffle=False, num_workers=int(config["training"].get("num_workers", 0)), seed=config.get("seed", 42))
    source_rows = loader.dataset.rows
    for row in source_rows:
        index = int(row["label_index"])
        if not 0 <= index < 8 or row["label"] != classes[index]:
            raise ValueError("评价清单与模型保存的标签映射不一致。")
    truth, predicted, scores, inference_seconds = [], [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            _synchronize(device)
            started = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            inference_seconds += time.perf_counter() - started
            probabilities = logits.softmax(1).cpu()
            truth.extend(labels.tolist())
            predicted.extend(probabilities.argmax(1).tolist())
            scores.extend(probabilities.max(1).values.tolist())
    metrics = classification_metrics(truth, predicted, classes)
    metrics.update(parameter_counts(model))
    status = load_config(run_dir / "status.json")
    metrics.update(dataset=dataset_name, manifest=str(manifest_path.relative_to(root)),
                   checkpoint_bytes=(run_dir / "best.pt").stat().st_size,
                   batch_amortized_inference_ms=inference_seconds * 1000 / len(truth), device=str(device),
                   training_seconds=status.get("seconds"), smoke_test=bool(checkpoint.get("smoke_test", False)))
    if any(row["dataset"] == "plantdoc" for row in source_rows):
        source_metrics = {}
        for split in sorted({row["original_split"] for row in source_rows}):
            positions = [index for index, row in enumerate(source_rows) if row["original_split"] == split]
            source_metrics[split] = classification_metrics([truth[index] for index in positions],
                                                          [predicted[index] for index in positions], classes)
        metrics["source_splits"] = source_metrics
    metrics_path = run_dir / "metrics.json"
    combined = load_config(metrics_path) if metrics_path.exists() else {}
    combined[dataset_name] = metrics
    save_json(metrics_path, combined)
    prediction_path = run_dir / "predictions.csv"
    existing = []
    if prediction_path.exists():
        with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = [row for row in csv.DictReader(handle) if row["dataset"] != dataset_name]
    new_rows = [{"dataset": dataset_name, "image_path": source["image_path"], "original_split": source["original_split"],
                 "true_label": classes[t], "true_index": t, "predicted_label": classes[p], "predicted_index": p,
                 "confidence": confidence, "correct": t == p}
                for source, t, p, confidence in zip(source_rows, truth, predicted, scores)]
    fields = ["dataset", "image_path", "original_split", "true_label", "true_index", "predicted_label", "predicted_index", "confidence", "correct"]
    temporary = prediction_path.with_name(prediction_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + new_rows)
    temporary.replace(prediction_path)
    return metrics


def evaluate_final(project_root):
    """只评价已保存的 E3 最佳正式实验；PlantDoc 合并集同时报告两个原始来源。"""
    from .experiments import selected_run

    root = Path(project_root).resolve()
    run_dir = selected_run(root, "E3")
    config = load_config(run_dir / "config.yaml")
    data = config["data"]
    results = {}
    for name, manifest in (("plantvillage_test", data["test_manifest"]),
                           ("plantdoc_external_all", data["external_manifest"])):
        results[name] = evaluate_checkpoint(root, run_dir, manifest, name)
    return results


def benchmark_model(model, image_size=224, device="cpu", repeats=30, warmup=5):
    """单张前向延迟，不含读图和预处理。"""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats 必须为正整数，warmup 不得为负。")
    device = get_device(str(device))
    model.to(device).eval()
    example, measurements = torch.zeros(1, 3, image_size, image_size, device=device), []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            _synchronize(device)
            started = time.perf_counter()
            model(example)
            _synchronize(device)
            if index >= warmup:
                measurements.append((time.perf_counter() - started) * 1000)
    return {"single_image_mean_ms": float(np.mean(measurements)), "single_image_std_ms": float(np.std(measurements)),
            "device": str(device), "repeats": repeats, "image_size": image_size}
