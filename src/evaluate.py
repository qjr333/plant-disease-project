"""Evaluation and inference helpers; no model selection is performed here."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import build_transforms, make_loader
from .models import build_model, parameter_counts
from .utils import get_device, load_config, save_json


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def classification_metrics(truth, predictions, class_names):
    if len(truth) == 0 or len(truth) != len(predictions):
        raise ValueError('真实标签与预测必须非空且等长。')
    indices = list(range(len(class_names)))
    return {'sample_count': len(truth), 'accuracy': float(accuracy_score(truth, predictions)),
            'macro_f1': float(f1_score(truth, predictions, labels=indices, average='macro', zero_division=0)),
            'classification_report': classification_report(truth, predictions, labels=indices, target_names=class_names, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, predictions, labels=indices).tolist()}


def load_trained_model(run_dir, device='cpu'):
    device = get_device(str(device))
    checkpoint = torch.load(Path(run_dir) / 'best.pt', map_location='cpu', weights_only=True)
    for key in ['model_name', 'model_state_dict', 'class_names', 'config']:
        if key not in checkpoint:
            raise ValueError(f'模型文件缺少 {key}，请使用本项目训练流程生成的模型。')
    model = build_model(checkpoint['model_name'], len(checkpoint['class_names']), weights=None,
                        finetuning=checkpoint['config']['model'].get('finetuning', 'freeze'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, checkpoint


def predict_image(model, image, transform, class_names, device='cpu'):
    device = get_device(str(device))
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    inputs = transform(image.convert('RGB')).unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.inference_mode():
        scores = model(inputs).softmax(1)[0].cpu().tolist()
    index = int(np.argmax(scores))
    return {'label': class_names[index], 'label_index': index, 'confidence': float(scores[index]),
            'probabilities': dict(zip(class_names, map(float, scores)))}


def evaluate_checkpoint(project_root, run_dir, manifest_path, dataset_name='test'):
    root, run_dir = Path(project_root), Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    config = load_config(run_dir / 'config.yaml')
    device = get_device(config.get('device', 'auto'))
    model, checkpoint = load_trained_model(run_dir, device)
    classes, data_config = checkpoint['class_names'], checkpoint['config']['data']
    transform = build_transforms('none', **{k: data_config[k] for k in ['image_size', 'mean', 'std'] if k in data_config})
    loader = make_loader(manifest_path, root, transform, batch_size=int(config['training']['batch_size']),
                         shuffle=False, num_workers=int(config['training'].get('num_workers', 0)), seed=config.get('seed', 42))
    source_rows = loader.dataset.rows
    for row in source_rows:
        index = int(row['label_index'])
        if not 0 <= index < len(classes) or row['label'] != classes[index]:
            raise ValueError('测试清单类别顺序与模型保存的类别映射不一致。')
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
    metrics.update(dataset=dataset_name, manifest=str(manifest_path.relative_to(root)),
                   checkpoint_bytes=(run_dir / 'best.pt').stat().st_size,
                   batch_amortized_inference_ms=inference_seconds * 1000 / len(truth), device=str(device))
    metrics_path = run_dir / 'metrics.json'
    combined = load_config(metrics_path) if metrics_path.exists() else {}
    combined[dataset_name] = metrics
    save_json(metrics_path, combined)
    prediction_path = run_dir / 'predictions.csv'
    existing = []
    if prediction_path.exists():
        with prediction_path.open(encoding='utf-8-sig', newline='') as handle:
            existing = [row for row in csv.DictReader(handle) if row['dataset'] != dataset_name]
    new_rows = [{'dataset': dataset_name, 'image_path': source['image_path'], 'true_label': classes[t],
                 'predicted_label': classes[p], 'predicted_index': p, 'confidence': confidence, 'correct': t == p}
                for source, t, p, confidence in zip(source_rows, truth, predicted, scores)]
    fields = ['dataset', 'image_path', 'true_label', 'predicted_label', 'predicted_index', 'confidence', 'correct']
    with prediction_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + new_rows)
    return metrics


def benchmark_model(model, image_size=224, device='cpu', repeats=30, warmup=5):
    """Batch-size-one forward latency; excludes disk loading and preprocessing."""
    if repeats < 1 or warmup < 0:
        raise ValueError('repeats 必须为正整数，warmup 不得为负。')
    device = get_device(str(device))
    model.to(device).eval()
    example = torch.zeros(1, 3, image_size, image_size, device=device)
    measurements = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            _synchronize(device)
            started = time.perf_counter()
            model(example)
            _synchronize(device)
            if index >= warmup:
                measurements.append((time.perf_counter() - started) * 1000)
    return {'single_image_mean_ms': float(np.mean(measurements)), 'single_image_std_ms': float(np.std(measurements)),
            'device': str(device), 'repeats': repeats, 'image_size': image_size}
