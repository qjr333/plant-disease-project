"""One training implementation shared by the backbone and ablation notebooks."""
from __future__ import annotations

import copy
import csv
import math
import re
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from .data import build_transforms, make_loader, read_manifest
from .models import build_model, keep_frozen_batchnorm_eval, parameter_counts
from .utils import approved_labels, environment_info, get_device, save_json, set_seed


def _check_labels(rows, class_names):
    for row in rows:
        index = int(row['label_index'])
        if not 0 <= index < len(class_names) or row['label'] != class_names[index]:
            raise ValueError(f'清单标签与当前共同类别配置不一致：{row.get("image_path")}')


def _epoch(model, loader, criterion, device, num_classes, optimizer=None):
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
                raise RuntimeError('训练损失出现非有限值，请检查学习率及输入。')
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not count:
        raise ValueError('训练或验证清单为空，无法开展实验。')
    return {'loss': total_loss / count,
            'accuracy': sum(t == p for t, p in zip(truth, predictions)) / count,
            'macro_f1': float(f1_score(truth, predictions, labels=list(range(num_classes)), average='macro', zero_division=0))}


def run_experiment(project_root, config, run_id):
    """Train against PlantVillage train/val only; never reads either test set."""
    root = Path(project_root).resolve()
    config = copy.deepcopy(config)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', run_id):
        raise ValueError('run_id 只能包含英文字母、数字、下划线和连字符。')
    classes = approved_labels(root)
    class_names = [item['label'] for item in classes]
    data_config, training = config['data'], config['training']
    train_path, val_path = root / data_config['train_manifest'], root / data_config['val_manifest']
    train_rows, val_rows = read_manifest(train_path), read_manifest(val_path)
    _check_labels(train_rows + val_rows, class_names)
    if {r['group_id'] for r in train_rows} & {r['group_id'] for r in val_rows}:
        raise ValueError('训练与验证出现重叠 group_id，请重新核对数据划分。')
    if {r['image_path'] for r in train_rows} & {r['image_path'] for r in val_rows}:
        raise ValueError('训练与验证存在相同图片路径。')
    if any(r['dataset'] != 'plantvillage' for r in train_rows + val_rows):
        raise ValueError('训练与验证只能包含 PlantVillage，不能使用 PlantDoc。')
    if int(training['epochs']) < 1 or int(training['patience']) < 1:
        raise ValueError('epochs 和 patience 必须为正整数。')
    monitor = training.get('monitor', 'macro_f1')
    if monitor not in {'macro_f1', 'accuracy', 'loss'}:
        raise ValueError('monitor 仅支持 macro_f1、accuracy 或 loss。')
    run_dir = root / 'outputs/runs' / run_id
    if run_dir.exists():
        raise FileExistsError(f'实验目录已存在，请换用新 run_id，避免覆盖：{run_dir}')
    seed = set_seed(config.get('seed', 42))
    device = get_device(config.get('device', 'auto'))
    transform_args = {k: data_config[k] for k in ['image_size', 'mean', 'std'] if k in data_config}
    train_transform = build_transforms(config.get('augmentation', 'basic'), **transform_args)
    val_transform = build_transforms('none', **transform_args)
    loader_args = {'project_root': root, 'batch_size': int(training['batch_size']),
                   'num_workers': int(training.get('num_workers', 0)), 'seed': seed}
    train_loader = make_loader(train_path, transform=train_transform, shuffle=True, **loader_args)
    val_loader = make_loader(val_path, transform=val_transform, shuffle=False, **loader_args)
    if not train_rows or not val_rows:
        raise ValueError('训练与验证集必须均有样本。')
    model_config = config['model']
    model = build_model(model_config['name'], len(classes), weights=model_config.get('weights', 'IMAGENET1K_V1'),
                        finetuning=model_config.get('finetuning', 'freeze')).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=float(training['lr']), weight_decay=float(training.get('weight_decay', 0)))
    criterion = torch.nn.CrossEntropyLoss()
    run_dir.mkdir(parents=True)
    save_json(run_dir / 'config.yaml', config)
    save_json(run_dir / 'environment.json', environment_info(device))
    class_to_idx = {label: i for i, label in enumerate(class_names)}
    save_json(run_dir / 'label_mapping.json', {'class_names': class_names, 'class_to_idx': class_to_idx, 'classes': classes})
    save_json(run_dir / 'status.json', {'status': 'training', 'parameters': parameter_counts(model)})
    history, best_score, stale = [], -math.inf, 0
    started = time.perf_counter()
    try:
        for epoch in range(1, int(training['epochs']) + 1):
            train_metrics = _epoch(model, train_loader, criterion, device, len(classes), optimizer)
            val_metrics = _epoch(model, val_loader, criterion, device, len(classes))
            row = {'epoch': epoch, **{'train_' + k: v for k, v in train_metrics.items()},
                   **{'val_' + k: v for k, v in val_metrics.items()},
                   'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            with (run_dir / 'history.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            score = -val_metrics['loss'] if monitor == 'loss' else val_metrics[monitor]
            if score > best_score:
                best_score, stale = score, 0
                checkpoint = {'model_name': model_config['name'], 'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              'class_names': class_names, 'class_to_idx': class_to_idx, 'config': config,
                              'epoch': epoch, 'monitor': monitor, 'validation_metrics': val_metrics}
                torch.save(checkpoint, run_dir / 'best.pt')
            else:
                stale += 1
            print(f'{run_id} | epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, val_macro_f1={val_metrics["macro_f1"]:.4f}')
            if stale >= int(training['patience']):
                break
    except BaseException as error:
        save_json(run_dir / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                           'error': str(error), 'epochs_completed': len(history)})
        raise
    save_json(run_dir / 'status.json', {'status': 'trained', 'epochs_completed': len(history), 'monitor': monitor,
                                       'best_score': best_score, 'seconds': time.perf_counter() - started})
    return run_dir
