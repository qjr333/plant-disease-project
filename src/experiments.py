"""E1→E2→E3 实验组织、验证集选择与运行状态汇总。"""
from __future__ import annotations

import copy
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import dataset_fingerprint
from .train import run_experiment
from .utils import load_config, load_labels, save_json


def load_experiment_configs(project_root):
    root = Path(project_root)
    return load_config(root / "configs/base.yaml"), load_config(root / "configs/experiments.yaml")


def _stage_name(stage):
    stage = str(stage).upper()
    if stage not in ("E1", "E2", "E3"):
        raise ValueError("stage 只能是 E1、E2 或 E3。")
    return stage


def _signature(config):
    config = copy.deepcopy(config)
    config.pop("smoke_test", None)
    config.pop("smoke_limit", None)
    return json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stage_configs(project_root, stage):
    root, stage = Path(project_root), _stage_name(stage)
    base, plan = load_experiment_configs(root)
    parent = None
    if stage == "E1":
        settings = [(name, "freeze", "basic") for name in plan["baseline_models"]]
    elif stage == "E2":
        parent = selected_run(root, "E1")
        settings = [(plan["selected_backbone"], strategy, "basic") for strategy in plan["finetuning_strategies"]]
    else:
        parent = selected_run(root, "E2")
        settings = [(plan["selected_backbone"], plan["selected_finetuning"], augmentation) for augmentation in plan["augmentations"]]
    configurations = []
    for name, strategy, augmentation in settings:
        for seed in plan.get("seeds", [42]):
            config = copy.deepcopy(base)
            config.update(seed=int(seed), experiment_stage=stage, augmentation=augmentation, smoke_test=False)
            config["model"].update(name=name, finetuning=strategy)
            config["training"]["monitor"] = "val_macro_f1"
            learning_rates = config["training"].get("lr_by_strategy", {})
            if strategy in learning_rates:
                config["training"]["lr"] = float(learning_rates[strategy])
            if parent is not None:
                config["parent_run_id"] = parent.name
            configurations.append(config)
    if not configurations:
        raise ValueError(f"{stage} 未配置任何实验候选。")
    return configurations


def run_statuses(project_root):
    """显示所有运行，包括未完成、失败和烟测；它们不会进入正式结果表。"""
    rows = []
    for run_dir in sorted((Path(project_root) / "outputs/runs").glob("*")):
        if not run_dir.is_dir():
            continue
        try:
            config = load_config(run_dir / "config.yaml")
            status = load_config(run_dir / "status.json")
            rows.append({"run_id": run_dir.name, "stage": config.get("experiment_stage"),
                         "model": config.get("model", {}).get("name"), "finetuning": config.get("model", {}).get("finetuning"),
                         "augmentation": config.get("augmentation"), "status": status.get("status", "unknown"),
                         "smoke_test": bool(config.get("smoke_test", False)), "epochs_completed": status.get("epochs_completed", 0),
                         "seconds": status.get("seconds"), "error": status.get("error")})
        except (OSError, ValueError, KeyError) as error:
            rows.append({"run_id": run_dir.name, "status": "invalid_record", "error": str(error)})
    return rows


def stage_results(project_root, stage):
    """只汇总完成的、正式的、标签/数据/当前阶段配置均匹配的验证结果。"""
    root, stage = Path(project_root), _stage_name(stage)
    expected = {_signature(config) for config in _stage_configs(root, stage)}
    fingerprint = dataset_fingerprint(root)
    class_names = [item["label"] for item in load_labels(root)]
    rows = []
    for run_dir in sorted((root / "outputs/runs").glob("*")):
        if not run_dir.is_dir():
            continue
        try:
            config = load_config(run_dir / "config.yaml")
            status = load_config(run_dir / "status.json")
            labels = load_config(run_dir / "label_mapping.json")
            if (config.get("experiment_stage") != stage or config.get("smoke_test", False)
                    or status.get("status") != "completed" or status.get("smoke_test", False)
                    or status.get("dataset_fingerprint") != fingerprint
                    or labels.get("class_names") != class_names or _signature(config) not in expected
                    or not (run_dir / "best.pt").is_file() or status.get("monitor") != "val_macro_f1"):
                continue
            metrics = status["validation_metrics"]
            score = float(metrics["macro_f1"])
            if not math.isfinite(score):
                continue
            rows.append({"run_id": run_dir.name, "stage": stage, "model": config["model"]["name"],
                         "finetuning": config["model"]["finetuning"], "augmentation": config["augmentation"],
                         "seed": config["seed"], "lr": config["training"]["lr"], "val_macro_f1": score,
                         "val_accuracy": metrics["accuracy"], "val_loss": metrics["loss"], "best_epoch": status["best_epoch"],
                         "epochs_completed": status["epochs_completed"], "seconds": status["seconds"],
                         "dataset_fingerprint": fingerprint})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: (-row["val_macro_f1"], row["run_id"]))


def run_stage(project_root, stage):
    """依次执行缺少有效完成结果的候选；每次新运行使用独立 run_id。"""
    root, stage = Path(project_root), _stage_name(stage)
    complete = {row["run_id"] for row in stage_results(root, stage)}
    signatures = {_signature(load_config(root / "outputs/runs" / name / "config.yaml")) for name in complete}
    runs = []
    for config in _stage_configs(root, stage):
        if _signature(config) in signatures:
            continue
        run_id = (f"{stage.lower()}_{config['model']['name']}_{config['model']['finetuning']}_{config['augmentation']}"
                  f"_s{config['seed']}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}")
        runs.append(run_experiment(root, config, run_id))
    return runs


def select_best(project_root, stage):
    """仅按验证 Macro-F1 选择；平局按 run_id 字典序，阶段候选必须全部完成。"""
    root, stage = Path(project_root), _stage_name(stage)
    rows = stage_results(root, stage)
    completed = {_signature(load_config(root / "outputs/runs" / row["run_id"] / "config.yaml")) for row in rows}
    expected = {_signature(config) for config in _stage_configs(root, stage)}
    if not expected.issubset(completed):
        raise ValueError(f"{stage} 尚有 {len(expected - completed)} 个正式候选未完成，请先完成全部候选再选择。")
    winner = rows[0]
    _, plan = load_experiment_configs(root)
    selections = plan.setdefault("selected_runs", {"e1": None, "e2": None, "e3": None})
    changed = selections.get(stage.lower()) != winner["run_id"]
    selections[stage.lower()] = winner["run_id"]
    if stage == "E1":
        plan["selected_backbone"] = winner["model"]
        if changed:
            plan.update(selected_finetuning=None, selected_augmentation=None)
            selections.update(e2=None, e3=None)
    elif stage == "E2":
        plan["selected_finetuning"] = winner["finetuning"]
        if changed:
            plan["selected_augmentation"] = None
            selections["e3"] = None
    else:
        plan["selected_augmentation"] = winner["augmentation"]
    save_json(root / "configs/experiments.yaml", plan)
    return winner


def selected_run(project_root, stage="E3"):
    """返回仍然有效的已选正式运行；阶段依赖、当前数据与配置均重新核对。"""
    root, stage = Path(project_root), _stage_name(stage)
    _, plan = load_experiment_configs(root)
    run_id = plan.get("selected_runs", {}).get(stage.lower())
    if not run_id:
        raise ValueError(f"尚未保存 {stage} 选择，请按 02→03→04 的顺序完成训练和选择。")
    row = next((row for row in stage_results(root, stage) if row["run_id"] == run_id), None)
    if row is None:
        raise ValueError(f"{stage} 已选运行不再有效：必须为当前配置、数据与标签对应的已完成正式实验。")
    if plan.get("selected_backbone") != row["model"]:
        raise ValueError("已选骨干与实验记录不一致，请重新执行阶段选择。")
    if stage in ("E2", "E3") and plan.get("selected_finetuning") != row["finetuning"]:
        raise ValueError("已选微调策略与实验记录不一致。")
    if stage == "E3" and plan.get("selected_augmentation") != row["augmentation"]:
        raise ValueError("已选增强方案与实验记录不一致。")
    return root / "outputs/runs" / run_id
