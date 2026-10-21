"""运行验收烟测，不进行正式训练或最终测试：python tests/smoke_check.py。"""
from pathlib import Path
import ast
import csv
import gc
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import environment_info, get_device, load_base_config, load_config, load_labels, save_json
from src.data import (MANIFEST_COLUMNS, PARTITIONS, build_transforms, make_loader, prepare_manifests,
                      read_manifest, validate_partitions)
from src.evaluate import evaluate_checkpoint, load_trained_model, predict_image
from src.experiments import run_statuses, stage_results
from src.models import build_model, keep_frozen_batchnorm_eval, parameter_counts, set_finetuning
from src.train import run_experiment

import torch


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    report = {"purpose": "软件验收；不是正式八分类研究指标", "environment": environment_info(get_device())}
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if any(part in (".venv", ".git", "__pycache__") for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    report["python_syntax"] = "passed"
    classes = load_labels(PROJECT_ROOT)
    class_names = [item["label"] for item in classes]
    report["labels"] = {"count": len(classes), "order": class_names}
    print("扫描实际图像、生成并核查固定清单……", flush=True)
    summary = prepare_manifests(PROJECT_ROOT)
    report["data_preparation"] = summary
    report["partition_validation"] = validate_partitions(PROJECT_ROOT)
    config = load_base_config(PROJECT_ROOT)
    transform = build_transforms("none", image_size=config["data"]["image_size"])
    loader_counts = {}
    for name in PARTITIONS:
        loader = make_loader(f"data/splits/{name}.csv", PROJECT_ROOT, transform, batch_size=16, num_workers=0)
        count = 0
        for images, labels in loader:
            assert images.shape[1:] == (3, 224, 224)
            assert torch.isfinite(images).all() and int(labels.min()) >= 0 and int(labels.max()) < 8
            count += len(labels)
        assert count == len(loader.dataset)
        loader_counts[name] = count
        print(f"完整迭代 {name}: {count}", flush=True)
    report["all_dataloader_iterations"] = loader_counts
    inputs, targets = torch.randn(2, 3, 224, 224), torch.tensor([0, 7])
    model_results = []
    for name in ("resnet18", "mobilenet_v2", "efficientnet_b0"):
        model = build_model(name, weights="IMAGENET1K_V1")
        for strategy in ("freeze", "partial", "full"):
            model.zero_grad(set_to_none=True)
            set_finetuning(model, name, strategy)
            model.train()
            keep_frozen_batchnorm_eval(model)
            frozen_bn = {key: value.clone() for key, value in model.named_buffers()
                         if key.rsplit(".", 1)[0] in model._frozen_batchnorm_names}
            logits = model(inputs)
            assert tuple(logits.shape) == (2, 8)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            buffers = dict(model.named_buffers())
            assert all(torch.equal(value, buffers[key]) for key, value in frozen_bn.items())
            model_results.append({"model": name, "strategy": strategy, "output": [2, 8], **parameter_counts(model),
                                  "backward": "passed", "frozen_batchnorm": "passed"})
        del model
        gc.collect()
        print(f"三策略前向与反向通过：{name}", flush=True)
    report["models"] = model_results
    config["device"] = "cpu"
    config["training"].update(epochs=1, batch_size=8)
    config["model"].update(name="resnet18", finetuning="freeze")
    config["experiment_stage"] = "E1"
    print("执行八类真实图像的小规模训练和保存/加载……", flush=True)
    run_dir = run_experiment(PROJECT_ROOT, config, smoke_limit=2)
    status = load_config(run_dir / "status.json")
    assert status["status"] == "completed" and status["smoke_test"]
    model, checkpoint = load_trained_model(run_dir)
    rows = read_manifest(PROJECT_ROOT / "data/splits/pv_val.csv")
    selected = [next(row for row in rows if row["label_index"] == label) for label in range(8)]
    manifest = run_dir / "smoke_validation.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(selected)
    metrics = evaluate_checkpoint(PROJECT_ROOT, run_dir, manifest, "smoke_validation")
    assert metrics["sample_count"] == 8 and len(metrics["confusion_matrix"]) == 8
    assert metrics["smoke_test"]
    prediction = predict_image(model, PROJECT_ROOT / selected[0]["image_path"], transform, class_names)
    assert len(prediction["probabilities"]) == 8
    assert run_dir.name not in {row["run_id"] for row in stage_results(PROJECT_ROOT, "E1")}
    report["smoke_training"] = {"run_id": run_dir.name, "status": status["status"], "sample_counts": status["sample_counts"],
                                "checkpoint_reload": "passed", "evaluation_on_validation_subset": "passed",
                                "inference": "passed", "excluded_from_formal_selection": True}
    report["run_statuses"] = run_statuses(PROJECT_ROOT)
    report["limitations"] = ["本次环境 MPS 不可用，实际训练与前反向验收使用 CPU；Apple Silicon 的 MPS 实际运行仍需本机验证。",
                             "未运行完整正式实验；未使用独立测试或 PlantDoc 指标调参。",
                             "可选 Grad-CAM/Gradio 依赖未安装时不进行实际界面和热力图验收。"]
    report["elapsed_seconds"] = time.perf_counter() - started
    save_json(PROJECT_ROOT / "reports/refactor_validation.json", report)
    print(json.dumps({"status": "passed", "data_counts": summary["counts"], "smoke_run": run_dir.name,
                      "seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
