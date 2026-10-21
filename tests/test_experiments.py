"""实验选择约束回归；不启动训练、不依赖真实图像或已有实验。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import experiments
from src.utils import load_config, save_json


class ExperimentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tomato-experiments-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.classes = [f"class_{index}" for index in range(8)]
        save_json(self.root / "configs/labels.yaml", {"classes": [
            {"label": name, "label_index": index, "display_name": name,
             "plantvillage_class": name, "plantdoc_classes": [f"pd_{index}"]}
            for index, name in enumerate(self.classes)
        ]})
        save_json(self.root / "configs/base.yaml", {
            "seed": 42, "model": {"name": "resnet18", "weights": "IMAGENET1K_V1", "finetuning": "freeze"},
            "data": {}, "training": {"lr": .001, "monitor": "val_macro_f1",
                                       "lr_by_strategy": {"freeze": .001, "partial": .0001, "full": .0001}},
            "augmentation": "basic",
        })
        save_json(self.root / "configs/experiments.yaml", {
            "baseline_models": ["resnet18", "mobilenet_v2", "efficientnet_b0"],
            "finetuning_strategies": ["freeze", "partial", "full"],
            "augmentations": ["none", "basic", "combined"], "seeds": [42],
            "selected_backbone": None, "selected_finetuning": None, "selected_augmentation": None,
            "selected_runs": {"e1": None, "e2": None, "e3": None},
        })
        patcher = patch.object(experiments, "dataset_fingerprint", return_value="current-data")
        self.fingerprint = patcher.start()
        self.addCleanup(patcher.stop)

    def add_run(self, config, run_id, score=.5, smoke=False, status="completed", fingerprint="current-data"):
        config = copy.deepcopy(config)
        config["smoke_test"] = smoke
        path = self.root / "outputs/runs" / run_id
        save_json(path / "config.yaml", config)
        save_json(path / "label_mapping.json", {"class_names": self.classes})
        save_json(path / "status.json", {
            "status": status, "smoke_test": smoke, "dataset_fingerprint": fingerprint,
            "monitor": "val_macro_f1", "validation_metrics": {"macro_f1": score, "accuracy": .5, "loss": 1.},
            "best_epoch": 1, "epochs_completed": 1, "seconds": 1.,
        })
        # 存在性检查用占位文件；本测试不会调用模型加载。
        (path / "best.pt").write_bytes(b"unit-test-placeholder")
        return path

    def complete_stage(self, stage):
        for index, config in enumerate(experiments._stage_configs(self.root, stage)):
            self.add_run(config, f"{stage.lower()}_{index}", score=.5 + index / 10)
        return experiments.select_best(self.root, stage)

    def test_stage_order_and_selected_lineage(self):
        with self.assertRaises(ValueError):
            experiments._stage_configs(self.root, "E2")
        with self.assertRaises(ValueError):
            experiments.selected_run(self.root, "E3")
        self.complete_stage("E1")
        with self.assertRaises(ValueError):
            experiments._stage_configs(self.root, "E3")
        self.complete_stage("E2")
        self.complete_stage("E3")
        self.assertEqual(experiments.selected_run(self.root).name, "e3_2")
        config = load_config(self.root / "outputs/runs/e3_2/config.yaml")
        self.assertEqual(config["parent_run_id"], "e2_2")

    def test_smoke_and_unfinished_runs_are_excluded(self):
        configs = experiments._stage_configs(self.root, "E1")
        self.add_run(configs[0], "smoke", .99, smoke=True)
        self.add_run(configs[0], "running", .99, status="running")
        self.assertEqual(experiments.stage_results(self.root, "E1"), [])
        self.assertEqual(len(experiments.run_statuses(self.root)), 2)
        winner = self.complete_stage("E1")
        self.assertEqual(winner["run_id"], "e1_2")

    def test_every_candidate_must_complete_and_ties_are_deterministic(self):
        configs = experiments._stage_configs(self.root, "E1")
        self.add_run(configs[0], "candidate_0")
        with self.assertRaises(ValueError):
            experiments.select_best(self.root, "E1")
        for index, config in enumerate(configs[1:], 1):
            self.add_run(config, f"candidate_{index}")
        self.assertEqual(experiments.select_best(self.root, "E1")["run_id"], "candidate_0")

    def test_changed_fingerprint_rejects_saved_selection(self):
        self.complete_stage("E1")
        self.fingerprint.return_value = "new-data"
        self.assertEqual(experiments.stage_results(self.root, "E1"), [])
        with self.assertRaises(ValueError):
            experiments.selected_run(self.root, "E1")


if __name__ == "__main__":
    unittest.main()
