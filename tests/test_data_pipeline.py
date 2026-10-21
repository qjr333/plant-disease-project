"""小型合成数据回归测试：泄漏检查、损坏/重复排除、可复现划分。"""
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import dataset_fingerprint, prepare_manifests, read_manifest, validate_partitions
from src.utils import load_config


class DataPreparationTests(unittest.TestCase):
    def test_grouped_split_quality_and_reproducibility(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/labels.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/labels.yaml").write_text(json.dumps(config))
            metadata = root / "data/metadata"
            metadata.mkdir(parents=True)
            originals, leaf_map = [], {}
            for item in config["classes"]:
                label = item["label_index"]
                for dataset, split, name in [("plantvillage", "color", item["plantvillage_class"])] + [
                    ("plantdoc", s, item["plantdoc_classes"][0]) for s in ("train", "test")]:
                    folder = root / "data" / dataset / split / name
                    folder.mkdir(parents=True)
                    for i in range(20 if dataset == "plantvillage" else 2):
                        path = folder / f"{i:03d}.png"
                        Image.new("RGB", (12, 12), (label * 25, i * 9, 30 if dataset == "plantvillage" else 70 if split == "train" else 90)).save(path)
                        if dataset == "plantvillage":
                            key = f"leaf_{label}_{i}"
                            original_class = f"Class_{label}"
                            leaf_map[key] = [f"{original_class}:::{i // 2}"]
                            originals.append({"new_path": path.relative_to(root / "data").as_posix(), "dataset": dataset,
                                              "variant": "color", "class_name": name, "original_path": f"raw/uuid___{key}.JPG",
                                              "original_class": original_class, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            first = config["classes"][0]["plantvillage_class"]
            folder = root / "data/plantvillage/color" / first
            # 无来源信息的副本字典序更前时，也应保留有官方叶片分组的原图。
            shutil.copyfile(folder / "000.png", folder / "000-copy.png")
            (folder / "broken.jpg").write_bytes(b"not an image")
            # 不同编码但 RGB 像素完全相同也必须排除。
            with Image.open(folder / "001.png") as image:
                image.save(folder / "same_pixels.png", compress_level=0)
            # 跨类完全重复的整组排除，不能任意选择一个标签。
            paths = [root / "data/plantdoc/train" / config["classes"][i]["plantdoc_classes"][0] / "conflict.png" for i in (1, 2)]
            for path in paths:
                Image.new("RGB", (13, 13), (241, 242, 243)).save(path)
            # 人工确认关系也要传递：三节点照片链跨标签时全部排除。
            photo_paths = []
            for index, (label, split, filename) in enumerate([
                (3, "train", "chain_a.png"), (3, "test", "chain_b.png"), (4, "train", "chain_c.png"),
                (5, "train", "annotated_a.png"), (5, "test", "annotated_b.png"),
                (6, "train", "same_photo_a.png"), (6, "test", "same_photo_b.png"),
            ]):
                path = root / "data/plantdoc" / split / config["classes"][label]["plantdoc_classes"][0] / filename
                Image.new("RGB", (15 + index, 15), (230, 200 + index, 190)).save(path)
                photo_paths.append(path.relative_to(root).as_posix())
            # 这些照片版本内容指纹不同，只能通过已有人工关系识别。
            self.assertEqual(len({hashlib.sha256((root / p).read_bytes()).hexdigest() for p in photo_paths}), 7)
            pairs = [{"a": photo_paths[a], "b": photo_paths[b], "relation": "same_photo_variant"}
                     for a, b in [(0, 1), (1, 2), (3, 4), (5, 6)]]
            (metadata / "confirmed_same_photo_pairs.json").write_text(json.dumps(pairs))
            # 一张有明确内容排除依据时，其同照片版本也不能重新进入评估集。
            with (metadata / "source_exclusions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_path", "reason"])
                writer.writeheader()
                writer.writerow({"image_path": photo_paths[3], "reason": "已确认非叶片摄影内容"})
            with (metadata / "file_manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(originals[0]))
                writer.writeheader()
                writer.writerows(originals)
            (metadata / "plantvillage_leaf_map.json").write_text(json.dumps(leaf_map))
            summary = prepare_manifests(root)
            self.assertEqual(summary["quality"]["exclusion_reasons"]["decode_error"], 1)
            self.assertEqual(summary["quality"]["exclusion_reasons"]["duplicate_label_conflict"], 5)
            self.assertEqual(summary["quality"]["exclusion_reasons"]["source_annotation"], 2)
            self.assertEqual(summary["quality"]["exclusion_reasons"]["duplicate_content_or_confirmed_photo"], 3)
            self.assertEqual(summary["quality"]["pv_known_leaf_group_images"], 160)
            self.assertEqual(summary["counts"]["plantdoc_external_all"], 33)
            external_paths = {r["image_path"] for r in read_manifest(root / "data/splits/plantdoc_external_all.csv")}
            self.assertFalse(external_paths.intersection(photo_paths[:5]))
            self.assertEqual(len(external_paths.intersection(photo_paths[5:])), 1)
            with (metadata / "excluded_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
                reasons = {row["image_path"]: row["reason"] for row in csv.DictReader(stream)}
            self.assertTrue(all(reasons[p] == "duplicate_label_conflict" for p in photo_paths[:3]))
            self.assertTrue(all(reasons[p] == "source_annotation" for p in photo_paths[3:5]))
            self.assertEqual(validate_partitions(root)["status"], "passed")
            fingerprint = dataset_fingerprint(root)
            self.assertEqual(prepare_manifests(root)["fingerprint"], fingerprint)
            for split in ("train", "val", "test"):
                rows = read_manifest(root / f"data/splits/pv_{split}.csv")
                self.assertEqual({r["label_index"] for r in rows}, set(range(8)))
            self.assertTrue((folder / "broken.jpg").is_file())
            self.assertTrue((folder / "000-copy.png").is_file())


if __name__ == "__main__":
    unittest.main()
