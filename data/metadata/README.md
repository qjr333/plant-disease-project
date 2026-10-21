# 元数据用途

本目录同时保存原始来源信息与本轮八分类生成结果，二者用途不同。

## 来源与标注证据（保留）

- `file_manifest.csv`：原始路径、当前路径、来源类别、文件哈希与整理记录；覆盖整个原始数据，不只是本研究八类。
- `class_mapping.csv`、`class_counts.csv`、`summary.json`、`quality_report.json`、`duplicate_review.csv`：原始数据整理阶段的类别对应、数量、完整性与重复记录。它们不是本轮实验指标。
- `plantvillage_leaf_map.json`：已有官方叶片分组的本地映射副本；仅在原始路径、类别命名空间和当前文件哈希相符时复用。来源见 `configs/labels.yaml`。
- `confirmed_same_photo_pairs.json`：45 条已有的同照片版本人工比较证据。新流程按传递关系合并，同类去一，跨类冲突整组排除；没有声称重新完成全量近重复筛查。
- `source_review.csv`：此前部分 PlantDoc 类别的内容初筛原始注记，只作追溯，不是本轮八类逐图审核或病理确诊。
- `source_exclusions.csv`：从已有注记提取的明确内容或标签疑点。移除了旧的“缺少官方分组即排除”和重复代表选择政策，保留混合理由中的标签疑点。新流程将明确疑点传播到已确认同照片组件。
- `source_evidence/plantdoc_visual_review/`：上述初筛的联系图与注记。引用其中路径时使用 `file_manifest.csv` 查找当前文件。

## 八分类自动生成结果

- `target_class_counts.csv`：当前八类、两个数据集及 PlantDoc 两来源的原始数量。
- `excluded_samples.csv`：本次实际排除的路径、原因与重复代表；只影响清单，原图仍保留。
- `data_quality.json`：解码/重复/标签冲突处理、官方分组覆盖与已知限制。
- `../splits/split_summary.json`：六份清单的类别数量、比例、检查与数据指纹。

完全重复按文件哈希与解码后 RGB 像素哈希识别。重复代表优先保留 PlantVillage，避免训练与外测重叠；优先保留有官方分组的代表。PlantDoc 同类重复优先保留原 `test` 来源，其余记入排除表，因此来源子集数量已受整体去重影响。两来源结果并非相互独立的重复实验。

已知官方叶片组整组划分；来源未知的图像使用单文件标识，不能据此宣称生物叶片身份独立。`validate_partitions` 核查清单一致性；实际文件解码与哈希计算由 `prepare_manifests` 完成。
