# 基于迁移学习的植物叶片病害识别与真实场景泛化优化研究

以 Notebook 为主要实验入口，比较预训练骨干、微调方法与增强策略，再用 PlantDoc 评价从标准背景到真实场景的泛化表现。当前目录是可继续填写和运行的项目骨架，尚未完成类别审核、正式数据划分和模型训练，没有预先生成实验结果。

## 文件组织

```text
plant-disease-project/
├── notebooks/               # 01—07：按编号编写、运行、分析实验
├── src/                     # 数据、模型、训练、评价和解释的共用函数
├── configs/                 # 默认参数、统一标签、实验组合
├── data/
│   ├── plantvillage/        # color、grayscale、segmented：保留原始图片
│   ├── plantdoc/            # train、test、docs：保留来源划分
│   ├── metadata/            # 整理记录、重复清单、人工审核和分组信息
│   └── splits/              # 01 Notebook 生成固定的正式划分清单
├── outputs/
│   ├── runs/               # 每次训练独立保存参数、权重和结果
│   ├── figures/            # 训练曲线、混淆矩阵、热力图
│   └── tables/             # 模型对比及跨场景评价表
├── demo/                   # 后期演示信息和示例图
├── reports/                # 研究方案副本、实验记录、参考文献
├── requirements.txt        # 核心实验依赖
└── requirements-demo.txt   # 后期解释和交互演示依赖
```

外层的 `研究方案.docx` 保留原件，`reports/研究方案.docx` 是创建骨架时保存的相同内容副本。原始数据目录保持现状；已有的压缩包也未删除。

## 开始使用

建议使用 **Python 3.11** 建立独立环境，以便运行完整实验和后期 Gradio 演示。当前机器的 Python 3.9 环境可用于阅读和部分基础校验，不能据此认为完整依赖已可用。在项目根目录运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name plant-disease --display-name "Python (plant-disease)"
jupyter lab
```

在 Notebook 中选择上述内核。Windows 环境激活命令为 `.venv\Scripts\activate`。依赖没有宣称为本机已验证的固定版本；首次完成运行后，在实验记录中保存 Python、PyTorch、torchvision 版本与设备信息。首次加载 ImageNet 预训练权重可能需要联网下载，数据集无需重新下载。

## Notebook 顺序

| 文件 | 工作 | 完成标志 |
| --- | --- | --- |
| `01_data_preparation.ipynb` | 数据统计、共同类别选择、重复和冲突审核、来源分组、固定划分 | 生成统一标签和四份正式清单 |
| `02_backbone_comparison.ipynb` | 比较 ResNet18、MobileNetV2、EfficientNet-B0 | 基于验证集比较三种骨干 |
| `03_finetuning_comparison.ipynb` | 比较冻结、部分微调、整体微调 | 根据验证集选择微调方法 |
| `04_augmentation_ablation.ipynb` | 比较无随机增强、几何增强、几何与颜色组合增强 | 根据验证集确定优化方案 |
| `05_cross_domain_evaluation.ipynb` | 对预先锁定的模型进行同域及 PlantDoc 外部评价 | 统一导出指标、逐图预测和结果表 |
| `06_gradcam_error_analysis.ipynb` | 对正确和错误样本做 Grad-CAM 分析 | 生成热力图及分析 |
| `07_gradio_demo.ipynb` | 加载已训练权重并启动演示 | 完成课程展示 |

先完成 **01—05**，再做 **06—07**。每本 Notebook 应重新启动内核后也能从头运行；Notebook 通过配置、清单和已保存文件衔接。训练结果只在实际运行后写入 `outputs/`。

## 需要先确认的数据事项

1. 主实验使用 `data/plantvillage/color`。灰度图、分割图与彩色图之间存在处理版本对应关系，不能合并视为独立样本。
2. 数据整理已保留重复与标签冲突供审核；`metadata/summary.json` 的整理完成状态不代表科研数据清洗或来源分组已完成。对所选类别的每个候选样本，在 `review_decisions.csv` 中明确填写 `keep` 或 `exclude`，并填写非空 `reason`，不能把未填写当作默认保留。
3. 现有 `class_mapping.csv` 仅记录原目录命名。最终统一类别须写入 `configs/labels.yaml`，不能直接用两个来源各自的类别序号训练和测试。
4. `sample_groups.csv` 用于记录来源叶片和重复样本的分组。每个保留样本必须填写非空 `group_id`、`group_method`，完成核对后才填写 `reviewed=true`；处理版本对应 ID 不等同于同一生物叶片 ID。可核对 [PlantVillage 叶片分组资料](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/leaf_grouping)。
5. 正式划分前处理重复和冲突，并将同一来源组放入同一个划分。PlantVillage 目标比例为约 70%／15%／15%，小类别和分组限制可能导致实际比例略有变化。
6. PlantDoc 原有 `train/test` 保留来源含义。本项目可在审核后把匹配类别的合格样本汇总为外部测试清单。PlantDoc 全部不参与训练、调参和早停，也不能根据外部测试成绩回头选择策略。

`data/splits/` 当前只保留空目录标记。审核和分组完成后，由 01 Notebook 真正生成 `pv_train.csv`、`pv_val.csv`、`pv_test.csv`、`plantdoc_external_test.csv`，不填充占位样本。审核或分组不完整时，01 应提示补齐并停止正式划分。

## 配置约定

三个 `.yaml` 文件当前使用 **JSON 语法**保存；JSON 是合法的 YAML 子集，既能通过 Python 标准库 `json` 读取，也能通过 PyYAML 读取。若改写为普通 YAML，则需要 PyYAML。路径均相对于项目根目录。

- `base.yaml`：默认使用 ResNet18、ImageNet 权重、冻结骨干、224×224 输入、基础增强、随机种子 42，默认训练 10 轮。这些是启动实验的初始参数，尚未经本项目验证。
- `labels.yaml`：初始 `status` 为 `pending_review`，`classes` 为空。审核完成后填写实际类别，将 `status` 更新为 `approved`。
- `experiments.yaml`：列出骨干、微调和增强候选项。`selected_backbone` 与 `selected_finetuning` 由验证集结果确定；`evaluation_runs` 保存预先选定的实验目录，外部评价前将 `evaluation_locked` 设为 `true`。

`labels.yaml` 中 `classes` 每一项的字段：

| 字段 | 含义 |
| --- | --- |
| `label` | 项目统一类别名，建议稳定的英文标识 |
| `label_index` | 从 0 开始的连续整数，同一模型全程保持一致 |
| `plantvillage_class` | 对应 PlantVillage 实际类别目录名 |
| `plantdoc_classes` | 对应 PlantDoc 实际类别目录名列表，可对应多个来源名称 |
| `display_name` | 可选，用于图表和演示的中文类别名 |

正式清单至少记录 `image_path`、`label`、`label_index`、`group_id`、`dataset`、`original_split`。图片路径相对于项目根目录。用于训练、验证、测试的类别编号必须与所保存的模型标签映射一致。

## 结果与演示

一次实验一个 `outputs/runs/<实验名>/` 目录；保存实际配置、标签顺序、运行环境、最佳权重、训练历史，正式评价后再保存指标和逐图预测。相同实验名重复运行时，应保留历史结果或使用新名称。训练、验证和外部测试的记录应能追溯到固定清单。

后期解释与演示按需安装：

```bash
python -m pip install -r requirements-demo.txt
```

`demo/disease_info.json` 暂为空，最终统一类别确定后再补中文名称及有来源依据的养护参考，避免演示使用不匹配的标签或虚构建议。课程研究报告、答辩幻灯片、权重和指标均应在实际完成后创建。

演示资料填写在 `classes` 列表中，每项使用 `label` 对应模型类别、`reference_text` 保存展示文字、`source` 记录参考来源；没有填写的类别不会显示养护文字。

## 参考

- [研究方案副本](reports/研究方案.docx)
- [PyTorch 官方迁移学习教程](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch Notebook 与共用模块组织示例](https://github.com/mrdbourke/pytorch-deep-learning)
- [植物病害识别相关项目](https://github.com/Jenpoer/plant-disease-detection)
- [PlantVillage 数据集](https://github.com/spMohanty/PlantVillage-Dataset)
- [PlantDoc 数据集](https://github.com/pratikkayal/PlantDoc-Dataset)

参考项目用于组织和方法借鉴，实际设置以本研究方案、可核对的数据和最终实验记录为准。
