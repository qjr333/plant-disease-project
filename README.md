# 基于迁移学习的番茄叶片病害识别与跨场景泛化优化研究

本项目以 Jupyter Notebook 为主要实验入口，研究番茄叶片的七种病害与健康状态，共八类。使用 ImageNet 预训练网络，在 PlantVillage 上训练并选择模型，再对 PlantVillage 独立测试集和 PlantDoc 真实场景图片进行最终评价。八分类正式实验尚需实际训练，不预填指标或沿用历史实验结论。

## 标签与数据来源

唯一正式标签文件是 `configs/labels.yaml`。编号与顺序在训练、验证、测试、模型保存和演示中一致。

| 编号 | 中文类别 | PlantVillage 目录 | PlantDoc 目录 |
| --- | --- | --- | --- |
| 0 | 番茄细菌斑病 | tomato_bacterial_spot | tomato_leaf_bacterial_spot |
| 1 | 番茄早疫病 | tomato_early_blight | tomato_early_blight_leaf |
| 2 | 番茄晚疫病 | tomato_late_blight | tomato_leaf_late_blight |
| 3 | 番茄叶霉病 | tomato_leaf_mold | tomato_mold_leaf |
| 4 | 番茄斑枯病 | tomato_septoria_leaf_spot | tomato_septoria_leaf_spot |
| 5 | 番茄花叶病毒病 | tomato_tomato_mosaic_virus | tomato_leaf_mosaic_virus |
| 6 | 番茄黄化曲叶病毒病 | tomato_tomato_yellow_leaf_curl_virus | tomato_leaf_yellow_virus |
| 7 | 健康番茄叶片 | tomato_healthy | tomato_leaf |

`tomato_leaf → 健康` 和 `tomato_leaf_yellow_virus → 黄化曲叶病毒病` 为本研究的**暂定语义映射**。现有原始说明未严格证明这两项等价，部分原始文件名也提示潜在错标；源标签沿用，不以模型预测直接改标。应在报告中保留这一限制，不能称已完成逐图病理学验证。[PlantDoc 源仓库](https://github.com/pratikkayal/PlantDoc-Dataset)及[数据集论文](https://arxiv.org/html/1911.10317)提供来源背景，论文也讨论标签可能有误的问题。

- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)：使用 `data/plantvillage/color/` 中指定八类。当前原始目标图像共 **15,080** 张。灰度、分割版本与其他类别原图保留，但不加入本研究样本。
- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)：指定八类原始目录文件数为 `train` 来源 **675** 张、`test` 来源 **69** 张，合计 **744** 张。这是清洗前数量，真实有效数由数据准备报告给出。
- PlantDoc 全部仅用于外部评价。其 `train` 目录是数据集来源标识，**不属于本项目模型训练集**；不能用于早停、选择模型、增强或超参数。`tomato_two_spotted_spider_mites_leaf` 不纳入研究。

已有数据无需重新下载。新环境应从上述数据源获取数据，按标签表将原目录名称映射为当前目录名；现有 `data/metadata/class_mapping.csv` 等来源元数据保留供追溯。不要将 PlantVillage 彩色、灰度和分割版本混合作为独立样本。

## 数据准备与使用边界

01 Notebook 调用 `src/data.py` 扫描实际图片，检查能否打开，计算文件与解码像素指纹，排除明确损坏和完全重复样本，并记录跨类别冲突及处置。排除只影响研究清单，原图不删除、不改写。

PlantVillage 按类别分层，以种子 **42** 生成约 **70% / 15% / 15%** 的训练、验证、独立测试清单；有可靠来源组时整组分配。现存官方叶片分组对原始八类精确唯一匹配 **9,735 / 15,080** 张，花叶病毒类缺少匹配。清洗后覆盖数、未知来源数与分配比例以本次质量报告为准。未知来源样本使用独立标识回退，因此不能宣称完全消除了同源叶片或近重复泄漏；哈希检查只能确认可识别的完全重复问题。

PlantDoc 两个来源先整体去重，再输出两个来源子集与合并外部集合。清单保留 `original_split`。最终评价同时报告来源 `train`、来源 `test` 和合并集合；合并结果不是两次独立试验。

生成文件包括：

```text
data/splits/
├── pv_train.csv
├── pv_val.csv
├── pv_test.csv
├── plantdoc_external_train.csv   # 仅 train 来源外部评价
├── plantdoc_external_test.csv    # 仅 test 来源外部评价
├── plantdoc_external_all.csv     # 全体去重后的外部评价
└── split_summary.json
```

各清单记录相对图片路径、统一标签、编号、来源组、原始 split 和图像指纹。质量报告与排除记录保存在 `data/metadata/`；无需逐张人工填写审核状态。历史来源/标注元数据可以保留作追溯，正式实验仅使用新八分类清单。01 完成后保持清单固定；数据变化后应重新准备并重新训练，不能把旧模型直接视作当前实验结果。

## 项目目录

```text
plant-disease-project/
├── configs/                 # base.yaml、labels.yaml、experiments.yaml
├── notebooks/               # 01—05 主实验；06—07 可选解释和演示
├── src/                     # 数据、模型、训练、实验管理、评价与解释
├── data/
│   ├── plantvillage/        # 原图只读保留
│   ├── plantdoc/            # 原图、来源划分与原说明保留
│   ├── metadata/            # 来源、质量、去重和分组信息
│   └── splits/              # 新八分类固定清单
├── outputs/
│   ├── runs/               # 每次训练独立 run_id
│   ├── tables/             # 验证比较、逐类指标、错误分析
│   └── figures/            # 训练曲线、混淆矩阵、可选热力图
├── demo/                   # 八类演示名称及可选示例
├── reports/                # 实验记录与报告资料
├── requirements.txt
└── requirements-demo.txt
```

`reports/研究方案.docx` 保留为历史规划材料，其旧类别与旧实验计划不作为现行配置；本 README、八分类配置和实际运行记录定义当前实验。 `.git`、开发环境和原始图像均保留。

## 安装环境与 VSCode

建议使用 Python 3.11 或兼容的独立环境。已有 `.venv` 可直接使用，首次安装时在项目根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name plant-disease --display-name "Python (plant-disease)"
```

Windows 激活命令为 `.venv\Scripts\activate`。在 VSCode 安装 Python、Jupyter 扩展，打开**项目文件夹**，为 Notebook 选择项目 `.venv` 对应内核。Notebook 会从当前目录及父目录寻找项目，支持从项目根目录或 `notebooks/` 启动。也可使用 `jupyter lab`。

本次实际验收使用已有 `.venv`：Python 3.9.6、PyTorch 2.8.0、torchvision 0.23.0；`requirements.txt` 固定本次直接依赖版本。新建其他 Python 版本环境时仍需验证依赖兼容性。

Apple Silicon 优先使用 MPS，必要时回退 CPU。第一次正式训练可能联网下载三种 ImageNet 权重；关闭训练开关不会下载。实际 Python、PyTorch、torchvision、设备与配置随每次训练保存，依赖解析版本以运行环境记录为准。

## 按顺序完成实验

每本 Notebook 重启内核后从第一格开始运行，不依赖上一本的内存变量。所有保存输出已清空，不展示历史指标。

| 顺序 | Notebook | 操作与完成标志 |
| --- | --- | --- |
| 01 | `01_data_preparation.ipynb` | 全部依次运行；第 2 节实际扫描并生成固定清单，查看数量、质量报告、分布图和加载检查 |
| 02 | `02_backbone_comparison.ipynb` | 比较三个骨干；在第 1 节设置 `RUN_TRAINING=True`，运行训练与结果单元；查看结果后设置 `SAVE_SELECTION=True` 并运行第 4 节保存 E1 选择 |
| 03 | `03_finetuning_comparison.ipynb` | 固定 E1 骨干，按同样开关比较 Freeze / Partial / Full，并保存 E2 选择 |
| 04 | `04_augmentation_ablation.ipynb` | 固定 E1 / E2，比较 None / Basic / Combined，并保存 E3 最终选择 |
| 05 | `05_cross_domain_evaluation.ipynb` | 读取选定 E3；设置 `RUN_EVALUATION=True` 后评价独立测试与外部数据，生成指标、混淆矩阵及错误表 |
| 06（可选） | `06_gradcam_error_analysis.ipynb` | 安装可选依赖，设置 `RUN_ANALYSIS=True`，展示最终模型的真实 Grad-CAM 案例 |
| 07（可选） | `07_gradio_demo.ipynb` | 设置 `LAUNCH_DEMO=True` 启动本地演示；停止时设置 `CLOSE_DEMO=True` 并运行停止单元 |

02—04 的 `RUN_TRAINING` 默认均为 `False`。查看已有结果时保持关闭。需要开启训练或保存选择时，修改对应开关后先重新运行该开关单元，再运行后续单元。07 默认仅本机启动，不开启外网分享。

## 配置与实验设计

- `configs/base.yaml`：默认图像 224、batch size 16、种子 42、最多 10 轮、AdamW、交叉熵、Early Stopping，以 `val_macro_f1` 选择最佳模型。`training.lr_by_strategy` 为不同微调策略设置合理的学习率，实际值随 run 保存；修改训练预算应在实验前统一确定。
- `configs/experiments.yaml`：E1 固定 Freeze / Basic 比较三种骨干；E2 固定选定骨干和 Basic 比较三种微调；E3 固定骨干和微调比较三种增强。每个候选从 ImageNet 权重重新初始化。
- `selected_backbone`、`selected_finetuning`、`selected_augmentation` 与 `selected_runs` 初始未选择。必须用本轮八类验证结果逐步选定；不继承历史选择。
- 部分微调的解冻范围在 `src/models.py` 定义。None 仅做确定性预处理，Basic 为几何增强，Combined 增加颜色扰动；具体参数以源码和保存配置为准。

各次训练使用独立 `outputs/runs/<run_id>/`，保存 `config.yaml`、`best.pt`、`history.csv`、运行状态及复现信息。历史至少包含 epoch、训练/验证 loss 与 accuracy、验证 Macro-F1、训练耗时。当前配置与数据指纹一致的完成候选自动跳过；中断候选以新 run 重试，不覆盖既有记录。状态文件与不完整文件可识别已完成、异常和中断实验；重试不承诺从中断 epoch 续训。

评价生成该 run 的 `metrics.json`、`predictions.csv`，以及 `outputs/tables/` 和 `outputs/figures/` 中的表图。模型和下一阶段选择都会校验八类顺序、清单指纹与实验状态，烟测不参与正式选择。

评价包括 Accuracy、Macro Precision、Macro Recall、Macro-F1、逐类 Precision / Recall / F1、混淆矩阵和训练耗时。只按 PlantVillage 验证集选择，两个最终测试来源不得用于调参。不要只比较总体 Accuracy；同时报告类别数量、Macro-F1 和逐类表现。

## 运行检查

在项目根目录、项目环境中可重跑：

```bash
python -B -m unittest discover -s tests -v
python -B tests/smoke_check.py
```

单元测试检查质量处理、已知来源分组、可复现划分及阶段选择。烟测会重新扫描数据、完整迭代六份清单，检查三个骨干与三种微调的前向/反向、冻结 BatchNorm，再用每类 2 张训练图和 1 张验证图跑一轮，验证保存、加载、推理和验证子集评价；明确标记 `smoke_test`，不参与正式模型选择。烟测不运行完整正式实验，也不计算最终测试指标。

`reports/refactor_changes.json` 记录本次清理与原始数据保留检查，`reports/refactor_validation.json` 记录实际验收及限制。`data/metadata/README.md` 说明来源证据与新生成结果的区别。可选 Grad-CAM、Gradio 需要另装依赖后验证。

可选解释与演示安装：

```bash
python -m pip install -r requirements-demo.txt
```

演示仅显示模型输出，不提供未核实的防治建议；`demo/disease_info.json` 与正式八类编号一致。

## 已知限制与记录原则

PlantDoc 样本较少、类别分布不均，背景、采集条件与标签定义可能不同；两个暂定类别映射尤其需要谨慎。未知来源样本、近重复以及原始标注噪声仍可能影响结果。默认仅随机种子 42 的一次实验，不能由此证明统计显著性或广泛田间适用性。

`reports/实验记录.md` 用于补充真实运行条件、选择依据与结论；尚未训练的指标保持空白。Grad-CAM 只用于观察模型敏感区域，不是病理学验证。原始数据集、环境、权重和缓存由 `.gitignore` 排除，不自行修改 Git 历史或推送远程。
