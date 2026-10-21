"""八分类 ImageNet 迁移学习模型与统一微调策略。"""
from typing import Optional

from torch import nn
from torchvision import models

SUPPORTED_MODELS = ("resnet18", "mobilenet_v2", "efficientnet_b0")
NUM_CLASSES = 8


def _check_name(name):
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"未知模型 {name!r}；可选：{SUPPORTED_MODELS}")


def build_model(name, num_classes=NUM_CLASSES, weights: Optional[str] = "IMAGENET1K_V1", finetuning="freeze"):
    """统一创建八分类模型；weights=None 仅供结构检查或加载已有权重。"""
    _check_name(name)
    if num_classes != NUM_CLASSES:
        raise ValueError("本项目固定为八分类，num_classes 必须为 8。")
    if finetuning not in ("freeze", "partial", "full"):
        raise ValueError("finetuning 只能是 freeze、partial 或 full。")
    model = getattr(models, name)(weights=weights)
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    model.model_name = name
    return set_finetuning(model, name, finetuning)


def set_finetuning(model, name, strategy):
    """partial 解冻 ResNet layer4、MobileNet features[14:]、EfficientNet features[7:]。"""
    _check_name(name)
    if strategy not in ("freeze", "partial", "full"):
        raise ValueError("strategy 只能是 freeze、partial 或 full。")
    for parameter in model.parameters():
        parameter.requires_grad = strategy == "full"
    head = model.fc if name == "resnet18" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True
    if strategy == "partial":
        stage = model.layer4 if name == "resnet18" else model.features[14 if name == "mobilenet_v2" else 7:]
        for parameter in stage.parameters():
            parameter.requires_grad = True
    model._frozen_batchnorm_names = tuple(
        module_name for module_name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and not any(parameter.requires_grad for parameter in module.parameters())
    )
    model.finetuning_strategy = strategy
    keep_frozen_batchnorm_eval(model)
    return model


def keep_frozen_batchnorm_eval(model):
    """model.train() 后固定冻结骨干的 BatchNorm 运行统计。"""
    modules = dict(model.named_modules())
    for name in getattr(model, "_frozen_batchnorm_names", ()):
        modules[name].eval()


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def gradcam_target_layer(model, name):
    _check_name(name)
    return model.layer4[-1] if name == "resnet18" else model.features[-1]
