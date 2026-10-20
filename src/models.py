"""TorchVision 骨干网络，以及可复现的冻结与微调策略。

接口依据 PyTorch 官方文档：
https://docs.pytorch.org/vision/stable/models.html
首次使用非空 weights 时，TorchVision 会按需下载预训练权重。
"""

from typing import Optional

from torch import nn
from torchvision import models


SUPPORTED_MODELS = ("resnet18", "mobilenet_v2", "efficientnet_b0")


def _check_name(name: str) -> None:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"未知模型 {name!r}；可选：{SUPPORTED_MODELS}")


def build_model(
    name: str,
    num_classes: int,
    weights: Optional[str] = "IMAGENET1K_V1",
    finetuning: str = "freeze",
) -> nn.Module:
    """替换分类层并设置微调策略；weights=None 可在离线环境构建模型。"""
    _check_name(name)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes 必须是至少为 2 的整数。")
    if finetuning not in ("freeze", "partial", "full"):
        raise ValueError("finetuning 只能是 freeze、partial 或 full。")
    model = getattr(models, name)(weights=weights)
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.model_name = name
    set_finetuning(model, name, finetuning)
    return model


def set_finetuning(model: nn.Module, name: str, strategy: str) -> nn.Module:
    """freeze 仅训练分类层；partial 解冻末阶段；full 解冻全部参数。

    partial 的定义固定为 ResNet layer4、MobileNet features[14:]、
    EfficientNet features[7:]，以及各模型分类层。改变策略后应重建优化器。
    """
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
    # 保存模块名称，避免将模块本身重复注册进模型或 state_dict。
    model._frozen_batchnorm_names = tuple(
        module_name
        for module_name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and not any(parameter.requires_grad for parameter in module.parameters())
    )
    model.finetuning_strategy = strategy
    keep_frozen_batchnorm_eval(model)
    return model


def keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    """每次 model.train() 后调用，防止冻结骨干的 BN 运行统计继续变化。"""
    modules = dict(model.named_modules())
    for name in getattr(model, "_frozen_batchnorm_names", ()):
        modules[name].eval()


def parameter_counts(model: nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """返回具有空间维度的末端特征层。"""
    _check_name(name)
    return model.layer4[-1] if name == "resnet18" else model.features[-1]
