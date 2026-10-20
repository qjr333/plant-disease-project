"""Optional Grad-CAM support used only after a real model has been trained."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .models import gradcam_target_layer
from .utils import get_device


def gradcam_image(model, model_name, image, transform, target_class=None, device='cpu'):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as error:
        raise ImportError('Grad-CAM 是后期可选功能，请先安装 requirements-demo.txt。') from error
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    device = get_device(str(device))
    model.to(device).eval()
    # Evaluation transforms use a fixed direct resize, matching the displayed image.
    inputs = transform(image.convert('RGB')).unsqueeze(0).to(device).requires_grad_(True)
    width, height = inputs.shape[-1], inputs.shape[-2]
    background = np.asarray(image.convert('RGB').resize((width, height)), dtype=np.float32) / 255.0
    targets = None if target_class is None else [ClassifierOutputTarget(int(target_class))]
    with torch.enable_grad():
        with GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_name)]) as cam:
            grayscale = cam(input_tensor=inputs, targets=targets)[0]
    return Image.fromarray(show_cam_on_image(background, grayscale, use_rgb=True))
