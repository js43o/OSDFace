import os
import random
import torch
import numpy as np
from safetensors.torch import load_file
from torchvision.transforms.functional import (
    rgb_to_grayscale,
    resize,
    InterpolationMode,
)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ArcFace 입력 이미지 변환 메서드
# RGB [0, 1] -> Grayscale [0, 1]
def process_arcface_input(image: torch.Tensor):
    output = resize(image, size=(128, 128), interpolation=InterpolationMode.BICUBIC)
    output = rgb_to_grayscale(output, num_output_channels=1)

    return output


def d_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(torch.relu(1.0 - real_logits))
    loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return 0.5 * (loss_real + loss_fake)


def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)


def initialize_e2f_weight(ckpt_path, e2f_model, fuse_type):
    ckpt = load_file(ckpt_path)
    new_ckpt = {}

    if fuse_type in ["addition", "cross_attn"]:
        for name, tensor in ckpt.items():
            if name.startswith("encoders.") or name.startswith("intro."):
                new_ckpt["lq_" + name] = tensor
                new_ckpt["mq_" + name] = tensor
            else:
                new_ckpt[name] = tensor

    elif fuse_type == "concat":
        for name, tensor in ckpt.items():
            if name.startswith("encoders.") or name.startswith("intro."):
                new_ckpt["lq_" + name] = tensor
                new_ckpt["mq_" + name] = tensor
            else:
                continue  # skip layers of different shapes

    e2f_model.load_state_dict(new_ckpt, False)
