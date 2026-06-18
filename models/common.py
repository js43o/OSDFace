# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/arch_util.py
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import init as init


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


"""
# 🔥 exp_08 (fusion_adapter)
class LatentResidualAdapter(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=32, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

        # 중요: 처음에는 delta = 0에 가깝게 시작
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mq_latent, lq_latent):
        delta = self.net(torch.cat([mq_latent, lq_latent], dim=1))
        return mq_latent + delta
"""


# 🔥 exp_09 (direct_fusion)
class LatentResidualAdapter(nn.Module):
    def __init__(self, hidden_channels=32, alpha=0.8, delta_scale=0.1, delta_bound=3.0):
        super().__init__()
        self.alpha = alpha
        self.delta_scale = delta_scale
        self.delta_bound = delta_bound

        self.net = nn.Sequential(
            nn.Conv2d(8, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 4, 3, padding=1),
        )

        # nn.init.zeros_(self.net[-1].weight)
        # nn.init.zeros_(self.net[-1].bias)

    def forward(self, mq_latent, lq_latent):
        base = (1.0 - self.alpha) * mq_latent + self.alpha * lq_latent

        delta = self.net(torch.cat([mq_latent, lq_latent], dim=1))

        # delta 자체를 부드럽게 제한
        delta = self.delta_bound * torch.tanh(delta / self.delta_bound)

        fused = base + self.delta_scale * delta
        return fused
