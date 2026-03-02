import torch
from torch import nn

from .naf import NAFBlock
from .stn import STNBlock


class NAF_STN_Block(nn.Module):
    def __init__(
        self, in_channel: int, in_resolution: int, num_naf: int, sampling=None
    ):
        super().__init__()

        self.nfbs = nn.Sequential(*[NAFBlock(in_channel) for _ in range(num_naf)])
        self.stn = STNBlock(in_channel, in_resolution)
        if sampling == "down":
            self.sampling = nn.Conv2d(in_channel, in_channel * 2, 2, 2)
        elif sampling == "up":
            self.sampling = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * 2, 1, bias=False), nn.PixelShuffle(2)
            )
        else:
            self.sampling = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.nfbs(x)
        x = self.stn(x)
        x = self.sampling(x)

        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, F_A, F_B):
        B, C, H, W = F_A.shape
        F_A = F_A.flatten(2).transpose(1, 2).contiguous()
        F_B = F_B.flatten(2).transpose(1, 2).contiguous()

        # LayerNorm before attention (Pre-Norm transformer style)
        q = self.norm_q(F_A)
        kv = self.norm_kv(F_B)

        # Multi-head cross-attention
        attn_out, _ = self.attn(q, kv, kv)

        # Output projection
        attn_out = self.proj(attn_out)

        # Add & Norm (residual connection within attention block)
        out = self.norm_out(F_A + attn_out)

        return out.transpose(1, 2).contiguous().reshape(B, C, H, W)


class CrossAttentionWithFeatureKey(nn.Module):
    def __init__(self, q_dim=512, kv_dim=512, heads=8):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.attn = nn.MultiheadAttention(q_dim, heads, batch_first=True)
        self.out_proj = nn.Linear(q_dim, q_dim)

    def forward(self, q_feat, kv_feat):
        B, Cq, Hq, Wq = q_feat.shape
        B, _Ckv, _Hkv, _Wkv = kv_feat.shape

        q = q_feat.flatten(2).transpose(1, 2).contiguous()  # (B, Hq*Wq, Cq)
        kv = kv_feat.flatten(2).transpose(1, 2).contiguous()  # (B, Hkv*Wkv, Ckv)

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        attn_out, _ = self.attn(q, k, v)

        out = self.out_proj(attn_out)
        out = out.transpose(1, 2).contiguous().view(B, Cq, Hq, Wq)

        return out


class CoarseRestorer(nn.Module):
    def __init__(self, res=128, in_channels=3, width=32):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.outro = nn.Conv2d(
            in_channels=width,
            out_channels=3,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.Sequential(
            NAF_STN_Block(width, res, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 2, res // 2, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 4, res // 4, num_naf=4, sampling="down"),
            NAF_STN_Block(width * 8, res // 8, num_naf=8, sampling="down"),
        )
        self.middle_blocks = NAF_STN_Block(width * 16, res // 16, num_naf=8)
        self.decoders = nn.Sequential(
            NAF_STN_Block(width * 16, res // 16, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 8, res // 8, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 4, res // 4, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 2, res // 2, num_naf=2, sampling="up"),
        )
        self.norm = nn.Tanh()

    def forward(self, x: torch.Tensor):
        enc_skips = []

        x = self.intro(x)
        for encoder in self.encoders:
            x = encoder(x)
            enc_skips.append(x)

        x = self.middle_blocks(x)
        for decoder, enc_skip in zip(self.decoders, enc_skips[::-1]):
            x = x + enc_skip
            x = decoder(x)

        x = self.outro(x)
        x = (self.norm(x) + 1.0) / 2.0  # normalize to [0, 1]

        return x


class DualEncoderCoarseRestorer(nn.Module):
    def __init__(self, res=128, in_channels=3, width=64, fuse_type="concat"):
        super().__init__()

        self.fuse_type = fuse_type
        SCALER = 2 if fuse_type == "concat" else 1

        self.lq_intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.mq_intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.outro = nn.Conv2d(
            in_channels=width * SCALER,
            out_channels=3,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.lq_encoders = nn.Sequential(
            NAF_STN_Block(width, res, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 2, res // 2, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 4, res // 4, num_naf=4, sampling="down"),
            NAF_STN_Block(width * 8, res // 8, num_naf=8, sampling="down"),
        )
        self.mq_encoders = nn.Sequential(
            NAF_STN_Block(width, res, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 2, res // 2, num_naf=2, sampling="down"),
            NAF_STN_Block(width * 4, res // 4, num_naf=4, sampling="down"),
            NAF_STN_Block(width * 8, res // 8, num_naf=8, sampling="down"),
        )

        if self.fuse_type == "addition":
            # ControlNet-like zero-convolution layers
            self.zero_convs = nn.ModuleList(
                [
                    self._make_zero_conv(width * 2),
                    self._make_zero_conv(width * 4),
                    self._make_zero_conv(width * 8),
                    self._make_zero_conv(width * 16),
                ]
            )
            self.zero_conv_mid = self._make_zero_conv(width * 16)

        self.middle_blocks = NAF_STN_Block(width * 16 * SCALER, res // 16, num_naf=8)

        self.decoders = nn.Sequential(
            NAF_STN_Block(width * 16 * SCALER, res // 16, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 8 * SCALER, res // 8, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 4 * SCALER, res // 4, num_naf=2, sampling="up"),
            NAF_STN_Block(width * 2 * SCALER, res // 2, num_naf=2, sampling="up"),
        )
        self.out_tanh = nn.Tanh()

    @staticmethod
    def _make_zero_conv(channels: int) -> nn.Conv2d:
        # 1x1 conv, weight와 bias 모두 0으로 초기화
        conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, lq_face: torch.Tensor, mq_face: torch.Tensor):
        skip_feats = []

        lq_feat = self.lq_intro(lq_face)
        mq_feat = self.mq_intro(mq_face)

        for idx, (lq_enc, mq_enc) in enumerate(zip(self.lq_encoders, self.mq_encoders)):
            lq_feat = lq_enc(lq_feat)
            mq_feat = mq_enc(mq_feat)

            if self.fuse_type == "concat":
                skip = torch.concat([lq_feat, mq_feat], dim=1)
            elif self.fuse_type == "addition":
                skip = lq_feat + self.zero_convs[idx](mq_feat)

            skip_feats.append(skip)

        # fuse the two encoders outputs
        if self.fuse_type == "concat":
            feat = torch.cat([lq_feat, mq_feat], dim=1)
        elif self.fuse_type == "addition":
            feat = lq_feat + self.zero_conv_mid(mq_feat)

        feat = self.middle_blocks(feat)

        for decoder, skip_feat in zip(self.decoders, skip_feats[::-1]):
            feat = feat + skip_feat
            feat = decoder(feat)

        feat = self.outro(feat)
        feat = (self.out_tanh(feat) + 1.0) / 2.0  # normalize to [0, 1]

        return feat
