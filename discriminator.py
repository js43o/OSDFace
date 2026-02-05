import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding


class SDXLPartialDiscriminator(nn.Module):
    def __init__(self, sdxl_unet_id, device):
        super().__init__()

        full_unet = UNet2DConditionModel.from_pretrained(
            sdxl_unet_id,
            subfolder="unet",
            addition_embed_type="text_time",
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        ).to(device)

        self.conv_in = full_unet.conv_in
        self.time_embedding = full_unet.time_embedding
        self.add_time_proj = full_unet.add_time_proj  # time_ids 투영용
        self.add_embedding = full_unet.add_embedding  # 결합된 임베딩 투영용

        self.down_blocks = full_unet.down_blocks
        self.mid_block = full_unet.mid_block

        # 모델 설정 저장 (forward에서 차원 참조용)
        self.config = full_unet.config

        # 메모리 정리
        del full_unet
        torch.cuda.empty_cache()

        # 가중치 고정
        self.requires_grad_(False)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

        self.to(device)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs):
        # Timestep Embedding (int -> sinusoidal -> vector)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Broadcast check
        if len(timesteps.shape) == 1 and timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.repeat(sample.shape[0])

        t_embeds = get_timestep_embedding(
            timesteps,
            embedding_dim=self.config.block_out_channels[0],
            downscale_freq_shift=0,
        ).to(dtype=sample.dtype)

        embeds = self.time_embedding(t_embeds)

        # Augmented Embedding (SDXL = Text + Time)
        if self.add_embedding is not None:
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")

            # time_ids 투영
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            # Text + Time 결합
            text_time_embeds = torch.concat([text_embeds, time_embeds], dim=-1).to(
                embeds.dtype
            )

            # 최종 투영 및 합산
            aug_embeds = self.add_embedding(text_time_embeds)
            embeds = embeds + aug_embeds

        sample = self.conv_in(sample)

        # Downsampling Pass
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, _ = downsample_block(
                    hidden_states=sample,
                    temb=embeds,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, _ = downsample_block(hidden_states=sample, temb=embeds)

        # Middle Block Pass
        if (
            hasattr(self.mid_block, "has_cross_attention")
            and self.mid_block.has_cross_attention
        ):
            sample = self.mid_block(
                sample,
                embeds,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, embeds)

        # MLP Head
        logit = self.mlp_head(sample)

        return logit
