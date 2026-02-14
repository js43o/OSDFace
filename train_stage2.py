import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from dataset_multipie import MultiPIEDataset
from models.lq_embed import vqvae_encoder, TwoLayerConv1x1
from discriminator import SDXLPartialDiscriminator
from utils.others import get_x0_from_noise, process_arcface_input, process_visual_image
from edge_aware_dists_demo import EdgeAwareDISTSLoss
from models.arcface.models import resnet_face18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
    )
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--max_epoch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--lambda_adv", type=float, default=1e-2, help="Adversarial loss weight"
    )
    parser.add_argument(
        "--lambda_id", type=float, default=1e-1, help="Identity loss weight"
    )
    parser.add_argument("--ckpt_path", type=str, default="pretrained")
    parser.add_argument(
        "--img_encoder_weight",
        type=str,
        default="pretrained/ddvqgan_encoder.ckpt",  # default="pretrained/associate_2.ckpt"
    )
    parser.add_argument(
        "--cat_prompt_embedding",
        action="store_true",
        help="use cat_prompt_embedding to exchange embedding change",
    )
    parser.add_argument(
        "--use_pos_embedding", action="store_true", help="use 2D pos embedding"
    )
    parser.add_argument(
        "--use_att_pool", action="store_true", help="use attention pool layer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/06",
        help="Root directory for saving results",
    )
    parser.add_argument(
        "--save_image_steps",
        type=int,
        default=2000,
        help="Interval to save image samples",
    )
    parser.add_argument(
        "--save_checkpoint_epochs",
        type=int,
        default=1,
        help="Interval to save model checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    if args.seed is not None:
        from accelerate.utils import set_seed

        set_seed(args.seed)

    # Scheduler & VAE
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.eval()
    vae.requires_grad_(False)

    # UNet from Stable Diffusion
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # Custom Modules
    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
            weights_only=False,
        )
    )

    # LQ Image Encoder
    img_encoder = vqvae_encoder(args)
    img_encoder.to(device=device)

    # SD Pipeline with LoRA
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe.load_lora_weights(args.ckpt_path)
    pipe.unet.train()
    pipe.unet.requires_grad_(False)

    unet_lora_state_dict = {}

    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            name_new = name.replace(".default_0", "")
            param.requires_grad = True
            unet_lora_state_dict[name_new] = param

    # Discriminator from SDXL
    discriminator = SDXLPartialDiscriminator(
        sdxl_unet_id="stabilityai/stable-diffusion-xl-base-1.0", device=device
    )

    # Identity Model
    id_model = resnet_face18(use_se=False)
    id_model.load_state_dict(
        torch.load(
            "models/arcface/weights/resnet18_110_wo_dist.pth", weights_only=False
        )
    )
    id_model.to(device=device)
    id_model.eval()

    lora_config_D = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "proj_in",
            "proj_out",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    discriminator = get_peft_model(discriminator, lora_config_D)
    discriminator.requires_grad_(False)
    discriminator.mlp_head.requires_grad_(True)

    # Optimizer & Loss
    optimizer_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=1e-4
    )
    optimizer_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=1e-4
    )

    criterion_mse = nn.MSELoss()
    criterion_perceptual = EdgeAwareDISTSLoss(device=device)
    criterion_id = nn.CosineEmbeddingLoss()

    # DataLoader
    dataset = MultiPIEDataset(
        "/4tb/datasets/multipie_crop_patch_v2",
        model_type="uni",
        use="train",
        res=512,
        use_blind=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Accelerator Prepare
    (
        pipe.unet,
        discriminator,
        embedding_change,
        optimizer_g,
        optimizer_d,
        dataloader,
    ) = accelerator.prepare(
        pipe.unet,
        discriminator,
        embedding_change,
        optimizer_g,
        optimizer_d,
        dataloader,
    )
    vae.to(device)

    # Mixed Precision
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # ------------------ 학습 루프 ------------------
    for epoch in range(args.max_epoch):
        for idx, (lq, gt) in enumerate(dataloader):
            # [-1, 1] 범위로 정규화
            lq = (lq - 0.5) * 2.0
            gt = (gt - 0.5) * 2.0

            # VAE Encoding
            with torch.no_grad():
                lq_latent = (
                    vae.encode(lq.to(dtype=weight_dtype)).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                gt_latent = (
                    vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            # Prompt Embedding
            prompt_embeds = []
            for lq_batch in lq:
                prompt_embed = img_encoder(lq_batch.unsqueeze(0)).reshape(1, 77, -1)
                prompt_embed = embedding_change(prompt_embed)
                prompt_embeds.append(prompt_embed)

            prompt_embeds = torch.cat(prompt_embeds)

            # ---------------- Generator Update ----------------
            optimizer_g.zero_grad()

            # Timesteps Sampling
            """
            timesteps_g = torch.full(
                (args.batch_size,), 399, device=device, dtype=torch.long
            )
            """
            timesteps_g = torch.randint(
                0, 1000, (args.batch_size,), device=device, dtype=torch.long
            )

            # UNet Inference
            model_pred = pipe.unet(
                lq_latent, timesteps_g, encoder_hidden_states=prompt_embeds
            ).sample

            # x0 예측 (Reconstruction)
            x_0_latent = get_x0_from_noise(
                lq_latent,
                model_pred,
                noise_scheduler.alphas_cumprod.to(device),
                timesteps_g,
            )

            # VAE Decoding
            restored_img = vae.decode(x_0_latent / vae.config.scaling_factor).sample

            # Identity Features
            gt_feature = id_model(process_arcface_input(gt))[0].unsqueeze(0)
            restored_feature = id_model(process_arcface_input(restored_img))[
                0
            ].unsqueeze(0)
            ID_TARGET = torch.ones((args.batch_size,), device=device)

            # Consistency Loss
            loss_pixel = (
                criterion_mse(restored_img, gt)
                + criterion_perceptual(restored_img, gt)
                + criterion_id(restored_feature, gt_feature, ID_TARGET) * args.lambda_id
            )

            # Noise Sampling for Discriminator
            D_t = torch.randint(
                0, 1000, (args.batch_size,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(x_0_latent)
            z_hat_t = noise_scheduler.add_noise(x_0_latent, noise, D_t)

            # SDXL용 더미 데이터
            added_cond_kwargs = {
                "text_embeds": torch.zeros(
                    args.batch_size, 1280, device=device, dtype=weight_dtype
                ),
                "time_ids": torch.tensor(
                    [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]],
                    device=device,
                    dtype=weight_dtype,
                ).repeat(args.batch_size, 1),
            }
            prompt_embeds_sdxl = torch.zeros(
                args.batch_size, 77, 2048, device=device, dtype=weight_dtype
            )

            logits_fake_for_g = discriminator(
                z_hat_t,
                D_t,
                encoder_hidden_states=prompt_embeds_sdxl,
                added_cond_kwargs=added_cond_kwargs,
            )

            loss_G_adv = F.binary_cross_entropy_with_logits(
                logits_fake_for_g, torch.ones_like(logits_fake_for_g)
            )

            total_loss_G = loss_pixel + (args.lambda_adv * loss_G_adv)
            accelerator.backward(total_loss_G)
            optimizer_g.step()

            # ---------------- Discriminator Update ----------------
            optimizer_d.zero_grad()

            # Real Image Noise Injection
            z_real_t = noise_scheduler.add_noise(gt_latent, noise, D_t)

            # Concatenate (Fake Detached + Real)
            # z_hat_t.detach() 필수!
            disc_input = torch.cat([z_hat_t.detach(), z_real_t], dim=0)
            disc_timesteps = torch.cat([D_t, D_t], dim=0)

            # Conditioning도 배치 사이즈 2배로 확장
            combined_cond_kwargs = {
                k: torch.cat([v, v], dim=0) for k, v in added_cond_kwargs.items()
            }
            combined_prompt_embeds = torch.cat(
                [prompt_embeds_sdxl, prompt_embeds_sdxl], dim=0
            )

            logits = discriminator(
                disc_input,
                disc_timesteps,
                encoder_hidden_states=combined_prompt_embeds,
                added_cond_kwargs=combined_cond_kwargs,
            )
            logits_fake, logits_real = torch.chunk(logits, 2, dim=0)

            loss_d_fake = F.binary_cross_entropy_with_logits(
                logits_fake, torch.zeros_like(logits_fake)
            )
            loss_d_real = F.binary_cross_entropy_with_logits(
                logits_real, torch.ones_like(logits_real)
            )

            loss_D = (loss_d_fake + loss_d_real) * 0.5
            accelerator.backward(loss_D)
            optimizer_d.step()

            # Logs Printing
            if idx % 100 == 0:
                if accelerator.is_main_process:
                    print(
                        f"Step {idx}: L_pix = {loss_pixel.item():.4f}, L_G = {loss_G_adv.item():.4f}, Loss D = {loss_D.item():.4f}"
                    )

            # Validation Images Saving
            if idx % args.save_image_steps == 0:
                if accelerator.is_main_process:
                    save_dir = os.path.join(args.output_dir, "samples")
                    os.makedirs(save_dir, exist_ok=True)

                    with torch.no_grad():
                        vis_lq = process_visual_image(lq)
                        vis_gt = process_visual_image(gt)
                        vis_restored = process_visual_image(restored_img)

                        n_save = min(4, args.batch_size)
                        grid = torch.cat(
                            [vis_lq[:n_save], vis_restored[:n_save], vis_gt[:n_save]],
                            dim=-1,
                        )

                        save_path = os.path.join(
                            save_dir, f"epoch_{epoch:02d}__step_{idx:06d}.png"
                        )
                        save_image(make_grid(grid, nrow=1, padding=2), save_path)
                        print(f"📸 Saved sample images to {save_path}")

        # Model Checkpoints Saving
        if epoch % args.save_checkpoint_epochs == 0:
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(
                    args.output_dir, "checkpoints", f"epoch_{epoch:02d}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                # Generator (UNet LoRA)
                pipe.save_lora_weights(
                    save_directory=ckpt_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    weight_name="unet_lora.safetensors",
                )

                # Embedding Change Module
                unwrapped_emb_change = accelerator.unwrap_model(embedding_change)
                torch.save(
                    unwrapped_emb_change.state_dict(),
                    os.path.join(ckpt_dir, "embedding_change.pth"),
                )

                # Discriminator (PEFT 모델이므로 save_pretrained 지원)
                unwrapped_discriminator = accelerator.unwrap_model(discriminator)
                # PEFT 모델은 save_pretrained로 LoRA config와 weight를 같이 저장
                unwrapped_discriminator.save_pretrained(
                    os.path.join(ckpt_dir, "discriminator_lora")
                )

                print(f"💾 Saved checkpoints to {ckpt_dir}")

    print("✅ Done!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
