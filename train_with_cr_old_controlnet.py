import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torchvision.utils import make_grid, save_image
from diffusers import ControlNetModel, StableDiffusionPipeline
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from dataset_multipie import MultiPIEDataset
from models.arcface.models import resnet_face18
from discriminator import SDXLPartialDiscriminator
from utils.others import get_x0_from_noise, process_arcface_input, process_visual_image
from edge_aware_dists_demo import EdgeAwareDISTSLoss
from models.cr.model import CoarseRestorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional pretrained ControlNet path. If omitted, a ControlNet is "
            "initialized from the SD U-Net."
        ),
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="a high-quality frontal face portrait",
        help="Fixed text prompt used for all training samples.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.75,
        help="Scale for ControlNet residuals injected into the U-Net.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--lambda_adv", type=float, default=1e-2, help="Adversarial loss weight"
    )
    parser.add_argument(
        "--lambda_id", type=float, default=1e-1, help="Identity loss weight"
    )
    parser.add_argument("--ckpt_path", type=str, default="pretrained")
    parser.add_argument(
        "--unet_lora_lr",
        type=float,
        default=1e-5,
        help="Learning rate for trainable U-Net LoRA weights.",
    )
    parser.add_argument(
        "--controlnet_lr",
        type=float,
        default=1e-4,
        help="Learning rate for ControlNet.",
    )
    parser.add_argument(
        "--d_lr",
        type=float,
        default=1e-4,
        help="Learning rate for SDXL discriminator.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/12_controlnet",
        help="Root directory for saving results",
    )
    parser.add_argument(
        "--save_image_steps",
        type=int,
        default=200,
        help="Interval to save image samples",
    )
    parser.add_argument(
        "--save_checkpoint_epochs",
        type=int,
        default=1,
        help="Interval to save model checkpoints",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for U-Net and ControlNet.",
    )
    return parser.parse_args()


def get_trainable_lora_state_dict(unet_model):
    state_dict = {}
    for name, param in unet_model.named_parameters():
        if "lora" in name:
            state_dict[name.replace(".default_0", "")] = param.detach().cpu()
    return state_dict


def encode_fixed_prompt(tokenizer, text_encoder, prompt_text, batch_size, device):
    text_inputs = tokenizer(
        [prompt_text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

    return prompt_embeds.expand(batch_size, -1, -1).contiguous()


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    if args.seed is not None:
        from accelerate.utils import set_seed

        set_seed(args.seed)

    # Core SD 2.1 components (now including tokenizer/text encoder for a fixed text prompt).
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # ControlNet for MQ frontal-face conditioning.
    if args.controlnet_model_name_or_path is not None:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        controlnet = ControlNetModel.from_unet(unet)
    controlnet.train()

    if args.use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Load existing U-Net LoRA weights and keep only LoRA trainable.
    pipe.load_lora_weights(args.ckpt_path, weight_name="unet_lora.safetensors")
    pipe.unet.train()
    pipe.unet.requires_grad_(False)

    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    # Coarse restorer modules for MQ frontal prior.
    cr_modules = []
    for i in range(5):
        cr_module = CoarseRestorer(width=32).to(device=device)
        cr_module.load_state_dict(
            load_file(
                "pretrained/cr_split/%s:%s/model.safetensors" % (i * 40, i * 40 + 40)
            ),
            strict=False,
        )
        cr_module.requires_grad_(False)
        cr_module.eval()
        cr_modules.append(cr_module)

    # Identity Model
    id_model = resnet_face18(use_se=False).to(device=device)
    id_model_ckpt = torch.load(
        "models/arcface/weights/resnet18_110_wo_dist.pth", weights_only=False
    )
    id_model.load_state_dict(id_model_ckpt)
    id_model.requires_grad_(False)
    id_model.eval()

    # Discriminator from SDXL
    discriminator = SDXLPartialDiscriminator(
        sdxl_unet_id="stabilityai/stable-diffusion-xl-base-1.0", device=device
    )
    d_lora_config = LoraConfig(
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
    discriminator = get_peft_model(discriminator, d_lora_config)
    discriminator.train()

    for name, param in discriminator.named_parameters():
        if "lora_" not in name and "mlp_head" not in name:
            param.requires_grad = False

    # Optimizers & Losses
    unet_lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]

    optimizer_g = torch.optim.AdamW(
        [
            {"params": unet_lora_params, "lr": args.unet_lora_lr},
            {"params": controlnet_params, "lr": args.controlnet_lr},
        ]
    )
    optimizer_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.d_lr
    )

    criterion_mse = nn.MSELoss()
    criterion_perceptual = EdgeAwareDISTSLoss(device=device)
    criterion_id = nn.CosineEmbeddingLoss()

    # DataLoader
    dataset = MultiPIEDataset(
        "/vcl4/Jiseung/datasets/multipie_crop_patch_v2",
        phase="train",
        size=512,
        use_blind=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Accelerator Prepare
    (
        pipe.unet,
        controlnet,
        discriminator,
        optimizer_g,
        optimizer_d,
        dataloader,
    ) = accelerator.prepare(
        pipe.unet,
        controlnet,
        discriminator,
        optimizer_g,
        optimizer_d,
        dataloader,
    )

    vae.to(device)
    text_encoder.to(device)

    # Mixed Precision helper dtype
    weight_dtype = torch.float32

    # Fixed prompt embedding cache for a full batch
    cached_prompt_embeds = encode_fixed_prompt(
        tokenizer, text_encoder, args.prompt_text, args.batch_size, device
    ).to(dtype=weight_dtype)

    # Training loop
    for epoch in range(args.max_epoch):
        for idx, (lq, gt, filename) in enumerate(dataloader):
            bs = lq.shape[0]
            lq_resized = interpolate(lq, size=(128, 128), mode="bicubic")

            with torch.no_grad():
                cr_out_list = []
                for b in range(lq_resized.shape[0]):
                    pid = int(filename[b][:3])
                    cr_idx = min((pid - 1) // 40, len(cr_modules) - 1)
                    cr_out_b = cr_modules[cr_idx](lq_resized[b].unsqueeze(0))
                    cr_out_list.append(cr_out_b)

                cr_out = torch.cat(cr_out_list, dim=0)
                mq_f = interpolate(cr_out, size=(512, 512), mode="bicubic")

            # Use the degraded non-frontal face as the main U-Net input source.
            lq_512 = interpolate(lq, size=(512, 512), mode="bicubic")

            # Normalize to [-1, 1]
            lq_512 = (lq_512 - 0.5) * 2.0
            mq_f = (mq_f - 0.5) * 2.0
            gt = (gt - 0.5) * 2.0

            with torch.no_grad():
                lq_latent = (
                    vae.encode(lq_512.to(dtype=weight_dtype)).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                gt_latent = (
                    vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            if bs == cached_prompt_embeds.shape[0]:
                prompt_embeds = cached_prompt_embeds
            else:
                prompt_embeds = encode_fixed_prompt(
                    tokenizer, text_encoder, args.prompt_text, bs, device
                ).to(dtype=weight_dtype)

            """
            🍞 Generator Update - - - - - - - - - - - - - - - - - - - -
            """
            optimizer_g.zero_grad()

            # One-step diffusion timestep
            timesteps_g = torch.full((bs,), 399, device=device, dtype=torch.long)

            # MQ frontal face supplies structure through ControlNet.
            down_block_res_samples, mid_block_res_sample = controlnet(
                lq_latent,
                timesteps_g,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=mq_f.to(dtype=weight_dtype),
                conditioning_scale=args.controlnet_conditioning_scale,
                return_dict=False,
            )

            # LQ non-frontal latent is directly fed to the main U-Net path.
            model_pred = pipe.unet(
                lq_latent,
                timesteps_g,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            x_0_latent = get_x0_from_noise(
                lq_latent,
                model_pred,
                noise_scheduler.alphas_cumprod.to(device),
                timesteps_g,
            )

            restored_img = vae.decode(x_0_latent / vae.config.scaling_factor).sample

            with torch.no_grad():
                # mq_feature = id_model(process_arcface_input(mq_f))
                gt_feature = id_model(process_arcface_input(gt))

            restored_feature = id_model(process_arcface_input(restored_img))
            ID_TARGET = torch.ones((bs,), device=device)

            loss_cons = (
                criterion_mse(restored_img, gt)
                + criterion_perceptual(restored_img, gt)
                + criterion_id(restored_feature, gt_feature, ID_TARGET) * args.lambda_id
            )

            # Adversarial loss for generator
            D_t = torch.randint(0, 1000, (bs,), device=device, dtype=torch.long)
            noise_fake = torch.randn_like(x_0_latent)
            z_hat_t = noise_scheduler.add_noise(x_0_latent, noise_fake, D_t)

            prompt_embeds_sdxl = torch.zeros(
                bs, 77, 2048, device=device, dtype=weight_dtype
            )
            added_cond_kwargs = {
                "text_embeds": torch.zeros(bs, 1280, device=device, dtype=weight_dtype),
                "time_ids": torch.tensor(
                    [[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]],
                    device=device,
                    dtype=weight_dtype,
                ).repeat(bs, 1),
            }

            logits_fake_for_g = discriminator(
                z_hat_t,
                D_t,
                encoder_hidden_states=prompt_embeds_sdxl,
                added_cond_kwargs=added_cond_kwargs,
            )

            loss_G_adv = F.binary_cross_entropy_with_logits(
                logits_fake_for_g, torch.ones_like(logits_fake_for_g)
            )

            total_loss_G = loss_cons + (args.lambda_adv * loss_G_adv)
            accelerator.backward(total_loss_G)
            optimizer_g.step()

            """
            🔪 Discriminator Update - - - - - - - - - - - - - - - - - - - -
            """
            optimizer_d.zero_grad()

            noise_real = torch.randn_like(gt_latent)
            z_real_t = noise_scheduler.add_noise(gt_latent, noise_real, D_t)

            logits_fake = discriminator(
                z_hat_t.detach(),
                D_t,
                encoder_hidden_states=prompt_embeds_sdxl,
                added_cond_kwargs=added_cond_kwargs,
            )
            logits_real = discriminator(
                z_real_t,
                D_t,
                encoder_hidden_states=prompt_embeds_sdxl,
                added_cond_kwargs=added_cond_kwargs,
            )

            loss_d_fake = F.binary_cross_entropy_with_logits(
                logits_fake, torch.zeros_like(logits_fake)
            )
            loss_d_real = F.binary_cross_entropy_with_logits(
                logits_real, torch.ones_like(logits_real)
            )

            loss_D = (loss_d_fake + loss_d_real) * 0.5
            accelerator.backward(loss_D)
            optimizer_d.step()

            if idx % 100 == 0 and accelerator.is_main_process:
                print(
                    f"Step {idx}: L_cons = {loss_cons.item():.4f}, "
                    f"L_G_adv = {loss_G_adv.item():.4f}, Loss_D = {loss_D.item():.4f}"
                )

            if idx % args.save_image_steps == 0 and accelerator.is_main_process:
                save_dir = os.path.join(args.output_dir, "samples")
                os.makedirs(save_dir, exist_ok=True)

                with torch.no_grad():
                    vis_lq = process_visual_image(lq_512)
                    vis_mq_f = process_visual_image(mq_f)
                    vis_restored = process_visual_image(restored_img)
                    vis_gt = process_visual_image(gt)

                    pred_noise = vae.decode(
                        model_pred / vae.config.scaling_factor
                    ).sample
                    vis_pred_noise = process_visual_image(pred_noise)

                    n_save = min(4, bs)
                    grid = torch.cat(
                        [
                            vis_lq[:n_save],
                            vis_mq_f[:n_save],
                            vis_pred_noise[:n_save],
                            vis_restored[:n_save],
                            vis_gt[:n_save],
                        ],
                        dim=-1,
                    )

                    save_path = os.path.join(
                        save_dir, f"epoch_{epoch:02d}__step_{idx:06d}.png"
                    )
                    save_image(make_grid(grid, nrow=1, padding=2), save_path)
                    print(f"📸 Saved sample images to {save_path}")

        if epoch % args.save_checkpoint_epochs == 0 and accelerator.is_main_process:
            ckpt_dir = os.path.join(
                args.output_dir, "checkpoints", f"epoch_{epoch:02d}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)

            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            unet_lora_state_dict = get_trainable_lora_state_dict(unwrapped_unet)
            StableDiffusionPipeline.save_lora_weights(
                save_directory=ckpt_dir,
                unet_lora_layers=unet_lora_state_dict,
                weight_name="unet_lora.safetensors",
            )

            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
            unwrapped_controlnet.save_pretrained(os.path.join(ckpt_dir, "controlnet"))

            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            unwrapped_discriminator.save_pretrained(
                os.path.join(ckpt_dir, "discriminator_lora")
            )

            print(f"💾 Saved checkpoints to {ckpt_dir}")

    print("✅ Done!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
