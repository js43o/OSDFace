import copy
import os
import glob
import argparse
import copy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as Fun
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from models.lq_embed import vqvae_encoder, TwoLayerConv1x1

from dataset_multipie import ANGLES_EXTREME, ANGLES_MODERATE
from models.cr.model import CoarseRestorer, DualEncoderCoarseRestorer


class OSDFace_test(nn.Module):
    def __init__(self, args, gpu_id, Unet):
        super().__init__()

        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae"
        )
        if args.merge_lora:
            self.unet = copy.deepcopy(Unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="unet"
            )

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        self.load_ckpt(args.ckpt_path)

        self.img_encoder = vqvae_encoder(args)

        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)

        self.timesteps = 399

        self.use_uni = args.use_uni
        self.use_noise = args.use_noise

        # Define CR modules
        if args.use_uni:
            self.uni_model = CoarseRestorer(res=128).to(self.device)
            self.uni_model.load_state_dict(load_file(args.ckpt_uni))
            self.uni_model.eval()
        else:
            self.m2f_model = CoarseRestorer().to(self.device)
            self.efr_model = CoarseRestorer().to(self.device)
            self.e2f_model = DualEncoderCoarseRestorer().to(self.device)

            self.m2f_model.load_state_dict(load_file(args.ckpt_m2f))
            self.efr_model.load_state_dict(load_file(args.ckpt_efr))
            self.e2f_model.load_state_dict(load_file(args.ckpt_e2f))

            self.m2f_model.eval()
            self.efr_model.eval()
            self.e2f_model.eval()

    def load_ckpt(self, ckpt_path):
        if not self.args.cat_prompt_embedding:
            self.embedding_change = TwoLayerConv1x1(512, 1024)
            self.embedding_change.load_state_dict(
                torch.load(
                    os.path.join(ckpt_path, "embedding_change.pth"),
                    weights_only=False,
                )
            )
            self.embedding_change.to(self.device, dtype=self.weight_dtype)
        if not self.args.merge_lora:
            pipe = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=None,
                tokenizer=None,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
            """
            pipe.unet = PeftModel.from_pretrained(
                pipe.unet, os.path.join(ckpt_path, "generator_lora")
            )
            pipe.load_lora_weights(
                os.path.join(ckpt_path, "generator_lora"),
                adapter_name="adapter_model",
            )
            """
            g_lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
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
            pipe.unet = get_peft_model(pipe.unet, g_lora_config)

            lora_state_dict = load_file(
                os.path.join(ckpt_path, "generator_lora", "adapter_model.safetensors")
            )
            set_peft_model_state_dict(pipe.unet, lora_state_dict)

            pipe.unet.to(self.device, dtype=self.weight_dtype)

            self.unet = pipe.unet

    @perfcount
    @torch.no_grad()
    def forward(self, lq, filename):
        # lq.shape = (bs, 3, 128, 128), range = [0, 1]
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            lq_512 = Fun.interpolate(lq, size=(512, 512), mode="bicubic") * 2.0 - 1.0
            prompt_embeds = self.img_encoder(lq_512).reshape(lq.shape[0], 77, -1)
            if not self.args.cat_prompt_embedding:
                prompt_embeds = self.embedding_change(prompt_embeds)

            prompt_embeds = Fun.normalize(prompt_embeds, dim=-1)  # 정규화

        with torch.cuda.stream(stream2):
            if self.use_uni:
                hq_f_pred = self.uni_model(lq)
            else:
                if filename[4:8] in ANGLES_EXTREME:
                    hq_e_pred = self.efr_model(lq)
                    hq_f_pred = self.e2f_model(hq_e_pred, lq)  # reversed order
                elif filename[4:8] in ANGLES_MODERATE:
                    hq_f_pred = self.m2f_model(lq)
                else:
                    hq_f_pred = self.m2f_model(lq)  # use M2F model
                    # raise "Exception: unrecognized pose in filename: %s" % filename[4:8]

            # 512*512 크기로 리사이징 및 [-1, 1] 범위로 정규화
            hq_f_pred = Fun.interpolate(hq_f_pred, size=(512, 512), mode="bicubic")
            hq_f_pred = (hq_f_pred - 0.5) * 2.0

            lq_latent = (
                self.vae.encode(hq_f_pred.to(self.weight_dtype)).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

            if self.use_noise:
                T_L = 400
                self.timesteps = torch.randint(
                    0,
                    T_L,
                    (lq_latent.shape[0],),
                    device=lq_latent.device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(lq_latent, device=lq_latent.device)
                lq_latent = self.noise_scheduler.add_noise(
                    lq_latent, noise, self.timesteps
                )

        torch.cuda.synchronize()

        model_pred = self.unet(
            lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds
        ).sample

        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps,
        ).float()

        output_image = (
            self.vae.decode(
                x_0.to(self.weight_dtype) / self.vae.config.scaling_factor
            ).sample
        ).clamp(-1, 1)
        output_image = output_image * 0.5 + 0.5

        return output_image.clamp(0.0, 1.0)


def main_worker(Unet, rank, gpu_id, image_names, weight_dtype, args):
    torch.cuda.set_device(gpu_id)

    model = OSDFace_test(args, gpu_id, Unet).to(gpu_id)

    if args.save_comp:
        os.makedirs(os.path.join(args.output_dir, "comp"), exist_ok=True)

    for image_name in tqdm(image_names):
        output_file_path = os.path.join(args.output_dir, os.path.basename(image_name))
        input_image = (
            Image.open(image_name)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        with torch.no_grad():
            lq = F.to_tensor(input_image).unsqueeze(0).to(gpu_id, dtype=weight_dtype)

            output_image = model(lq, os.path.basename(image_name))

            output_pil = transforms.ToPILImage()(output_image[0].cpu())
            output_pil = output_pil.resize(
                (args.output_size, args.output_size), resample=Image.Resampling.BICUBIC
            )
            output_pil.save(output_file_path)

            if args.save_comp:
                comp_pil = Image.new(
                    "RGB", (input_image.width + output_pil.width, input_image.height)
                )
                comp_pil.paste(input_image, (0, 0))
                comp_pil.paste(output_pil, (input_image.width, 0))
                comp_pil.save(
                    os.path.join(args.output_dir, "comp", os.path.basename(image_name))
                )


def run_inference(args, Unet):
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f"{args.input_image}/*.[jpJP][pnPN]*[gG]"))
    else:
        image_names = [args.input_image]

    # random.shuffle(image_names)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    num_gpus = len(args.gpu_ids)
    images_per_gpu = len(image_names) // num_gpus
    processes = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        start_idx = rank * images_per_gpu
        end_idx = (
            start_idx + images_per_gpu if rank != num_gpus - 1 else len(image_names)
        )
        image_subset = image_names[start_idx:end_idx]

        p = mp.Process(
            target=main_worker,
            args=(Unet, rank, gpu_id, image_subset, weight_dtype, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image",
        "-i",
        type=str,
        default="../../datasets/multipie_validation_128/lq",
        help="path to the input image",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="the directory to save the output",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
        help="sd model path",
    )
    parser.add_argument("--seed", type=int, default=114, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=128)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--use_uni", action="store_true", help="use single CR module")
    parser.add_argument(
        "--save_comp", action="store_true", help="save comparision image"
    )
    parser.add_argument(
        "--ckpt_uni",
        type=str,
        default="./pretrained/cr/0A1_uni_pix=1.0_vgg=0.001_adv=0.0025_id=0.0025/29/model.safetensors",
    )
    parser.add_argument(
        "--ckpt_m2f",
        type=str,
        default="./pretrained/cr/0A7_m2f_pix=1.0_vgg=0.001_adv=0.001_id=0.0025/29/model.safetensors",
    )
    parser.add_argument(
        "--ckpt_efr",
        type=str,
        default="./pretrained/cr/0A2_e2e_pix=1.0_vgg=0.001_id=0.01/29/model.safetensors",
    )
    parser.add_argument(
        "--ckpt_e2f",
        type=str,
        default="./pretrained/cr/0A41_e2f_pix=1.0_vgg=0.001_adv=0.0025_id=0.0025/29/model.safetensors",
    )
    parser.add_argument(
        "--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp32"
    )
    parser.add_argument(
        "--img_encoder_weight", type=str, default="pretrained/associate_2.ckpt"
    )
    parser.add_argument(
        "--gpu_ids", nargs="+", type=int, default=[0], help="List of GPU IDs to use"
    )
    parser.add_argument(
        "--cat_prompt_embedding",
        action="store_true",
        help="use cat_prompt_embedding to exchange embedding change",
    )
    parser.add_argument(
        "--use_att_pool", action="store_true", help="use attention pool layer"
    )
    parser.add_argument(
        "--use_pos_embedding", action="store_true", help="use 2D pos embedding"
    )
    parser.add_argument("--merge_lora", action="store_true", help="merge LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64)
    parser.add_argument(
        "--use_noise",
        action="store_true",
        help="whether to add random noise to input LQ image",
    )

    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    Unet = None
    if args.merge_lora:
        # Unet = merge_Unet(args)
        pass
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(glob.glob(f"{args.input_image}/*"))} images to process.')

    run_inference(args, Unet)
