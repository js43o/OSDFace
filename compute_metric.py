import os
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from pyiqa import create_metric
import argparse
import shutil
import random
import face_alignment
import numpy as np
from tqdm import tqdm

from dataset_multipie import ANGLES_EXTREME, ANGLES_MODERATE
from helpers.arcface.models import resnet_face18
from helpers.utils import process_arcface_input

parser = argparse.ArgumentParser()
parser.add_argument(
    "--comp_path",
    type=str,
    default="outputs/00_osdface/epoch_02",
)
parser.add_argument(
    "--gt_path",
    type=str,
    default="../../datasets/multipie_validation_128_v2/gt",
)
parser.add_argument(
    "--pose_group",
    type=str,
    required=True,
    help='Pose group of the evaluation dataset ("all" | "e" | "m" | "custom")',
)
parser.add_argument(
    "--match", action="store_true", help="Match the number of comp and GT samples"
)
args = parser.parse_args()

print("✅", args.comp_path)

gt_filenames = sorted(os.listdir(os.path.join(args.gt_path)))
comp_filenames = sorted(os.listdir(os.path.join(args.comp_path)))

device = "cuda"
TEMP_DIR = "temp_fid_%s" % int(random.random() * 10000000)

os.makedirs(os.path.join(TEMP_DIR, "gt"), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, "comp"), exist_ok=True)

"""
TARGET_POSES = [
    "11_0",
    "12_0",
    "09_0",
    "19_1",
    "08_1",
    "20_0",
    "01_0",
    "24_0",
    "08_0",
    "13_0",
    "14_0",
    "05_0",
    "04_1",
    "19_0",
]
"""

# TARGET_POSES = ["19_1", "08_1"]
# TARGET_POSES = ["11_0", "24_0"]
# TARGET_POSES = ["12_0", "01_0"]
# TARGET_POSES = ["09_0", "20_0"]

# TARGET_POSES = ["08_0", "19_0"]
# TARGET_POSES = ["13_0", "04_1"]
TARGET_POSES = ["14_0", "05_0"]

if args.pose_group == "all":
    print("😄 all pose groups are selected (%s samples)" % len(gt_filenames))
if args.pose_group == "e":
    gt_filenames = [f for f in gt_filenames if f[4:8] in ANGLES_EXTREME]
    comp_filenames = [f for f in comp_filenames if f[4:8] in ANGLES_EXTREME]
    print("😮 Only compute pose group E (%s samples)" % len(gt_filenames))
elif args.pose_group == "m":
    gt_filenames = [f for f in gt_filenames if f[4:8] in ANGLES_MODERATE]
    comp_filenames = [f for f in comp_filenames if f[4:8] in ANGLES_MODERATE]
    print("☺️ Only compute pose group M (%s samples)" % len(gt_filenames))
elif args.pose_group == "custom":
    gt_filenames = [f for f in gt_filenames if f[4:8] in TARGET_POSES]
    comp_filenames = [f for f in comp_filenames if f[4:8] in TARGET_POSES]
    print(
        "🤡 Only compute custom poses: %s (%s samples)"
        % (", ".join(TARGET_POSES), len(gt_filenames))
    )

if args.match:
    common_filenames = set(gt_filenames) & set(comp_filenames)
    gt_filenames = [f for f in gt_filenames if f in common_filenames]
    comp_filenames = [f for f in comp_filenames if f in common_filenames]

gt_filenames.sort()
comp_filenames.sort()

assert len(gt_filenames) == len(
    comp_filenames
), "The number of GT and compared images must be the same. %s vs. %s" % (
    len(gt_filenames),
    len(comp_filenames),
)

get_psnr = create_metric("psnr", device=device)
get_ssim = create_metric("ssim", device=device)
get_lpips = create_metric("lpips", device=device)
get_niqe = create_metric("niqe", device=device)
get_fid = create_metric("fid", device=device)
get_musiq = create_metric("musiq", device=device)

# Identity Model
id_model = resnet_face18(use_se=False).to(device)
id_model.load_state_dict(
    torch.load("helpers/arcface/weights/resnet18_110_wo_dist.pth", weights_only=False)
)
id_model.requires_grad_(False)  # freeze the identity model
id_model.eval()

# Facial Landmarks Detector
landmarks_detector = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, flip_input=False
)


psnr = 0.0
ssim = 0.0
lpips = 0.0
niqe = 0.0
musiq = 0.0
ids = 0.0
lmd = 0.0
lmd_count = 0

for idx, filename in enumerate(tqdm(gt_filenames)):
    gt_filepath = os.path.join(args.gt_path, filename)
    gt_image = (
        to_tensor(
            Image.open(gt_filepath)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        .unsqueeze(0)
        .to(device)
    )

    comp_filepath = os.path.join(args.comp_path, filename)
    comp_image = (
        to_tensor(
            Image.open(comp_filepath)
            .convert("RGB")
            .resize((128, 128), resample=Image.Resampling.BICUBIC)
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        # Arcface Feature 추출
        pred_feature = id_model(process_arcface_input(comp_image))
        gt_feature = id_model(process_arcface_input(gt_image))

        pred_feature = F.normalize(pred_feature, dim=1)
        gt_feature = F.normalize(gt_feature, dim=1)

        ids_score = F.cosine_similarity(pred_feature, gt_feature, dim=1).item()

        # 얼굴 랜드마크 추출
        comp_lds = landmarks_detector.get_landmarks(comp_filepath)
        gt_lds = landmarks_detector.get_landmarks(gt_filepath)

        if comp_lds is None or gt_lds is None:
            print("No face detected. Skip calculating lmd.")
            continue

        comp_landmarks, gt_landmarks = np.array(comp_lds[0]), np.array(gt_lds[0])
        lmd_score = np.linalg.norm(comp_landmarks - gt_landmarks, axis=1).mean()
        lmd_count += 1

    psnr += get_psnr(gt_image, comp_image).item()
    ssim += get_ssim(gt_image, comp_image).item()
    lpips += get_lpips(gt_image, comp_image).item()
    niqe += get_niqe(comp_image).item()
    musiq += get_musiq(comp_image).item()

    ids += ids_score
    lmd += lmd_score

    shutil.copy(gt_filepath, os.path.join(TEMP_DIR, "gt", filename))
    shutil.copy(comp_filepath, os.path.join(TEMP_DIR, "comp", filename))

fid = get_fid(os.path.join(TEMP_DIR, "gt"), os.path.join(TEMP_DIR, "comp"))

scores = {
    "PSNR": psnr / len(gt_filenames),
    "SSIM": ssim / len(gt_filenames),
    "LPIPS": lpips / len(gt_filenames),
    "NIQE": niqe / len(gt_filenames),
    "MUSIQ": musiq / len(gt_filenames),
    "FID (HQ)": fid.item(),
    "IDS": ids / len(gt_filenames),
    "LMD": lmd / lmd_count,
}
results = ["%s=%s" % (k, v) for k, v in scores.items()]

print("✅ Done!")
print("\n".join(results))

shutil.rmtree(TEMP_DIR)
