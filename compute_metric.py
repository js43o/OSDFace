import os
from pyiqa import create_metric
from tqdm import tqdm
import argparse

DEVICE = "cuda"
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    default="./outputs/multipie_validation_128_resized",
    help="Path of restored images",
)
parser.add_argument(
    "-g",
    "--gt_path",
    type=str,
    default="../../datasets/multipie_validation_128/gt",
    help="Path of GT images",
)
args = parser.parse_args()

# Prepare metrics
psnr_metric = create_metric("psnr", device=DEVICE)
ssim_metric = create_metric("ssim", device=DEVICE)
lpips_metric = create_metric("lpips", device=DEVICE)
niqe_metric = create_metric("niqe", device=DEVICE)
fid_metric = create_metric("fid", device=DEVICE)

psnr = 0.0
ssim = 0.0
lpips = 0.0
niqe = 0.0

# Read images
gt_images = sorted(os.listdir(args.gt_path))
pred_images = sorted(os.listdir(args.input_path))

assert len(gt_images) == len(
    pred_images
), "The number of GT and predicted images must be the same."

# Compute metrics
for gt_name, pred_name in tqdm(zip(gt_images, pred_images), total=len(gt_images)):
    assert (
        gt_name == pred_name
    ), "Each pair of GT and predicted image must have the same filename."

    gt_path = os.path.join(args.gt_path, gt_name)
    pred_path = os.path.join(args.input_path, pred_name)

    psnr += psnr_metric(pred_path, gt_path).item()
    ssim += ssim_metric(pred_path, gt_path).item()
    lpips += lpips_metric(pred_path, gt_path).item()
    niqe += niqe_metric(pred_path).item()


fid_score = fid_metric(
    os.path.join(args.gt_path),
    os.path.join(args.input_path),
)

logs = {
    "PSNR": psnr / len(gt_images),
    "SSIM": ssim / len(gt_images),
    "LPIPS": lpips / len(gt_images),
    "NIQE": niqe / len(gt_images),
    "FID": fid_score,
}
print(" ✅ Done!")
print(logs)
