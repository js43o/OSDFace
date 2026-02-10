import os
import cv2
from models import *
import torch
import numpy as np
from torch.nn import DataParallel
import argparse

from models.arcface.models import resnet_face18

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    default="./outputs/multipie_validation_128_resized",
    help="Path of restored images for computing identity similarity",
)
parser.add_argument(
    "-g",
    "--gt_path",
    type=str,
    default="../../datasets/multipie_validation_128/gt",
    help="Path of GT images for computing identity similarity",
)
args = parser.parse_args()


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image.shape[0] != 128 or image.shape[1] != 128:
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


model = DataParallel(resnet_face18(use_se=False))
model.load_state_dict(
    torch.load("models/arcface/weights/resnet18_110.pth", weights_only=False)
)
model.to(torch.device("cuda"))
model.eval()

gt_images = [
    load_image(os.path.join(args.gt_path, filename))
    for filename in sorted(os.listdir(args.gt_path))
]
input_images = [
    load_image(os.path.join(args.input_path, filename))
    for filename in sorted(os.listdir(args.input_path))
]

assert len(gt_images) == len(input_images)


score = 0.0
LEN_NUM = len(gt_images)

with torch.no_grad():
    for i in range(LEN_NUM):
        print("Computing IDS for %s of %s images..." % (i + 1, LEN_NUM))
        gt_image = torch.from_numpy(gt_images[i]).to(torch.device("cuda"))
        input_image = torch.from_numpy(input_images[i]).to(torch.device("cuda"))

        gt_feature = model(gt_image)[0].cpu().numpy()
        input_feature = model(input_image).cpu()[0].numpy()

        score += np.dot(gt_feature, input_feature.T) / (
            np.linalg.norm(gt_feature) * np.linalg.norm(input_feature)
        )

score /= LEN_NUM

print("Computed Identity Similarity: %.4f" % score)
