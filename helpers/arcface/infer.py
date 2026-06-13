import torch
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import (
    to_tensor,
    rgb_to_grayscale,
    resize,
    InterpolationMode,
)
from PIL import Image

from models import resnet_face18


# ArcFace 입력 이미지 변환 메서드
# RGB [0, 1] -> Grayscale [0, 1]
def process_arcface_input(image: torch.Tensor):
    output = resize(image, size=(128, 128), interpolation=InterpolationMode.BICUBIC)
    output = rgb_to_grayscale(output, num_output_channels=1)

    return output


"""
image = cv2.imread(
    "/vcl4/Jiseung/datasets/lfw_aligned/Abel_Pacheco_0001.png", 0
)  # (H, W, 1)

image = np.dstack((image, np.fliplr(image)))  # (H, W, 2)
image = image.transpose((2, 0, 1))  # (2, H, W)
image = image[:, np.newaxis, :, :]  # (2, 1, H, W)
image = image.astype(np.float32, copy=False)
image -= 127.5
image /= 127.5  # [-1, 1]

print(image.shape, image.min(), image.max())

"""
# print(load_image("/vcl4/Jiseung/datasets/lfw_aligned/Abel_Pacheco_0001.png").shape)

a = (
    to_tensor(
        Image.open("/vcl4/Jiseung/datasets/lfw_aligned/Abel_Pacheco_0001.png")
        .convert("RGB")
        .resize((128, 128))
    )
    * 2.0
    - 1.0
)
b = (
    to_tensor(
        Image.open("/vcl4/Jiseung/datasets/lfw_aligned/Abel_Pacheco_0002.png")
        .convert("RGB")
        .resize((128, 128))
    )
    * 2.0
    - 1.0
)
c = (
    to_tensor(
        Image.open("/vcl4/Jiseung/datasets/lfw_aligned/Abel_Aguilar_0001.png")
        .convert("RGB")
        .resize((128, 128))
    )
    * 2.0
    - 1.0
)

restored = torch.stack((a, c))
gt = torch.stack((b, b))

# Identity Model
id_model = resnet_face18(use_se=False)
id_model.load_state_dict(
    torch.load("weights/resnet18_110_wo_dist.pth", weights_only=False)
)
id_model.eval()


restored_feat = id_model(process_arcface_input(restored))
gt_feat = id_model(process_arcface_input(gt))

print(cosine_similarity(restored_feat, gt_feat))
