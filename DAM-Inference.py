import cv2
import matplotlib
import numpy as np
import os
import torch
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


INPUT_SIZE = 518

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

NUM_WORKERS = 0
BATCH_SIZE = 1
IMG_ROOT = "../data/refcoco/train2014"
BUILT_DEPTH_ROOT = "../data/refcoco/train2014_depth"
os.makedirs(BUILT_DEPTH_ROOT, exist_ok=True)


class DAMDataset(Dataset):

    def __init__(self, imgs_path):
        self.imgs_path = imgs_path
        self.transform = Compose(
            [
                Resize(
                    width=INPUT_SIZE,
                    height=INPUT_SIZE,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img_name = self.imgs_path[index].stem
        img_path = self.imgs_path[index].as_posix()
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform({"image": img})["image"]
        img = torch.from_numpy(img)
        return img_name, H, W, img


def get_imgs_from_directory(directory):
    images_path = [file for file in Path(directory).glob("*.jpg") if file.is_file()]
    assert len(images_path) > 0, "No images found in {}".format(directory)
    return images_path


def post_process_depth(depth, h, w):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[
        0, 0
    ]
    return depth


def build_model(device, weight_name):
    depth_anything = DepthAnythingV2(**model_configs[weight_name])
    path = f"checkpoints/depth_anything_v2_{weight_name}.pth"
    depth_anything.load_state_dict(torch.load(path, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything


def visualize_depth(depth):
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth


if __name__ == "__main__":
    DEVICE = "cuda"
    weight_name = "vitl"
    model = build_model(DEVICE, weight_name)
    imgs_path = get_imgs_from_directory(IMG_ROOT)

    dataset = DAMDataset(imgs_path)
    dataloader = DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for img_names, hs, ws, img_tensor in dataloader:
        img_tensor = img_tensor.to(DEVICE)
        depth = model(img_tensor)
        # depth = post_process_depth(depth)
        # depth = visualize_depth(depth)
