import argparse
import itertools
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def normalize_to_float_0_1(a: np.array):
    if a.dtype == np.uint16:
        a = np.asarray(a, np.float32)
        a *= 1.0 / np.iinfo(np.uint16).max
    else:
        raise NotImplementedError(f"dtype {a.dtype} not implemented")
    return a


class StackDataset(Dataset):
    def __init__(
        self,
        feature_img_path: Path,
        label_img_path: Optional[Path] = None,
        mask_img_path: Optional[Path] = None,
        patch_size: int = 64,
        stride: int = 1,
    ):
        # Load the input image stack
        assert ".tif" in str(feature_img_path).lower()
        self.feature_img: np.array = iio.imread(feature_img_path)
        # .tif stack loaded as (C, H, W)
        assert len(self.feature_img.shape) == 3

        # Load the label image
        self.label_img: Optional[np.array] = iio.imread(label_img_path) if label_img_path else None
        if self.label_img is not None:
            # Greyscale loaded (H, W)
            if len(self.label_img.shape) == 2:
                self.label_img = np.expand_dims(self.label_img, axis=0)
            # RGB loaded (H, W, C)
            elif len(self.label_img.shape) == 3:
                self.label_img = np.moveaxis(self.label_img, [0, 1, 2], [1, 2, 0])
            # Now should be (C, H, W)
            assert len(self.label_img.shape) == 3
            # Should be greyscale or RGB (not e.g. RGBA)
            assert self.label_img.shape[0] in [1, 3]
            # And should match feature image in HxW
            assert self.label_img.shape[-2:] == self.feature_img.shape[-2:]

        self.patch_size: int = patch_size
        self.stride: int = stride

        # Generate possible points
        ys = range(0, self.feature_img.shape[1], stride)
        xs = range(0, self.feature_img.shape[2], stride)
        self.points_list = list(itertools.product(ys, xs))

        # TODO mask these

    def __len__(self) -> int:
        return len(self.points_list)

    def __getitem__(self, idx):
        y, x = self.points_list[idx]

        feature = self.feature_img[:, y:y+self.patch_size, x:x+self.patch_size].copy()
        feature = normalize_to_float_0_1(feature)

        label = None
        if self.label_img is not None:
            label = self.label_img[:, y:y+self.patch_size, x:x+self.patch_size].copy()

        return {
            "feature": feature,
            "label": label,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input-stack", required=True)
    parser.add_argument("--train-label-image", required=True)
    args = parser.parse_args()

    train_input_stack_path: Path = Path(args.train_input_stack)
    train_label_image_path: Path = Path(args.train_label_image)

    train_dataset: StackDataset = StackDataset(
        train_input_stack_path,
        train_label_image_path,
        stride=16,
    )

    # import matplotlib.pyplot as plt
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None)
    # for sample in train_dataloader:
    #     label = np.array(sample["label"])
    #     label = np.moveaxis(label, [1, 2, 0], [0, 1, 2])
    #     plt.imshow(label)
    #     plt.show()
    #     input()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)




if __name__ == "__main__":
    main()
