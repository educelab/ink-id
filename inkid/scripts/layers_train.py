import argparse
import datetime
import itertools
import logging
import multiprocessing
from pathlib import Path
import random
import time
from typing import Optional

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import inkid


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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
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
        feature_imgs: list[np.array],
        label_imgs: Optional[list[np.array]] = None,
        patch_size: int = 64,
        stride: int = 1,
    ):
        self.feature_imgs = feature_imgs
        for feature_img in self.feature_imgs:
            # .tif stack loaded as (C, H, W)
            assert len(feature_img.shape) == 3

        # Make sure feature images all the same depth
        assert len(np.unique([img.shape[0] for img in self.feature_imgs]))

        # Load the label image
        self.label_imgs = None
        if label_imgs is not None:
            self.label_imgs = []
            for label_img in label_imgs:
                # Greyscale or indexed loaded (H, W)
                if len(label_img.shape) == 2:
                    label_img = np.expand_dims(label_img, axis=0)
                # RGB loaded (H, W, C)
                elif len(label_img.shape) == 3:
                    label_img = np.moveaxis(label_img, [0, 1, 2], [1, 2, 0])

                # Now should be (C, H, W)
                assert len(label_img.shape) == 3
                # Should be greyscale or RGB (not e.g. RGBA)
                assert label_img.shape[0] in [1, 3]

                self.label_imgs.append(label_img)

            # Label images should match feature images in HxW
            for feature_img, label_img in zip(self.feature_imgs, self.label_imgs):
                assert label_img.shape[-2:] == feature_img.shape[-2:]

            # Label images should all be the same depth
            assert len(np.unique([img.shape[0] for img in self.label_imgs]))

        self.patch_size: int = patch_size
        self.stride: int = stride

        # Generate possible points
        self.points_lists = []
        for feature_img in self.feature_imgs:
            ys = range(0, feature_img.shape[1], stride)
            xs = range(0, feature_img.shape[2], stride)
            self.points_lists.append(list(itertools.product(ys, xs)))

        # TODO mask these

    def __len__(self) -> int:
        return sum([len(points_list) for points_list in self.points_lists])

    def __getitem__(self, idx):
        i = x = y = None
        for i, points_list in enumerate(self.points_lists):
            points_list_len = len(points_list)
            if idx < points_list_len:
                y, x = points_list[idx]
                break
            idx -= points_list_len

        if y is None:
            raise IndexError

        feature = np.zeros(
            (self.feature_imgs[0].shape[0], self.patch_size, self.patch_size),
            dtype=self.feature_imgs[0].dtype,
        )
        a = self.feature_imgs[i][:, y : y + self.patch_size, x : x + self.patch_size]
        feature[: a.shape[0], : a.shape[1], : a.shape[2]] = a
        feature = normalize_to_float_0_1(feature)

        label = None
        if self.label_imgs is not None:
            label = np.zeros(
                (self.label_imgs[i].shape[0], self.patch_size, self.patch_size),
                dtype=self.label_imgs[i].dtype,
            )
            b = self.label_imgs[i][:, y : y + self.patch_size, x : x + self.patch_size]
            label[: b.shape[0], : b.shape[1], : b.shape[2]] = b

        return {
            "feature": feature,
            "label": label,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input-stack", required=True)
    parser.add_argument("--train-label-image", required=True)
    parser.add_argument(
        "--output", metavar="output", help="output directory", required=True
    )
    parser.add_argument(
        "--dataloaders-num-workers", metavar="n", type=int, default=None
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--learning-rate", metavar="n", type=float, default=0.001
    )  # TODO use this?
    parser.add_argument("--summary-every-n-batches", metavar="n", type=int, default=10)
    args = parser.parse_args()

    dir_name = datetime.datetime.today().strftime("%Y-%m-%d_%H.%M.%S")
    output_path = Path(args.output) / dir_name
    output_path.mkdir(parents=True)

    # Define directories for prediction images and checkpoints
    predictions_dir = Path(output_path) / "predictions"
    predictions_dir.mkdir()
    checkpoints_dir = Path(output_path) / "checkpoints"
    checkpoints_dir.mkdir()
    diagnostic_images_dir = Path(output_path) / "diagnostic_images"
    diagnostic_images_dir.mkdir()

    if args.dataloaders_num_workers is None:
        args.dataloaders_num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Fix random seeds
    def fix_random_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    fix_random_seed(args.random_seed)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_path / f"{dir_name}.log"),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"PyTorch device: {device}")

    train_input_stack_path: Path = Path(args.train_input_stack)
    train_label_image_path: Path = Path(args.train_label_image)

    # Load the input image stack
    assert ".tif" in str(train_input_stack_path).lower()
    train_feature_img = iio.imread(train_input_stack_path)

    # Load the label image
    train_label_img = iio.imread(train_label_image_path).astype(np.int64)

    train_ds: StackDataset = StackDataset(
        [train_feature_img],
        [train_label_img],
        stride=16,
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # TODO check that indexed colors are loaded as the values I expect
    model = UNet(n_channels=65, n_classes=2)
    model.to(device)

    opt = torch.optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy": inkid.metrics.accuracy,
        "precision": inkid.metrics.precision,
        "recall": inkid.metrics.recall,
        "fbeta": inkid.metrics.fbeta,
        "auc": inkid.metrics.auc,
    }
    metric_results = {metric: [] for metric in metrics}

    model.train()
    end = time.time()
    for batch_num, batch in enumerate(train_dl):
        x_b = batch["feature"].to(device)
        y_b = torch.squeeze(batch["label"]).to(device)
        pred_b = model(x_b)

        for metric_name, metric_fn in metrics.items():
            metric_result = metric_fn(pred_b, y_b)
            metric_results[metric_name].append(metric_result)

        loss = criterion(pred_b, y_b)
        print(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch_num % args.summary_every_n_batches == 0:
            logging.info(
                # f"Epoch: {epoch:>5d}/{args.training_epochs:<5d}"
                f"Batch: {batch_num:>5d}/{len(train_dl):<5d} "
                # f"{inkid.metrics.metrics_str(metric_results)} "
                f"Seconds: {time.time() - end:5.3g}"
            )
            end = time.time()


if __name__ == "__main__":
    main()
