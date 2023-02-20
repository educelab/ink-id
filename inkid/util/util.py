"""Miscellaneous operations used in ink-id."""

from copy import deepcopy
import itertools
from io import BytesIO
import json
import requests
from urllib.parse import urlsplit, urlunsplit

import math
import os
from pathlib import Path
from xml.dom.minidom import parseString

from dicttoxml import dicttoxml
from matplotlib import colormaps
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

import inkid


def are_coordinates_within(p1, p2, distance):
    """Return if two points would have overlapping boxes.

    Given two (x, y) points and a distance, imagine creating squares
    with side lengths equal to that distance and centering them on
    each point. Return if the squares overlap at all.

    """
    (x1, y1) = p1
    (x2, y2) = p2
    return abs(x1 - x2) < distance and abs(y1 - y2) < distance


def save_volume_to_image_stack(volume, dirname):
    """Save a volume to a stack of .tif images.

    Given a volume as an np.array of [0, 1] floats and a directory name, save the volume as a stack of .tif images in
    that directory, with filenames starting at 0 and going up to the z height of the volume.

    """
    Path(dirname).mkdir(parents=True, exist_ok=True)
    for z in range(volume.shape[0]):
        image = volume[z, :, :]
        image *= np.iinfo(np.uint16).max  # Assume incoming [0, 1] floats
        image = image.astype(np.uint16)
        image = Image.fromarray(image)
        image.save(Path(dirname) / f"{z}.tif")


def remap(x, in_min, in_max, out_min, out_max):
    val = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    if math.isnan(val):
        return 0
    else:
        return val


def get_descriptive_statistics(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    t_mean = tensor.mean()
    t_std = tensor.std()
    t_median = np.median(tensor)
    t_var = tensor.var()

    return np.array([t_min, t_max, t_mean, t_std, t_median, t_var])


# https://www.geeksforgeeks.org/serialize-python-dictionary-to-xml/
def dict_to_xml(data):
    xml = dicttoxml(data)
    dom = parseString(xml)
    return dom.toprettyxml()


def perform_validation(model, dataloader, metrics, device, domain_transfer_model=None):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {
            label_type: {metric: [] for metric in metrics[label_type]}
            for label_type in metrics
        }
        for batch in tqdm(dataloader):
            xb = batch["feature"].to(device)
            if domain_transfer_model is not None:
                xb = torch.squeeze(xb, 1)
                xb = domain_transfer_model(xb)
                xb = torch.unsqueeze(xb, 1)
            preds = model(xb)
            total_loss = None
            for label_type in getattr(model, "module", model).labels:
                yb = (
                    xb.clone()
                    if label_type == "autoencoded"
                    else batch[label_type].to(device)
                )
                pred = preds[label_type]
                for metric, fn in metrics[label_type].items():
                    metric_result = fn(pred, yb)
                    metric_results[label_type][metric].append(metric_result)
                    if metric == "loss":
                        if total_loss is None:
                            total_loss = metric_result
                        else:
                            total_loss = total_loss + metric_result
            if total_loss is not None:
                if "total" not in metric_results:
                    metric_results["total"] = {"loss": []}
                metric_results["total"]["loss"].append(total_loss)
    model.train()
    return metric_results


def generate_prediction_images(
    dataloader, model, device, predictions_dir, suffix, prediction_averaging, global_step, domain_transfer_model=None
):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_copy = deepcopy(
                batch
            )  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            batch_metadata = batch_copy["feature_metadata"]
            batch_features = batch_copy["feature"]
            # Only do those label types actually included in model output
            for label_type in {
                "ink_classes",
                "rgb_values",
                "volcart_texture",
            }.intersection(getattr(model, "module", model).labels):
                output_size = {
                    "volcart_texture": 1,
                    "ink_classes": 2,
                    "rgb_values": 3,
                }[label_type]
                # Smooth predictions via augmentation. Augment each subvolume 8-fold via rotations and flips
                if prediction_averaging:
                    rotations = range(4)
                    flips = [False, True]
                else:
                    rotations = [0]
                    flips = [False]
                batch_preds = None
                for rotation, flip in itertools.product(rotations, flips):
                    # Example batch_features.shape = [64, 1, 48, 48, 48] (BxCxDxHxW)
                    # Augment via rotation and flip
                    aug_pxb = batch_features.rot90(rotation, [3, 4])
                    if flip:
                        aug_pxb = aug_pxb.flip(4)
                    aug_pxb = aug_pxb.to(device)
                    if domain_transfer_model is not None:
                        aug_pxb = torch.squeeze(aug_pxb, 1)
                        aug_pxb = domain_transfer_model(aug_pxb)
                        aug_pxb = torch.unsqueeze(aug_pxb, 1)
                    preds = model(aug_pxb)
                    pred = preds[label_type]
                    if label_type == "ink_classes":
                        pred = F.softmax(pred, dim=1)
                    pred = deepcopy(
                        pred.cpu()
                    )  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
                    # Example pred.shape = [64, 2, 48, 48] (BxCxHxW)
                    # Undo flip and rotation
                    if flip:
                        pred = pred.flip(3)
                    pred = pred.rot90(-rotation, [2, 3])
                    pred = np.expand_dims(pred.numpy(), axis=0)
                    # Example pred.shape = [1, 64, 2, 48, 48] (BxCxHxW)
                    # Save this augmentation to the batch totals
                    if batch_preds is None:
                        batch_preds = np.zeros(
                            (
                                0,
                                batch_features.shape[0],
                                output_size,
                                pred.shape[3],
                                pred.shape[4],
                            )
                        )
                    batch_preds = np.append(batch_preds, pred, axis=0)
                # Average over batch of predictions after augmentation
                batch_pred = batch_preds.mean(0)
                # Separate these three lists
                source_paths, xs, ys, _, _, _, _, _, _ = batch_metadata
                for prediction, source_path, x, y in zip(
                    batch_pred, source_paths, xs, ys
                ):
                    dataloader.dataset.source(source_path).store_prediction(
                        int(x), int(y), prediction, label_type
                    )
    for region in dataloader.dataset.regions():
        region.write_predictions(predictions_dir, suffix, step=global_step)
        region.reset_predictions()
    model.train()


def json_schema(schema_name):
    """Return the JSON schema of the specified name from the inkid/schemas directory."""
    file_path = Path(inkid.__file__).parent / "schemas" / f"{schema_name}.schema.json"
    with open(file_path, "r") as f:
        return json.load(f)


def dummy_volpkg_path():
    return Path(inkid.__file__).parent / "examples" / "DummyTest.volpkg"


def get_raw_data_from_file_or_url(filename, return_relative_url=False):
    """Return the raw file contents from a filename or URL.

    Supports absolute and relative file paths as well as the http and https
    protocols.

    """
    url = urlsplit(filename)
    if url.scheme in ("http", "https"):
        response = requests.get(filename)
        if response.status_code != 200:
            raise ValueError(
                f"Unable to fetch URL " f"(code={response.status_code}): {filename}"
            )
        data = response.content
    elif url.scheme == "":
        with open(filename, "rb") as f:
            data = f.read()
    else:
        raise ValueError(f"Unsupported URL: {filename}")
    relative_url = (
        url.scheme,
        url.netloc,
        os.path.dirname(url.path),
        url.query,
        url.fragment,
    )
    if return_relative_url:
        return BytesIO(data), relative_url
    else:
        return BytesIO(data)


def normalize_path(path, relative_url):
    """Normalize path to be absolute and with URL where appropriate."""
    url = urlsplit(path)
    # Leave existing URLs and absolute file paths alone
    if url.scheme != "" or os.path.isabs(path):
        return path
    # For all others, we generate a new URL relative to the
    # region set file itself. This handles all schemas as well
    # as regular file paths.
    new_url = list(relative_url)
    new_url[2] = os.path.abspath(os.path.join(new_url[2], path))
    return urlunsplit(new_url)


def uint16_to_float32_normalized_0_1(img):
    # Convert to float
    img = np.asarray(img, np.float32)
    # Normalize to [0, 1]
    img *= 1.0 / np.iinfo(np.uint16).max
    return img


def window_0_1_array(arr, window_min, window_max):
    """Assumes input array is in [0, 1] range and contrast stretches to new min/max"""
    clipped = np.clip(arr, window_min, window_max)
    shifted = clipped - window_min
    windowed = shifted / (window_max - window_min)
    return windowed


def plot_with_colorbar(img, cmap="turbo", vmin=None, vmax=None):
    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    # convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close(fig)
    return pil_img


def subvolume_to_sample_img(
    subvolume,
    volume,
    vol_coord,
    padding,
    background_color,
    autoencoded_subvolume=None,
    domain_transfer_subvolume=None,
    include_vol_slices=True,
):
    max_size = (300, 300)
    z_shape, y_shape, x_shape = subvolume.shape

    sub_images = []

    # Get central slices of subvolume
    subvolume_slices = []
    z_idx: int = z_shape // 2
    y_idx: int = y_shape // 2
    x_idx: int = x_shape // 2
    subvolume_slices.append(subvolume[z_idx, :, :])
    subvolume_slices.append(subvolume[:, y_idx, :])
    subvolume_slices.append(subvolume[:, :, x_idx])

    # Get min/max across all three slices
    min_val = min([s.min() for s in subvolume_slices])
    max_val = max([s.max() for s in subvolume_slices])

    # Plot with color bar
    for subvolume_slice in subvolume_slices:
        subvolume_img = plot_with_colorbar(subvolume_slice, vmin=min_val, vmax=max_val)
        sub_images.append(subvolume_img)

    if autoencoded_subvolume is not None:
        # Get central slices of autoencoded subvolume
        autoencoded_subvolume_slices = []
        autoencoded_subvolume_slices.append(autoencoded_subvolume[z_idx, :, :])
        autoencoded_subvolume_slices.append(autoencoded_subvolume[:, y_idx, :])
        autoencoded_subvolume_slices.append(autoencoded_subvolume[:, :, x_idx])

        # Get min/max across all three slices
        min_val = min([s.min() for s in autoencoded_subvolume_slices])
        max_val = max([s.max() for s in autoencoded_subvolume_slices])

        # Plot with color bar
        for autoencoded_subvolume_slice in autoencoded_subvolume_slices:
            autoencoded_subvolume_img = plot_with_colorbar(autoencoded_subvolume_slice, vmin=min_val, vmax=max_val)
            sub_images.append(autoencoded_subvolume_img)

    if domain_transfer_subvolume is not None:
        # Get central slices of domain transfer subvolume
        domain_transfer_subvolume_slices = []
        domain_transfer_subvolume_slices.append(domain_transfer_subvolume[z_idx, :, :])
        domain_transfer_subvolume_slices.append(domain_transfer_subvolume[:, y_idx, :])
        domain_transfer_subvolume_slices.append(domain_transfer_subvolume[:, :, x_idx])

        # Get min/max across all three slices
        min_val = min([s.min() for s in domain_transfer_subvolume_slices])
        max_val = max([s.max() for s in domain_transfer_subvolume_slices])

        # Plot with color bar
        for domain_transfer_subvolume_slice in domain_transfer_subvolume_slices:
            domain_transfer_subvolume_img = plot_with_colorbar(domain_transfer_subvolume_slice, vmin=min_val, vmax=max_val)
            sub_images.append(domain_transfer_subvolume_img)

    if include_vol_slices:
        # Get intersection slices of volume for each axis
        vol_slices = []
        for axis in (0, 1, 2):  # x, y, z
            # Get slice image from volume
            vol_slice_idx = vol_coord[axis]
            if axis == 0:
                vol_slice = volume.x_slice(vol_slice_idx)
            elif axis == 1:
                vol_slice = volume.y_slice(vol_slice_idx)
            else:
                vol_slice = volume.z_slice(vol_slice_idx)
            vol_slice = uint16_to_float32_normalized_0_1(vol_slice)
            # Convert to PIL Image
            vol_slice_img = Image.fromarray(vol_slice, mode="F")
            # Draw crosshairs around subvolume
            draw = ImageDraw.Draw(vol_slice_img)
            # Find (x, y) coordinates in this slice image space
            subvolume_img_x_y = list(vol_coord).copy()
            subvolume_img_x_y.pop(axis)
            x, y = subvolume_img_x_y
            w, h = vol_slice_img.size
            # Draw lines through that (x, y) but don't draw them at the center, so the actual spot is not obscured
            c = np.amax(vol_slice)
            r = max(vol_slice_img.size) // 50
            width = max(vol_slice_img.size) // 100
            draw.line([(0, y), (x - r, y)], fill=c, width=width)  # Left of (x, y)
            draw.line([(x + r, y), (w, y)], fill=c, width=width)  # Right of (x, y)
            draw.line([(x, 0), (x, y - r)], fill=c, width=width)  # Above (x, y)
            draw.line([(x, y + r), (x, h)], fill=c, width=width)  # Below (x, y)
            # Reduce size and add to list of images for this subvolume
            vol_slice_img.thumbnail(max_size)
            vol_slices.append(np.array(vol_slice_img))

        # Get min/max across all three slices
        min_val = min([s.min() for s in vol_slices])
        max_val = max([s.max() for s in vol_slices])

        # Plot with color bar
        for vol_slice in vol_slices:
            vol_slice_img = plot_with_colorbar(vol_slice, vmin=min_val, vmax=max_val)
            sub_images.append(vol_slice_img)

    width = sum([s.size[0] for s in sub_images]) + padding * (len(sub_images) - 1)
    height = max([s.size[1] for s in sub_images])

    img = Image.new("RGB", (width, height), background_color)
    x_ctr = 0
    for s in sub_images:
        img.paste(s, (x_ctr, 0))
        x_ctr += s.size[0] + padding

    return img


def create_colormap_swatch(cmap, w, h):
    swatch = Image.new("RGB", (w, h))
    for x in range(w):
        intensity = int((x / w) * 255)
        swatch.paste(
            (intensity, intensity, intensity), (x, 0, x + 1, h)
        )
    if cmap is not None:
        color_map = colormaps[cmap]
        swatch = swatch.convert("L")
        img_data = np.array(swatch)
        swatch = Image.fromarray(np.uint8(color_map(img_data) * 255))
    return swatch


def save_subvolume_batch_to_img(
    model,
    device,
    dataloader,
    outdir,
    filename,
    padding=10,
    background_color=(128, 128, 128),
    include_autoencoded=False,
    include_vol_slices=True,
    domain_transfer_model=None,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))
    subvolumes = torch.squeeze(batch["feature"], 1)  # Remove channels axis

    if include_autoencoded:
        model.eval()  # Turn off training mode for batch norm and dropout purposes
        with torch.no_grad():
            autoencodeds = model(batch["feature"].to(device))["autoencoded"]
            autoencodeds = autoencodeds.cpu()
            autoencodeds = np.squeeze(autoencodeds, axis=1)
        model.train()
    else:
        autoencodeds = [None] * len(batch["feature"])

    if domain_transfer_model is not None:
        domain_transfer_model.eval()
        with torch.no_grad():
            domain_transfer_subvolumes = domain_transfer_model(subvolumes.to(device))
            domain_transfer_subvolumes = np.asarray(domain_transfer_subvolumes.cpu())
        domain_transfer_model.train()
    else:
        domain_transfer_subvolumes = [None] * len(batch["feature"])

    imgs = []
    for i, (subvolume, autoencoded, domain_transfer_subvolume) in enumerate(zip(subvolumes, autoencodeds, domain_transfer_subvolumes)):
        dataloader.dataset.source(batch["feature_metadata"].path[i])
        volume = dataloader.dataset.source(batch["feature_metadata"].path[i]).volume
        imgs.append(
            subvolume_to_sample_img(
                subvolume,
                volume,
                (
                    batch["feature_metadata"].x[i],
                    batch["feature_metadata"].y[i],
                    batch["feature_metadata"].z[i],
                ),
                padding,
                background_color,
                autoencoded_subvolume=autoencoded,
                domain_transfer_subvolume=domain_transfer_subvolume,
                include_vol_slices=include_vol_slices,
            )
        )

    width = imgs[0].size[0] + padding * 2
    height = imgs[0].size[1] * len(imgs) + padding * (len(imgs) + 1)

    composite_img = Image.new("RGB", (width, height), background_color)

    for i, img in enumerate(imgs):
        composite_img.paste(img, (padding, img.size[1] * i + padding * (i + 1)))

    outfile = Path(outdir) / filename
    composite_img.save(outfile)
