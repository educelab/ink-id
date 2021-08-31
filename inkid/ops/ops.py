"""Miscellaneous operations used in ink-id."""

from collections import namedtuple
from copy import deepcopy
import itertools
import inspect
from io import BytesIO
import json
import logging
import requests
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import math
import os
import subprocess
from xml.dom.minidom import parseString

from dicttoxml import dicttoxml
from matplotlib import cm
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

import inkid


def add_subvolume_args(parser):
    parser.add_argument('--subvolume-method', default='nearest_neighbor',
                        help='method for sampling subvolumes', choices=['nearest_neighbor', 'interpolated'])
    parser.add_argument('--subvolume-shape-microns', metavar='um', nargs=3, type=float, default=None,
                        help='subvolume shape (microns) in (z, y, x)')
    parser.add_argument('--subvolume-shape-voxels', metavar='n', nargs=3, type=int,
                        help='subvolume shape (voxels) in (z, y, x)', default=[48, 48, 48])
    parser.add_argument('--move-along-normal', metavar='n', type=float, default=0,
                        help='number of voxels to move along normal vector before sampling a subvolume')
    parser.add_argument('--normalize-subvolumes', action='store_true',
                        help='normalize each subvolume to zero mean and unit variance')


def take_from_dataset(dataset, n_samples):
    """Take only n samples from a dataset to reduce the size."""
    if n_samples < len(dataset):
        dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[n_samples, len(dataset) - n_samples],
            generator=torch.Generator().manual_seed(42)
        )[0]
    return dataset


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
    os.makedirs(dirname, exist_ok=True)
    for z in range(volume.shape[0]):
        image = volume[z, :, :]
        image *= np.iinfo(np.uint16).max  # Assume incoming [0, 1] floats
        image = image.astype(np.uint16)
        image = Image.fromarray(image)
        image.save(os.path.join(dirname, str(z) + '.tif'))


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

    return np.array([
        t_min,
        t_max,
        t_mean,
        t_std,
        t_median,
        t_var
    ])


def rclone_transfer_to_remote(rclone_remote, output_path):
    folders = []
    path = os.path.abspath(output_path)
    while True:
        path, folder = os.path.split(path)
        if folder != '':
            folders.append(folder)
        else:
            if path != '':
                folders.append(path)
            break
    folders.reverse()

    if rclone_remote is None:
        for folder in folders:
            if '-drive' in folder:
                rclone_remote = folder
                break

    if rclone_remote not in folders:
        print('Provided rclone transfer remote was not a directory '
              'name in the output path, so it is not clear where in the '
              'remote to put the files. Transfer canceled.')
    else:
        while folders.pop(0) != rclone_remote:
            continue

        command = [
            'rclone',
            'move',
            '-v',
            '--delete-empty-src-dirs',
            output_path,
            rclone_remote + ':' + os.path.join(*folders)
        ]
        logging.info(' '.join(command))
        subprocess.call(command)


# https://www.geeksforgeeks.org/serialize-python-dictionary-to-xml/
def dict_to_xml(data):
    xml = dicttoxml(data)
    dom = parseString(xml)
    return dom.toprettyxml()


def perform_validation(model, dataloader, metrics, device):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {label_type: {metric: [] for metric in metrics[label_type]} for label_type in metrics}
        for batch in tqdm(dataloader):
            xb = batch['feature'].to(device)
            preds = model(xb)
            total_loss = None
            for label_type in model.labels:
                yb = xb.clone() if label_type == 'autoencoded' else batch[label_type].to(device)
                if label_type == 'ink_classes':
                    _, yb = yb.max(1)
                pred = preds[label_type]
                for metric, fn in metrics[label_type].items():
                    metric_result = fn(pred, yb)
                    metric_results[label_type][metric].append(metric_result)
                    if metric == 'loss':
                        if total_loss is None:
                            total_loss = metric_result
                        else:
                            total_loss = total_loss + metric_result
            if total_loss is not None:
                if 'total' not in metric_results:
                    metric_results['total'] = {'loss': []}
                metric_results['total']['loss'].append(total_loss)
    model.train()
    return metric_results


def generate_prediction_images(dataloader, model, device, predictions_dir, suffix, prediction_averaging):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_copy = deepcopy(batch)  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            batch_metadata = batch_copy['feature_metadata']
            batch_features = batch_copy['feature']
            # Only do those label types actually included in model output
            for label_type in {'ink_classes', 'rgb_values', 'volcart_texture'}.intersection(model.labels):
                output_size = {
                    'volcart_texture': 1,
                    'ink_classes': 2,
                    'rgb_values': 3,
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
                    preds = model(aug_pxb.to(device))
                    pred = preds[label_type]
                    if label_type == 'ink_classes':
                        pred = F.softmax(pred, dim=1)
                    pred = deepcopy(pred.cpu())  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
                    # Example pred.shape = [64, 2, 48, 48] (BxCxHxW)
                    # Undo flip and rotation
                    if flip:
                        pred = pred.flip(3)
                    pred = pred.rot90(-rotation, [2, 3])
                    pred = np.expand_dims(pred.numpy(), axis=0)
                    # Example pred.shape = [1, 64, 2, 48, 48] (BxCxHxW)
                    # Save this augmentation to the batch totals
                    if batch_preds is None:
                        batch_preds = np.zeros((0, batch_features.shape[0], output_size, pred.shape[3], pred.shape[4]))
                    batch_preds = np.append(batch_preds, pred, axis=0)
                # Average over batch of predictions after augmentation
                batch_pred = batch_preds.mean(0)
                # Separate these three lists
                source_paths, xs, ys, _, _, _, _, _, _ = batch_metadata
                for prediction, source_path, x, y in zip(batch_pred, source_paths, xs, ys):
                    dataloader.dataset.get_source(source_path).store_prediction(
                        int(x),
                        int(y),
                        prediction,
                        label_type
                    )
    dataloader.dataset.save_predictions(predictions_dir, suffix)
    dataloader.dataset.reset_predictions()
    model.train()


def json_schema(schema_name):
    """Return the JSON schema of the specified name from the inkid/schemas directory."""
    file_path = os.path.join(os.path.dirname(inkid.__file__), 'schemas', schema_name + '.schema.json')
    with open(file_path, 'r') as f:
        return json.load(f)


def dummy_volpkg_path():
    return os.path.join(os.path.dirname(inkid.__file__), 'examples', 'DummyTest.volpkg')


def get_raw_data_from_file_or_url(filename, return_relative_url=False):
    """Return the raw file contents from a filename or URL.

    Supports absolute and relative file paths as well as the http and https
    protocols.

    """
    url = urlsplit(filename)
    if url.scheme in ('http', 'https'):
        response = requests.get(filename)
        if response.status_code != 200:
            raise ValueError(f'Unable to fetch URL '
                             f'(code={response.status_code}): {filename}')
        data = response.content
    elif url.scheme == '':
        with open(filename, 'rb') as f:
            data = f.read()
    else:
        raise ValueError(f'Unsupported URL: {filename}')
    relative_url = (url.scheme,
                    url.netloc,
                    os.path.dirname(url.path),
                    url.query,
                    url.fragment)
    if return_relative_url:
        return BytesIO(data), relative_url
    else:
        return BytesIO(data)


def normalize_path(path, relative_url):
    """Normalize path to be absolute and with URL where appropriate."""
    url = urlsplit(path)
    # Leave existing URLs and absolute file paths alone
    if url.scheme != '' or os.path.isabs(path):
        return path
    # For all others, we generate a new URL relative to the
    # region set file itself. This handles all schemas as well
    # as regular file paths.
    new_url = list(relative_url)
    new_url[2] = os.path.abspath(os.path.join(
        new_url[2],
        path
    ))
    return urlunsplit(new_url)


def uint16_to_float32_normalized_0_1(img):
    # Convert to float
    img = np.asarray(img, np.float32)
    # Normalize to [0, 1]
    img *= 1.0 / np.iinfo(np.uint16).max
    return img


def float_0_1_to_cmap_rgb_uint8(img, cmap='turbo'):
    color_map = cm.get_cmap(cmap)
    return Image.fromarray(np.uint8(color_map(img) * 255))


def subvolume_to_sample_img(subvolume, volume, vol_coord, padding, background_color,
                            autoencoded_subvolume=None, include_vol_slices=True):
    max_size = (300, 300)
    red = (255, 0, 0)
    z_shape, y_shape, x_shape = subvolume.shape

    sub_images = []

    # Get central slices of subvolume
    z_idx: int = z_shape // 2
    sub_images.append(float_0_1_to_cmap_rgb_uint8(subvolume[z_idx, :, :]))
    y_idx: int = y_shape // 2
    sub_images.append(float_0_1_to_cmap_rgb_uint8(subvolume[:, y_idx, :]))
    x_idx: int = x_shape // 2
    sub_images.append(float_0_1_to_cmap_rgb_uint8(subvolume[:, :, x_idx]))

    if include_vol_slices:
        # Get intersection slices of volume for each axis
        for axis in (0, 1, 2):  # x, y, z
            # Get slice image from volume
            vol_slice_idx = vol_coord[axis]
            if axis == 0:
                vol_slice = volume.x_slice(vol_slice_idx)
            elif axis == 1:
                vol_slice = volume.y_slice(vol_slice_idx)
            else:
                vol_slice = volume.z_slice(vol_slice_idx)
            # Color map
            vol_sub_img = float_0_1_to_cmap_rgb_uint8(vol_slice)
            # Draw crosshairs around subvolume
            draw = ImageDraw.Draw(vol_sub_img)
            # Find (x, y) coordinates in this slice image space
            subvolume_img_x_y = list(vol_coord).copy()
            subvolume_img_x_y.pop(axis)
            x, y = subvolume_img_x_y
            w, h = vol_sub_img.size
            # Draw lines through that (x, y) but don't draw them at the center, so the actual spot is not obscured
            r = max(vol_sub_img.size) // 50
            width = max(vol_sub_img.size) // 100
            draw.line([(0, y), (x - r, y)], fill=red, width=width)  # Left of (x, y)
            draw.line([(x + r, y), (w, y)], fill=red, width=width)  # Right of (x, y)
            draw.line([(x, 0), (x, y - r)], fill=red, width=width)  # Above (x, y)
            draw.line([(x, y + r), (x, h)], fill=red, width=width)  # Below (x, y)
            # Reduce size and add to list of images for this subvolume
            vol_sub_img.thumbnail(max_size)
            sub_images.append(vol_sub_img)

    if autoencoded_subvolume is not None:
        sub_images.append(float_0_1_to_cmap_rgb_uint8(autoencoded_subvolume[z_idx, :, :]))
        sub_images.append(float_0_1_to_cmap_rgb_uint8(autoencoded_subvolume[:, y_idx, :]))
        sub_images.append(float_0_1_to_cmap_rgb_uint8(autoencoded_subvolume[:, :, x_idx]))

    width = sum([s.size[0] for s in sub_images]) + padding * (len(sub_images) - 1)
    height = max([s.size[1] for s in sub_images])

    img = Image.new('RGB', (width, height), background_color)
    x_ctr = 0
    for s in sub_images:
        img.paste(s, (x_ctr, 0))
        x_ctr += s.size[0] + padding

    return img


def save_subvolume_batch_to_img(model, device, dataloader, outdir, padding=10, background_color=(128, 128, 128),
                                include_autoencoded=False, iteration=None, include_vol_slices=True):
    os.makedirs(outdir, exist_ok=True)

    batch = next(iter(dataloader))
    if include_autoencoded:
        model.eval()  # Turn off training mode for batch norm and dropout purposes
        with torch.no_grad():
            autoencodeds = model(batch['feature'].to(device))['autoencoded']
            autoencodeds = autoencodeds.cpu()
            autoencodeds = np.squeeze(autoencodeds, axis=1)
        model.train()
    else:
        autoencodeds = [None] * len(batch['feature'])
    subvolumes = np.squeeze(batch['feature'], axis=1)  # Remove channels axis

    imgs = []
    for i, (subvolume, autoencoded) in enumerate(zip(subvolumes, autoencodeds)):
        volume = dataloader.dataset.get_source(batch['feature_metadata'].path[i]).volume
        imgs.append(subvolume_to_sample_img(
            subvolume,
            volume,
            (batch['feature_metadata'].x[i], batch['feature_metadata'].y[i], batch['feature_metadata'].z[i]),
            padding,
            background_color,
            autoencoded_subvolume=autoencoded,
            include_vol_slices=include_vol_slices
        ))

    width = imgs[0].size[0] + padding * 2
    height = imgs[0].size[1] * len(imgs) + padding * (len(imgs) + 1)

    composite_img = Image.new('RGB', (width, height), background_color)

    for i, img in enumerate(imgs):
        composite_img.paste(img, (padding, img.size[1] * i + padding * (i + 1)))

    outfile = os.path.join(outdir, f'sample_batch')
    if iteration is not None:
        outfile += f'_{iteration}'
    outfile += '.png'
    composite_img.save(outfile)


# Automatically get the current list of classes in inkid.model https://stackoverflow.com/a/1796247
def model_choices():
    return [s for (s, _) in inspect.getmembers(sys.modules['inkid.model'], inspect.isclass)]


# https://discuss.pytorch.org/t/a-tensorboard-problem-about-use-add-graph-method-for-deeplab-v3-in-torchvision/95808/2
class ImmutableOutputModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        x = self.model(x)

        if isinstance(x, dict):
            x_named_tuple = namedtuple('ModelEndpoints', sorted(x.keys()))
            x = x_named_tuple(**x)
        elif isinstance(x, list):
            x = tuple(x)

        return x
