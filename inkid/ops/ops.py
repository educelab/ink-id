"""Miscellaneous operations used in ink-id."""

import itertools
from io import BytesIO
import json
import logging
import requests
from urllib.parse import urlsplit, urlunsplit

import math
import os
import subprocess
from xml.dom.minidom import parseString

from dicttoxml import dicttoxml
import numpy as np
from PIL import Image
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


def visualize_batch(xb, yb):
    xb, yb = xb.numpy(), yb.numpy()
    xb = np.max(np.concatenate(np.squeeze(xb), 1), 0) / 255
    yb = np.concatenate(yb, 1)[1] * 255
    img = np.concatenate((xb, yb), 1)
    Image.fromarray(img).show()


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

    Given a volume as a np.array and a directory name, save the volume
    as a stack of .tif images in that directory, with filenames
    starting at 0 and going up to the z height of the volume.

    """
    os.makedirs(dirname, exist_ok=True)
    for z in range(volume.shape[0]):
        image = volume[z, :, :]
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


def perform_validation(model, dataloader, metrics, device, label_type):
    """Run the validation process using a model and dataloader, and return the results of all metrics."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        metric_results = {metric: [] for metric in metrics}
        for _, xb, yb in tqdm(dataloader):
            pred = model(xb.to(device))
            yb = yb.to(device)
            if label_type == 'ink_classes':
                _, yb = yb.max(1)  # Argmax
            for metric, fn in metrics.items():
                metric_results[metric].append(fn(pred, yb))
    model.train()
    return metric_results


def generate_prediction_images(dataloader, model, output_size, label_type, device, predictions_dir, suffix,
                               prediction_averaging):
    """Helper function to generate a prediction image given a model and dataloader, and save it to a file."""
    model.eval()  # Turn off training mode for batch norm and dropout purposes
    with torch.no_grad():
        for batch_metadata, batch_features in tqdm(dataloader):
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
                pred = model(aug_pxb.to(device))
                if label_type == 'ink_classes':
                    pred = F.softmax(pred, dim=1)
                pred = pred.cpu()
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
            source_paths, xs, ys = batch_metadata
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
    file_path = os.path.join(os.path.dirname(inkid.__file__), 'schemas', schema_name + '.schema.json')
    with open(file_path, 'r') as f:
        return json.load(f)


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
