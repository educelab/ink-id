"""Miscellaneous operations used in ink-id."""

import logging
import math
import os
import subprocess

import numpy as np
from PIL import Image
import torch


def add_subvolume_args(parser):
    parser.add_argument('--subvolume-method', default='nearest_neighbor',
                        help='method for sampling subvolumes', choices=['nearest_neighbor', 'interpolated'])
    parser.add_argument('--subvolume-shape-microns', metavar='um', nargs=3, type=float, default=None,
                        help='subvolume shape (microns) in (z, y, x)')
    parser.add_argument('--subvolume-shape-voxels', metavar='n', nargs=3, type=int, required=True,
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
    """Take the first n samples from a dataset to reduce the size."""
    if n_samples < len(dataset):
        dataset = torch.utils.data.random_split(dataset, [n_samples, len(dataset) - n_samples])[0]
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
