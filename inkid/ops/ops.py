"""Miscellaneous operations used in ink-id."""

import inspect
import math
import os
import subprocess

import numpy as np
from PIL import Image

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


def default_arguments_file():
    """Return the default arguments file.

    https://stackoverflow.com/questions/247770/retrieving-python-module-path

    """
    return os.path.join(os.path.dirname(inspect.getfile(inkid)), 'default_arguments.txt')


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
    if rclone_remote is not None:
        folders = []
        path = os.path.abspath(output_path)
        while True:
            path, folder = os.path.split(path)
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()

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
            print(' '.join(command))
            subprocess.call(command)
