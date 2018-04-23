"""Miscellaneous operations used in ink-id."""

import inspect
import json
import os

from jsmin import jsmin
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
    os.makedirs(dirname)
    for z in range(volume.shape[0]):
        image = volume[z, :, :]
        image = image.astype(np.uint16)
        image = Image.fromarray(image)
        image.save(os.path.join(dirname, str(z) + '.tif'))


def load_default_parameters():
    """Return the default network parameters for ink-id.

    Find the directory that the inkid package is loaded from, and then
    return the network parameters in parameters.json.

    https://stackoverflow.com/questions/247770/retrieving-python-module-path

    """
    return load_parameters_from_json(
        os.path.join(os.path.dirname(inspect.getfile(inkid)), 'parameters.json')
    )


def load_parameters_from_json(filename):
    """Return a dict of the parameters stored in a JSON file.

    Given a filename to a .json, remove the comments from that file
    and return a Python dictionary built from the JSON.

    """
    with open(filename, 'r') as f:
        # minify to remove comments
        minified = jsmin(str(f.read()))
        return json.loads(minified)['parameters']
