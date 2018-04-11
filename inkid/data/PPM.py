import os
import re
import struct
import sys

import numpy as np
from PIL import Image
import progressbar

class PPM:
    def __init__(self, path, volume, mask_path, ink_label_path):
        self._path = path
        self._volume = volume
        self._predicted_ink_classes = None

        ppm_path_stem, _ = os.path.splitext(self._path)

        if mask_path is not None:
            self._mask_path = mask_path
        else:
            default_mask_path = ppm_path_stem + '_mask.png'
            if os.path.isfile(default_mask_path):
                self._mask_path = default_mask_path
            else:
                self._mask_path = None

        self._mask = None
        if self._mask_path is not None:
            self._mask = np.array(Image.open(self._mask_path))

        if ink_label_path is not None:
            self._ink_label_path = ink_label_path
        else:
            default_ink_label_path = ppm_path_stem + '_ink-label.png'
            if os.path.isfile(default_ink_label_path):
                self._ink_label_path = default_ink_label_path
            else:
                self._ink_label_path = None

        self._ink_label = None
        if self._ink_label_path is not None:
            self._ink_label = np.asarray(Image.open(self._ink_label_path), np.int16)

        self.process_PPM_file(self._path)

    def process_PPM_file(self, filename):
        """Read a PPM file and store the data in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3].

        """
        comments_re = re.compile('^#')
        width_re = re.compile('^width')
        height_re = re.compile('^height')
        dim_re = re.compile('^dim')
        ordering_re = re.compile('^ordered')
        type_re = re.compile('^type')
        version_re = re.compile('^version')
        header_terminator_re = re.compile('^<>$')

        with open(filename, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if comments_re.match(line):
                    pass
                elif width_re.match(line):
                    self._width = int(line.split(': ')[1])
                elif height_re.match(line):
                    self._height = int(line.split(': ')[1])
                elif dim_re.match(line):
                    self._dim = int(line.split(': ')[1])
                elif ordering_re.match(line):
                    self._ordering = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    self._type = line.split(': ')[1].strip()
                    # assert self._type in ['double', 'int', 'float']
                    assert self._type in ['double']  # TODO support other data formats
                elif version_re.match(line):
                    self._version = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    print('Warning: PPM header contains unknown line: {}'.format(line.strip()))
            print(
                'Processing PPM data for {} with width {}, height {}, dim {}... '.format(
                    self._path, self._width, self._height, self._dim
                )
            )

            self._data = np.empty((self._height, self._width, self._dim))

            bar = progressbar.ProgressBar()
            for y in bar(range(self._height)):
                for x in range(self._width):
                    for idx in range(self._dim):
                        self._data[y, x, idx] = struct.unpack('d', f.read(8))[0]
            print()

    def get_default_bounds(self):
        """Return the full bounds of the PPM in (x0, y0, x1, y1) format."""
        return (0, 0, self._width, self._height)

    def is_on_surface(self, x, y, r=1):
        """Return whether a point is on the surface mask.

        Check a point and a square of radius r around it, and return
        False if any of those points are not on the surface
        mask. Return True otherwise.

        """
        square = self._mask[y-r:y+r+1, x-r:x+r+1]
        return np.size(square) > 0 and np.min(square) != 0

    def get_point_with_normal(self, ppm_x, ppm_y):
        return self._data[ppm_y][ppm_x]

    def point_to_ink_classes_label(self, point):
        assert self._ink_label is not None
        x, y = point
        label = self._ink_label[y, x]
        if label != 0:
            return np.asarray([0.0, 1.0], np.float32)
        else:
            return np.asarray([1.0, 0.0], np.float32)

    def point_to_subvolume(self, point, subvolume_shape):
        ppm_x, ppm_y = point
        x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
        return self._volume.get_subvolume_using_normal(
            (x, y, z),
            subvolume_shape,
            normal_vec=(n_x, n_y, n_z),
        )
        
