# For PPM.initialized_ppms https://stackoverflow.com/a/33533514
from __future__ import annotations

import logging
import os
import re
import struct
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm


class PPM:
    initialized_ppms: Dict[str, PPM] = dict()

    def __init__(self, path: str, lazy_load: bool = False):
        self._path = path
        ppm_path_stem, _ = os.path.splitext(self._path)

        header = PPM.parse_ppm_header(path)
        self.width: int = header['width']
        self.height: int = header['height']
        self._dim: int = header['dim']
        self._ordered: bool = header['ordered']
        self._type: str = header['type']
        self._version: str = header['version']

        self._data: Optional[np.typing.ArrayLike] = None

        logging.info(f'Initialized PPM for {self._path} with width {self.width}, '
                     f'height {self.height}, dim {self._dim}')

        if not lazy_load:
            self.ensure_loaded()

    def is_loaded(self):
        return self._data is not None

    def ensure_loaded(self):
        if not self.is_loaded():
            self.load_ppm_data()

    @classmethod
    def from_path(cls, path: str) -> PPM:
        if path in cls.initialized_ppms:
            return cls.initialized_ppms[path]
        cls.initialized_ppms[path] = PPM(path)
        return cls.initialized_ppms[path]

    @staticmethod
    def parse_ppm_header(filename):
        comments_re = re.compile('^#')
        width_re = re.compile('^width')
        height_re = re.compile('^height')
        dim_re = re.compile('^dim')
        ordered_re = re.compile('^ordered')
        type_re = re.compile('^type')
        version_re = re.compile('^version')
        header_terminator_re = re.compile('^<>$')

        with open(filename, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if comments_re.match(line):
                    pass
                elif width_re.match(line):
                    width = int(line.split(': ')[1])
                elif height_re.match(line):
                    height = int(line.split(': ')[1])
                elif dim_re.match(line):
                    dim = int(line.split(': ')[1])
                elif ordered_re.match(line):
                    ordered = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    val_type = line.split(': ')[1].strip()
                    assert val_type in ['double']
                elif version_re.match(line):
                    version = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    logging.warning('PPM header contains unknown line: {}'.format(line.strip()))

        return {
            'width': width,
            'height': height,
            'dim': dim,
            'ordered': ordered,
            'type': val_type,
            'version': version
        }

    def load_ppm_data(self):
        """Read a PPM file and store the data in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3]. TODO update

        """
        logging.info(f'Loading PPM data for {self._path} with width {self.width}, '
                     f'height {self.height}, dim {self._dim}...')

        self._data = np.empty((self.height, self.width, self._dim))

        with open(self._path, 'rb') as f:
            header_terminator_re = re.compile('^<>$')
            while True:
                line = f.readline().decode('utf-8')
                if header_terminator_re.match(line):
                    break

            for y in tqdm(range(self.height)):
                for x in range(self.width):
                    for idx in range(self._dim):
                        # Only works if dim == 6: (x, y, z, n_x, n_y, n_z)
                        if self._dim == 6:
                            self._data[y, x, idx] = struct.unpack('d', f.read(8))[0]
        print()

    def get_point_with_normal(self, ppm_x, ppm_y):
        self.ensure_loaded()
        return self._data[ppm_y][ppm_x]

    # TODO reimplement model_3d_to_2d with new dataset thing. make sure square corners use shape_microns
    # def point_to_subvolume(self, point, subvolume_shape_voxels, subvolume_shape_microns,
    #                        out_of_bounds=None, move_along_normal=None,
    #                        jitter_max=None, augment_subvolume=None,
    #                        method=None, normalize=None,
    #                        model_3d_to_2d=None):
    #     ppm_x, ppm_y = point
    #     x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
    #     square_corners = None
    #     if model_3d_to_2d:
    #         square_corners = []
    #         y_d, x_d = np.array([subvolume_shape_voxels[1], subvolume_shape_voxels[2]]) // 2
    #         if 0 <= x - x_d and x + x_d < self._width and 0 <= y - y_d and y + y_d < self._height:
    #             # Top left
    #             square_corners.append(self.get_point_with_normal(ppm_x - x_d, ppm_y - y_d)[0:3])
    #             # Top right
    #             square_corners.append(self.get_point_with_normal(ppm_x + x_d, ppm_y - y_d)[0:3])
    #             # Bottom left
    #             square_corners.append(self.get_point_with_normal(ppm_x - x_d, ppm_y + y_d)[0:3])
    #             # Bottom right
    #             square_corners.append(self.get_point_with_normal(ppm_x + x_d, ppm_y + y_d)[0:3])
    #     return self._volume.get_subvolume(
    #         (x, y, z),
    #         subvolume_shape_voxels,
    #         subvolume_shape_microns,
    #         normal=(n_x, n_y, n_z),
    #         out_of_bounds=out_of_bounds,
    #         move_along_normal=move_along_normal,
    #         jitter_max=jitter_max,
    #         augment_subvolume=augment_subvolume,
    #         method=method,
    #         normalize=normalize,
    #         square_corners=square_corners,
    #     )

    def scale_down_by(self, scale_factor):
        self.ensure_loaded()

        self.width //= scale_factor
        self.height //= scale_factor

        new_data = np.empty((self.height, self.width, self._dim))

        logging.info('Downscaling PPM by factor of {} on all axes...'.format(scale_factor))
        for y in tqdm(range(self.height)):
            for x in range(self.width):
                for idx in range(self._dim):
                    new_data[y, x, idx] = self._data[y * scale_factor, x * scale_factor, idx]

        self._data = new_data

    def write(self, filename):
        self.ensure_loaded()

        with open(filename, 'wb') as f:
            logging.info('Writing PPM to file {}...'.format(filename))
            f.write('width: {}\n'.format(self.width).encode('utf-8'))
            f.write('height: {}\n'.format(self.height).encode('utf-8'))
            f.write('dim: {}\n'.format(self._dim).encode('utf-8'))
            f.write('ordered: {}\n'.format('true' if self._ordered else 'false').encode('utf-8'))
            f.write('type: double\n'.encode('utf-8'))
            f.write('version: {}\n'.format(self._version).encode('utf-8'))
            f.write('<>\n'.encode('utf-8'))
            for y in tqdm(range(self.height)):
                for x in range(self.width):
                    for idx in range(self._dim):
                        f.write(struct.pack('d', self._data[y, x, idx]))
