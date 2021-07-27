# For PPM.initialized_ppms https://stackoverflow.com/a/33533514
from __future__ import annotations

import logging
import re
import struct
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import inkid.ops


class PPM:
    initialized_ppms: Dict[str, PPM] = dict()

    def __init__(self, path: str, lazy_load: bool = False):
        self._path = path

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

        width, height, dim, ordered, val_type, version = [None] * 6

        data = inkid.ops.get_raw_data_from_file_or_url(filename)
        while True:
            line = data.readline().decode('utf-8')
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
        """Read the PPM file data and store the it in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3].

        """
        logging.info(f'Loading PPM data for {self._path} with width {self.width}, '
                     f'height {self.height}, dim {self._dim}...')

        self._data = np.empty((self.height, self.width, self._dim))

        data = inkid.ops.get_raw_data_from_file_or_url(self._path)
        header_terminator_re = re.compile('^<>$')
        while True:
            line = data.readline().decode('utf-8')
            if header_terminator_re.match(line):
                break

        for y in tqdm(range(self.height)):
            for x in range(self.width):
                for idx in range(self._dim):
                    # Only works if dim == 6: (x, y, z, n_x, n_y, n_z)
                    if self._dim == 6:
                        self._data[y, x, idx] = struct.unpack('d', data.read(8))[0]
        print()

    def get_point_with_normal(self, ppm_x, ppm_y):
        self.ensure_loaded()
        return self._data[ppm_y][ppm_x]

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
