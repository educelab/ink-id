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

    def __init__(self, path):
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

        # self._ink_classes_prediction_image = np.zeros((self._height, self._width), np.uint16)
        # self._ink_classes_prediction_image_written_to = False
        # self._rgb_values_prediction_image = np.zeros((self._height, self._width, 3), np.uint8)
        # self._rgb_values_prediction_image_written_to = False

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

    # def point_to_ink_classes_label(self, point, shape):
    #     assert self._ink_label is not None
    #     x, y = point
    #     label = np.stack((np.ones(shape), np.zeros(shape))).astype(np.float32)  # Create array of no-ink labels
    #     y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
    #     # Iterate over label indices
    #     for idx, _ in np.ndenumerate(label):
    #         _, y_idx, x_idx = idx
    #         y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
    #         x_s = x - x_d + x_idx
    #         # Bounds check to make sure inside PPM
    #         if 0 <= y_s < self._ink_label.shape[0] and 0 <= x_s < self._ink_label.shape[1]:
    #             if self._ink_label[y_s, x_s] != 0:
    #                 label[:, y_idx, x_idx] = [0.0, 1.0]  # Mark this "ink"
    #     return label
    #
    # def point_to_rgb_values_label(self, point, shape):
    #     assert self._rgb_label is not None
    #     x, y = point
    #     label = np.zeros((3,) + shape).astype(np.float32)
    #     y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
    #     # Iterate over label indices
    #     for idx, _ in np.ndenumerate(label):
    #         _, y_idx, x_idx = idx
    #         y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
    #         x_s = x - x_d + x_idx
    #         # Bounds check to make sure inside PPM
    #         if 0 <= y_s < self._rgb_label.shape[0] and 0 <= x_s < self._rgb_label.shape[1]:
    #             label[:, y_idx, x_idx] = self._rgb_label[y_s, x_s]
    #     return label
    #
    # def point_to_voxel_vector(self, point, length_in_each_direction,
    #                           out_of_bounds=None):
    #     ppm_x, ppm_y = point
    #     x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
    #     return self._volume.get_voxel_vector(
    #         (x, y, z),
    #         (n_x, n_y, n_z),
    #         length_in_each_direction,
    #         out_of_bounds
    #     )
    #
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
    #
    # def reconstruct_predicted_ink_classes(self, class_probabilities, ppm_xy):
    #     assert len(ppm_xy) == 2
    #     x, y = ppm_xy
    #     value = class_probabilities[1] * np.iinfo(np.uint16).max  # Convert ink class probability to image intensity
    #     y_d, x_d = np.array(value.shape) // 2  # Calculate distance from center to edges of square we are writing
    #     # Iterate over label indices
    #     for idx, v in np.ndenumerate(value):
    #         y_idx, x_idx = idx
    #         y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
    #         x_s = x - x_d + x_idx
    #         # Bounds check to make sure inside PPM
    #         if 0 <= y_s < self._ink_classes_prediction_image.shape[0] \
    #                 and 0 <= x_s < self._ink_classes_prediction_image.shape[1]:
    #             self._ink_classes_prediction_image[y_s, x_s] = v
    #     self._ink_classes_prediction_image_written_to = True
    #
    # def reconstruct_predicted_rgb(self, rgb, ppm_xy):
    #     assert len(ppm_xy) == 2
    #     x, y = ppm_xy
    #     value = np.clip(rgb, 0, np.iinfo(np.uint8).max)  # Restrict value to uint8 range
    #     y_d, x_d = np.array(value.shape)[1:] // 2  # Calculate distance from center to edges of square we are writing
    #     # Iterate over label indices
    #     for idx in np.ndindex(value.shape[1:]):
    #         y_idx, x_idx = idx
    #         v = value[:, y_idx, x_idx]
    #         y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
    #         x_s = x - x_d + x_idx
    #         # Bounds check to make sure inside PPM
    #         if 0 <= y_s < self._rgb_values_prediction_image.shape[0] \
    #                 and 0 <= x_s < self._rgb_values_prediction_image.shape[1]:
    #             self._rgb_values_prediction_image[y_s, x_s] = v
    #     self._rgb_values_prediction_image_written_to = True
    #
    # def reset_predictions(self):
    #     if self._ink_label is not None:
    #         self._ink_classes_prediction_image = np.zeros((self._height, self._width), np.uint16)
    #         self._ink_classes_prediction_image_written_to = False
    #     if self._rgb_label is not None:
    #         self._rgb_values_prediction_image = np.zeros((self._height, self._width, 3), np.uint8)
    #         self._rgb_values_prediction_image_written_to = False
    #
    # def save_predictions(self, directory, suffix):
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #
    #     im = None
    #     if self._ink_classes_prediction_image_written_to:
    #         im = Image.fromarray(self._ink_classes_prediction_image)
    #     elif self._rgb_values_prediction_image_written_to:
    #         im = Image.fromarray(self._rgb_values_prediction_image)
    #     if im is not None:
    #         im.save(
    #             os.path.join(
    #                 directory,
    #                 '{}_prediction_{}.png'.format(
    #                     self._name,
    #                     suffix,
    #                 ),
    #             ),
    #         )

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
