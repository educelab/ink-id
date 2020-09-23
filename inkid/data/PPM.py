import logging
import os
import re
import struct
from enum import Enum, auto

import numpy as np
from PIL import Image
import progressbar
from skimage import color


class ColorSpace(Enum):
    RGB = auto()
    HSV = auto()
    LAB = auto()


class PPM:
    def __init__(self, path, volume=None, mask_path=None, ink_label_path=None,
                 rgb_label_path=None, invert_normal=False):
        self._path = path
        self._volume = volume

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

        if rgb_label_path is not None:
            self._rgb_label_path = rgb_label_path
        else:
            default_rgb_label_path = ppm_path_stem + '_reference.png'
            if os.path.isfile(default_rgb_label_path):
                self._rgb_label_path = default_rgb_label_path
            else:
                self._rgb_label_path = None

        self._ink_label = None
        if self._ink_label_path is not None:
            self._ink_label = np.asarray(Image.open(self._ink_label_path).convert('LA'), np.uint16)

        self._rgb_label = None
        if self._rgb_label_path is not None:
            self._rgb_label = np.asarray(Image.open(self._rgb_label_path).convert('RGB'), np.uint8)

        self._invert_normal = invert_normal
        if self._invert_normal:
            logging.info('Normals are being inverted for this PPM.')

        self.process_PPM_file(self._path)

        self._ink_classes_prediction_image = np.zeros((self._height, self._width), np.uint16)
        self._rgb_values_prediction_image = np.zeros((self._height, self._width, 3), np.uint8)

    @staticmethod
    def parse_PPM_header(filename):
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

    def process_PPM_file(self, filename):
        """Read a PPM file and store the data in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3].

        """
        header = PPM.parse_PPM_header(filename)
        self._width = header['width']
        self._height = header['height']
        self._dim = header['dim']
        self._ordered = header['ordered']
        self._type = header['type']
        self._version = header['version']

        logging.info(
            'Processing PPM data for {} with width {}, height {}, dim {}... '.format(
                self._path, self._width, self._height, self._dim
            )
        )

        self._data = np.empty((self._height, self._width, self._dim))

        with open(filename, 'rb') as f:
            header_terminator_re = re.compile('^<>$')
            while True:
                line = f.readline().decode('utf-8')
                if header_terminator_re.match(line):
                    break

            bar = progressbar.ProgressBar()
            for y in bar(range(self._height)):
                for x in range(self._width):
                    for idx in range(self._dim):
                        # This only works if we assume dimension 6
                        # PPMs with (x, y, z, n_x, n_y, n_z)
                        if self._dim == 6:
                            if idx in [3, 4, 5] and self._invert_normal:
                                self._data[y, x, idx] = -1 * struct.unpack('d', f.read(8))[0]
                            else:
                                self._data[y, x, idx] = struct.unpack('d', f.read(8))[0]
        print()

    def get_default_bounds(self):
        """Return the full bounds of the PPM in (x0, y0, x1, y1) format."""
        return 0, 0, self._width, self._height

    def is_on_surface(self, x, y, r=1):
        """Return whether a point is on the surface mask.

        Check a point and a square of radius r around it, and return
        False if any of those points are not on the surface
        mask. Return True otherwise.

        """
        square = self._mask[y-r:y+r+1, x-r:x+r+1]
        return np.size(square) > 0 and np.min(square) != 0

    def is_ink(self, x, y):
        assert self._ink_label is not None
        return self._ink_label[y, x] != 0

    def get_point_with_normal(self, ppm_x, ppm_y):
        return self._data[ppm_y][ppm_x]

    def point_to_ink_classes_label(self, point, shape):
        assert self._ink_label is not None
        x, y = point
        label = np.stack((np.ones(shape), np.zeros(shape))).astype(np.float32)  # Create array of no-ink labels
        y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._ink_label.shape[0] and 0 <= x_s < self._ink_label.shape[1]:
                if self._ink_label[y_s, x_s] != 0:
                    label[:, y_idx, x_idx] = [0.0, 1.0]  # Mark this "ink"
        return label

    def point_to_color_values_label(self, point, shape, color_space: ColorSpace = ColorSpace.RGB):
        assert self._rgb_label is not None
        x, y = point
        label = np.zeros((3,) + shape).astype(np.float32)
        y_d, x_d = np.array(shape) // 2  # Calculate distance from center to edges of square we are sampling
        # Iterate over label indices
        for idx, _ in np.ndenumerate(label):
            _, y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._rgb_label.shape[0] and 0 <= x_s < self._rgb_label.shape[1]:
                rgb = self._rgb_label[y_s, x_s]
                if color_space is ColorSpace.RGB:
                    label[:, y_idx, x_idx] = rgb
                elif color_space is ColorSpace.HSV:
                    label[:, y_idx, x_idx] = color.rgb2hsv(rgb)
                elif color_space is ColorSpace.LAB:
                    label[:, y_idx, x_idx] = color.rgb2lab(rgb)
        return label

    def point_to_voxel_vector(self, point, length_in_each_direction,
                              out_of_bounds=None):
        ppm_x, ppm_y = point
        x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
        return self._volume.get_voxel_vector(
            (x, y, z),
            (n_x, n_y, n_z),
            length_in_each_direction,
            out_of_bounds
        )

    def point_to_subvolume(self, point, subvolume_shape,
                           out_of_bounds=None, move_along_normal=None,
                           jitter_max=None, augment_subvolume=None,
                           method=None, normalize=None,
                           pad_to_shape=None,
                           fft=None, dwt=None, dwt_channel_subbands=None, model_3d_to_2d=None):
        ppm_x, ppm_y = point
        x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
        square_corners = None
        if model_3d_to_2d:
            square_corners = []
            y_d, x_d = np.array([subvolume_shape[1], subvolume_shape[2]]) // 2
            if 0 <= x - x_d and x + x_d < self._width and 0 <= y - y_d and y + y_d < self._height:
                # Top left
                square_corners.append(self.get_point_with_normal(ppm_x - x_d, ppm_y - y_d)[0:3])
                # Top right
                square_corners.append(self.get_point_with_normal(ppm_x + x_d, ppm_y - y_d)[0:3])
                # Bottom left
                square_corners.append(self.get_point_with_normal(ppm_x - x_d, ppm_y + y_d)[0:3])
                # Bottom right
                square_corners.append(self.get_point_with_normal(ppm_x + x_d, ppm_y + y_d)[0:3])
        return self._volume.get_subvolume(
            (x, y, z),
            subvolume_shape,
            normal=(n_x, n_y, n_z),
            out_of_bounds=out_of_bounds,
            move_along_normal=move_along_normal,
            jitter_max=jitter_max,
            augment_subvolume=augment_subvolume,
            method=method,
            normalize=normalize,
            pad_to_shape=pad_to_shape,
            square_corners=square_corners,
            fft=fft,
            dwt=dwt,
            dwt_channel_subbands=dwt_channel_subbands,
        )

    def reconstruct_predicted_ink_classes(self, class_probabilities, ppm_xy):
        assert len(ppm_xy) == 2
        x, y = ppm_xy
        value = class_probabilities[1] * np.iinfo(np.uint16).max  # Convert ink class probability to image intensity
        y_d, x_d = np.array(value.shape) // 2  # Calculate distance from center to edges of square we are writing
        # Iterate over label indices
        for idx, v in np.ndenumerate(value):
            y_idx, x_idx = idx
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._ink_classes_prediction_image.shape[0] \
                    and 0 <= x_s < self._ink_classes_prediction_image.shape[1]:
                self._ink_classes_prediction_image[y_s, x_s] = v

    def reconstruct_predicted_color(self, value, ppm_xy, color_space: ColorSpace = ColorSpace.RGB):
        assert len(ppm_xy) == 2
        x, y = ppm_xy
        y_d, x_d = np.array(value.shape)[1:] // 2  # Calculate distance from center to edges of square we are writing
        # Iterate over label indices
        for idx in np.ndindex(value.shape[1:]):
            y_idx, x_idx = idx
            v = value[:, y_idx, x_idx]
            y_s = y - y_d + y_idx  # Sample point is center minus distance (half edge length) plus label index
            x_s = x - x_d + x_idx
            # Bounds check to make sure inside PPM
            if 0 <= y_s < self._rgb_values_prediction_image.shape[0] \
                    and 0 <= x_s < self._rgb_values_prediction_image.shape[1]:
                if color_space is ColorSpace.HSV:
                    v = color.hsv2rgb(v)
                elif color_space is ColorSpace.LAB:
                    v = color.lab2rgb(v)
                # Restrict value to uint8 range
                self._rgb_values_prediction_image[y_s, x_s] = np.clip(v, 0, np.iinfo(np.uint8).max)

    def reset_predictions(self):
        if self._ink_label is not None:
            self._ink_classes_prediction_image = np.zeros((self._height, self._width), np.uint16)
        if self._rgb_label is not None:
            self._rgb_values_prediction_image = np.zeros((self._height, self._width, 3), np.uint8)

    def save_predictions(self, directory, iteration):
        if not os.path.exists(directory):
            os.makedirs(directory)

        im = None
        if self._ink_classes_prediction_image.any():
            im = Image.fromarray(self._ink_classes_prediction_image)
        elif self._rgb_values_prediction_image.any():
            im = Image.fromarray(self._rgb_values_prediction_image)
        if im is not None:
            im.save(
                os.path.join(
                    directory,
                    '{}_prediction_{}.png'.format(
                        os.path.splitext(os.path.basename(self._path))[0],
                        iteration,
                    ),
                ),
            )

    def scale_down_by(self, scale_factor):
        self._width //= scale_factor
        self._height //= scale_factor

        new_data = np.empty((self._height, self._width, self._dim))

        logging.info('Downscaling PPM by factor of {} on all axes...'.format(scale_factor))
        bar = progressbar.ProgressBar()
        for y in bar(range(self._height)):
            for x in range(self._width):
                for idx in range(self._dim):
                    new_data[y, x, idx] = self._data[y * scale_factor, x * scale_factor, idx]

        self._data = new_data

    def write(self, filename):
        with open(filename, 'wb') as f:
            logging.info('Writing PPM to file {}...'.format(filename))
            f.write('width: {}\n'.format(self._width).encode('utf-8'))
            f.write('height: {}\n'.format(self._height).encode('utf-8'))
            f.write('dim: {}\n'.format(self._dim).encode('utf-8'))
            f.write('ordered: {}\n'.format('true' if self._ordered else 'false').encode('utf-8'))
            f.write('type: double\n'.encode('utf-8'))
            f.write('version: {}\n'.format(self._version).encode('utf-8'))
            f.write('<>\n'.encode('utf-8'))
            bar = progressbar.ProgressBar()
            for y in bar(range(self._height)):
                for x in range(self._width):
                    for idx in range(self._dim):
                        f.write(struct.pack('d', self._data[y, x, idx]))
