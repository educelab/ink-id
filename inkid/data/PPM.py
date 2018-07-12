import os
import re
import struct

import numpy as np
from PIL import Image
import progressbar


class PPM:
    def __init__(self, path, volume, mask_path, ink_label_path,
                 rgb_label_path, invert_normal):
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
            self._ink_label = np.asarray(Image.open(self._ink_label_path), np.uint16)

        self._rgb_label = None
        if self._rgb_label_path is not None:
            self._rgb_label = np.asarray(Image.open(self._rgb_label_path).convert('RGB'), np.uint8)

        self._invert_normal = False
        if invert_normal is not None:
            self._invert_normal = invert_normal
        if self._invert_normal:
            print('Normals are being inverted for this PPM.')

        self.process_PPM_file(self._path)

        self._ink_classes_prediction_image = np.zeros((self._height, self._width), np.uint16)
        self._rgb_values_prediction_image = np.zeros((self._height, self._width, 3), np.uint8)

    @staticmethod
    def parse_PPM_header(filename):
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
                    width = int(line.split(': ')[1])
                elif height_re.match(line):
                    height = int(line.split(': ')[1])
                elif dim_re.match(line):
                    dim = int(line.split(': ')[1])
                elif ordering_re.match(line):
                    ordering = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    val_type = line.split(': ')[1].strip()
                    assert val_type in ['double']
                elif version_re.match(line):
                    version = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    print('Warning: PPM header contains unknown line: {}'.format(line.strip()))

        return {
            'width': width,
            'height': height,
            'dim': dim,
            'ordering': ordering,
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
        self._ordering = header['ordering']
        self._type = header['type']
        self._version = header['version']

        print(
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

    def point_to_rgb_values_label(self, point):
        assert self._rgb_label is not None
        x, y = point
        label = self._rgb_label[y, x]
        assert len(label) == 3
        return np.asarray(label, np.float32)

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
                           pad_to_shape=None):
        ppm_x, ppm_y = point
        x, y, z, n_x, n_y, n_z = self.get_point_with_normal(ppm_x, ppm_y)
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
        )

    def reconstruct_predicted_ink_classes(self, class_probabilities, ppm_xy):
        intensity = class_probabilities[1] * np.iinfo(np.uint16).max
        self.reconstruct_prediction_value(intensity, ppm_xy)

    def reconstruct_prediction_value(self, value, ppm_xy, square_r=2):
        assert len(ppm_xy) == 2
        x, y = ppm_xy
        self._ink_classes_prediction_image[y-square_r:y+square_r, x-square_r:x+square_r] = value

    def reconstruct_predicted_rgb(self, rgb, ppm_xy, square_r=2):
        assert len(ppm_xy) == 2
        assert len(rgb) == 3
        x, y = ppm_xy
        # Restrict value to uint8 range
        rgb = [max(min(np.iinfo(np.uint8).max, int(val)), 0) for val in rgb]
        self._rgb_values_prediction_image[y-square_r:y+square_r, x-square_r:x+square_r] = rgb

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
                    '{}_prediction_{}.tif'.format(
                        os.path.splitext(os.path.basename(self._path))[0],
                        iteration,
                    ),
                ),
            )
