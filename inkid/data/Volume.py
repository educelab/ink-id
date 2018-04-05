import math
import os

from mathutils import Matrix, Vector
import numpy as np
from PIL import Image
import progressbar


class Volume:
    def __init__(self, slices_path):
        slices_abs_path = os.path.abspath(slices_path)
        slice_files = []
        for root, dirs, files in os.walk(slices_abs_path):
            for filename in files:
                slice_files.append(os.path.join(root, filename))
        slice_files.sort()
        
        self._data = []
        print('Loading volume slices from {}...'.format(slices_abs_path))
        bar = progressbar.ProgressBar()
        for slice_file in bar(slice_files):
            self._data.append(np.array(Image.open(slice_file)))
        self._data = np.array(self._data)
        print('Loaded volume {} with shape {}'.format(slices_abs_path, self._data.shape))


        # x = self.get_subvolume_using_normal(
        #     (100, 100, 100),
        #     (48, 96, 96),
        #     (0.3, 0.5, 1)
        # )
        # print(x)

    def intensity_at_xyz(self, x, y, z):
        """Get the intensity value at a voxel position."""
        return self._data[int(z), int(y), int(x)]


    def interpolate_at_xyz(self, x, y, z):
        """Get the intensity value at a subvoxel position.

        Values are trilinearly interpolated.

        https://en.wikipedia.org/wiki/Trilinear_interpolation

        """
        dx, x0 = math.modf(x)
        dy, y0 = math.modf(y)
        dz, z0 = math.modf(z)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        c00 = self.intensity_at_xyz(x0, y0, z0) * (1 - dx) + self.intensity_at_xyz(x1, y0, z0) * dx
        c10 = self.intensity_at_xyz(x0, y1, z0) * (1 - dx) + self.intensity_at_xyz(x1, y0, z0) * dx
        c01 = self.intensity_at_xyz(x0, y0, z1) * (1 - dx) + self.intensity_at_xyz(x1, y0, z1) * dx
        c11 = self.intensity_at_xyz(x0, y1, z1) * (1 - dx) + self.intensity_at_xyz(x1, y1, z1) * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        c = c0 * (1 - dz) + c1 * dz
        return c


    def get_subvolume(self, center_xyz, shape_zyx,
                      x_vec=(1, 0, 0),
                      y_vec=(0, 1, 0),
                      z_vec=(0, 0, 1)):
        assert(len(center_xyz) == 3)
        assert(len(shape_zyx) == 3)
        assert(len(x_vec) == 3)
        assert(len(y_vec) == 3)
        assert(len(z_vec) == 3)

        center_xyz = np.array(center_xyz)
        x_vec = np.array(x_vec)
        y_vec = np.array(y_vec)
        z_vec = np.array(z_vec)

        subvolume = np.zeros(shape_zyx)

        # Iterate over the subvolume space
        bar = progressbar.ProgressBar()
        for z in bar(range(shape_zyx[0])):
            for y in range(shape_zyx[1]):
                for x in range(shape_zyx[2]):
                    x_offset = -1 * (shape_zyx[2] - 1) / 2.0 + x
                    y_offset = -1 * (shape_zyx[1] - 1) / 2.0 + y
                    z_offset = -1 * (shape_zyx[0] - 1) / 2.0 + z

                    volume_point = center_xyz \
                                   + x_offset * x_vec \
                                   + y_offset * y_vec \
                                   + z_offset * z_vec
                    subvolume[z, y, x] = self.interpolate_at_xyz(
                        volume_point[0],
                        volume_point[1],
                        volume_point[2],
                    )

        return subvolume


    def get_subvolume_using_normal(self, center_xyz, shape_zyx, normal_vec=(0, 0, 1)):
        x_vec = Vector([1, 0, 0])
        y_vec = Vector([0, 1, 0])
        z_vec = Vector([0, 0, 1])
        normal_vec = Vector(normal_vec)

        quaternion = z_vec.rotation_difference(normal_vec)

        x_vec = (quaternion * x_vec).normalized()
        y_vec = (quaternion * y_vec).normalized()
        z_vec = (quaternion * z_vec).normalized()

        return self.get_subvolume(center_xyz, shape_zyx, x_vec, y_vec, z_vec)

        
