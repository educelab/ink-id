"""
Define the Volume class to represent volumetric data.
"""

import math
import os
import random

import mathutils
import numpy as np
from PIL import Image
import progressbar


class Volume:
    """Represent a volume and support accesses of the volume data.

    The volume class supports the access of raw volume data, either
    directly (indexing into the 3d array) or through supporting
    functions (e.g. extracting a subvolume from an arbitrary position
    and orientation).

    Notably this class knows nothing about the PPMs or regions that
    the ink-id user might be working with, or any notion of ground
    truth, etc.

    Indexing conventions:
    - Points/coordinates are in (x, y, z)
    - 3D vectors are (x, y, z)
    - Volumes are indexed by [z, y, x]
    - Volume shapes are (z, y, x)

    """
    def __init__(self, slices_path):
        """Initialize a volume using a path to the slices directory.

        Get the absolute path and filename for each slice in the given
        directory. Load them into a contiguous volume in memory,
        represented as a numpy array and indexed self._data[z, y, x].

        Ignores hidden files in that directory, but will get all other
        files, so it must be a directory with only image files.

        """
        slices_abs_path = os.path.abspath(slices_path)
        slice_files = []
        for root, dirs, files in os.walk(slices_abs_path):
            for filename in files:
                # Make sure it is not a hidden file and it's a
                # .tif. In the future we might add other formats.
                if filename[0] != '.' and os.path.splitext(filename)[1] == '.tif':
                    slice_files.append(os.path.join(root, filename))
        slice_files.sort()

        self._data = []
        print('Loading volume slices from {}...'.format(slices_abs_path))
        bar = progressbar.ProgressBar()
        for slice_file in bar(slice_files):
            self._data.append(np.array(Image.open(slice_file)))
        print()
        self._data = np.array(self._data)
        print('Loaded volume {} with shape (z, y, x) = {}'.format(
            slices_abs_path,
            self._data.shape)
        )

    def intensity_at(self, x, y, z):
        """Get the intensity value at a voxel position."""
        return self._data[int(z), int(y), int(x)]

    def interpolate_at(self, x, y, z, return_zero_instead_of_index_error=False):
        """Get the intensity value at a subvoxel position.

        Values are trilinearly interpolated.

        https://en.wikipedia.org/wiki/Trilinear_interpolation

        Potential speed improvement:
        https://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

        """
        try:
            dx, x0 = math.modf(x)
            dy, y0 = math.modf(y)
            dz, z0 = math.modf(z)

            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            c00 = self.intensity_at(x0, y0, z0) * (1 - dx) + self.intensity_at(x1, y0, z0) * dx
            c10 = self.intensity_at(x0, y1, z0) * (1 - dx) + self.intensity_at(x1, y0, z0) * dx
            c01 = self.intensity_at(x0, y0, z1) * (1 - dx) + self.intensity_at(x1, y0, z1) * dx
            c11 = self.intensity_at(x0, y1, z1) * (1 - dx) + self.intensity_at(x1, y1, z1) * dx

            c0 = c00 * (1 - dy) + c10 * dy
            c1 = c01 * (1 - dy) + c11 * dy

            c = c0 * (1 - dz) + c1 * dz
            return c

        except IndexError:
            if return_zero_instead_of_index_error:
                return 0.0
            else:
                raise IndexError

    def get_voxel_vector(self, center, normal, length_in_each_direction, out_of_bounds):
        """Get a voxel vector from within the volume."""
        assert len(center) == 3
        assert len(normal) == 3

        normal = mathutils.Vector(normal).normalized()

        if out_of_bounds is None:
            out_of_bounds = 'all_zeros'
        assert out_of_bounds in ['all_zeros', 'partial_zeros', 'index_error']

        center = np.array(center)

        voxel_vector = []

        try:
            for i in range(-length_in_each_direction, length_in_each_direction + 1):
                x, y, z = center + i * normal
                voxel_vector.append(self.interpolate_at(x, y, z, out_of_bounds == 'partial_zeros'))
            return voxel_vector

        except IndexError:
            if out_of_bounds == 'all_zeros':
                return [0.0] * (length_in_each_direction * 2 + 1)
            else:
                raise IndexError

    def get_subvolume(self, center, shape, normal, out_of_bounds,
                      move_along_normal, jitter_max,
                      augment_subvolume, method):
        """Get a subvolume from a center point and normal vector.

        At the time of writing, this function very closely resembles
        but is not identical to a similar one in the
        VolumeCartographer core. Volume::getVoxelNeighborsInterpolated
        is centered on a position and goes out an integer number of
        points from that position in each axis, so the side lengths
        are always odd and there is always a voxel at the center of
        the returned volume. Not the case here, since we tend to want
        even-sided subvolumes for machine learning. See
        getVoxelNeighborsInterpolated in
        https://code.vis.uky.edu/seales-research/volume-cartographer/blob/develop/core/include/vc/core/types/Volume.hpp.

        Args:
            center: The starting center point of the subvolume.
            shape: The desired shape of the subvolume.
            normal: The normal vector at the center point.
            out_of_bounds: String indicating what to do if the requested
                subvolume does not fit entirely within the volume.
            move_along_normal: Scalar of how many units to translate
                the center point along the center vector.
            jitter_max: Jitter the center point a random amount up to
                this value in either direction along the normal vector.
            augment_subvolume: Whether or not to perform augmentation.
            method: String to indicate how to get the volume data.

        Returns:
            A numpy array of the requested shape.

        """
        assert len(center) == 3
        assert len(shape) == 3

        if normal is None:
            normal = mathutils.Vector((0, 0, 1))
        else:
            normal = mathutils.Vector(normal).normalized()

        if out_of_bounds is None:
            out_of_bounds = 'all_zeros'
        assert out_of_bounds in ['all_zeros', 'partial_zeros', 'index_error']

        if move_along_normal is None:
            move_along_normal = 0

        if jitter_max is None:
            jitter_max = 0

        if augment_subvolume is None:
            augment_subvolume = False

        if method is None:
            method = 'snap_to_axis_aligned'
        assert method in ['snap_to_axis_aligned', 'interpolated', 'nearest_neighbor']
        if method == 'snap_to_axis_aligned':
            method_fn = self.get_subvolume_snap_to_axis_aligned
        elif method == 'interpolated':
            method_fn = self.get_subvolume_interpolated
        elif method == 'nearest_neighbor':
            method_fn = self.get_subvolume_nearest_neighbor

        center = np.array(center)
        center += (move_along_normal + random.randint(-jitter_max, jitter_max)) * normal

        subvolume = method_fn(
            center,
            shape,
            normal,
            out_of_bounds,
        )

        if augment_subvolume:
            flip_direction = np.random.randint(4)
            if flip_direction == 0:
                subvolume = np.flip(subvolume, axis=1)  # Flip y
            elif flip_direction == 1:
                subvolume = np.flip(subvolume, axis=2)  # Flip x
            elif flip_direction == 2:
                subvolume = np.flip(subvolume, axis=1)  # Flip x and y
                subvolume = np.flip(subvolume, axis=2)

            rotate_direction = np.random.randint(4)
            subvolume = np.rot90(subvolume, k=rotate_direction, axes=(1, 2))

        assert subvolume.shape == tuple(shape)

        return subvolume

    def get_subvolume_snap_to_axis_aligned(self, center, shape,
                                           normal, out_of_bounds):
        """Snap to and get the closest axis-aligned subvolume.

        Snap the normal vector to the closest axis vector (including
        in the negative directions) and get a subvolume as if that
        were the original normal vector.

        Implemented for speed, not accuracy (instead of full
        interpolation for example, which would be the opposite).

        """
        strongest_normal_axis = np.argmax(np.absolute(normal))
        x, y, z = (int(round(i)) for i in center)
        z_r, y_r, x_r = (i // 2 for i in shape)

        # z in subvolume space is along x in volume space
        if strongest_normal_axis == 0:
            subvolume = self._data[z-y_r:z+y_r, y-x_r:y+x_r, x-z_r:x+z_r]
            subvolume = np.rot90(subvolume, axes=(2, 0))

        # z in subvolume space is along y in volume space
        elif strongest_normal_axis == 1:
            subvolume = self._data[z-x_r:z+x_r, y-z_r:y+z_r, x-y_r:x+y_r]
            subvolume = np.rot90(subvolume, axes=(1, 0))

        # z in subvolume space is along z in volume space
        elif strongest_normal_axis == 2:
            subvolume = self._data[z-z_r:z+z_r, y-y_r:y+y_r, x-x_r:x+x_r]

        # If the normal was pointed along a negative axis, flip the
        # subvolume over
        if normal[strongest_normal_axis] < 0:
            subvolume = np.rot90(subvolume, k=2, axes=(0, 1))

        if out_of_bounds == 'all_zeros':
            if subvolume.shape != tuple(shape):
                subvolume = np.zeros(shape)
        elif out_of_bounds == 'partial_zeros':
            pass
        elif out_of_bounds == 'index_error':
            pass
        else:
            raise ValueError('Out of bounds method not recognized.')

        return subvolume

    def get_subvolume_interpolated(self, center, shape, normal,
                                   out_of_bounds):
        # x_vec = np.array(x_vec)
        # y_vec = np.array(y_vec)
        # z_vec = np.array(z_vec)

        # subvolume = np.zeros(shape_zyx)

        # # Iterate over the subvolume space
        # for z in range(shape_zyx[0]):
        #     for y in range(shape_zyx[1]):
        #         for x in range(shape_zyx[2]):
        #             # Convert from an index relative to an origin in
        #             # the corner to a position relative to the
        #             # subvolume center (which may not correspond
        #             # exactly to one of the subvolume voxel positions
        #             # if any of the side lengths are even).
        #             x_offset = -1 * (shape_zyx[2] - 1) / 2.0 + x
        #             y_offset = -1 * (shape_zyx[1] - 1) / 2.0 + y
        #             z_offset = -1 * (shape_zyx[0] - 1) / 2.0 + z

        #             # Calculate the corresponding position in the
        #             # volume.
        #             volume_point = center_xyz \
        #                            + x_offset * x_vec \
        #                            + y_offset * y_vec \
        #                            + z_offset * z_vec
        #             try:
        #                 subvolume[z, y, x] = self.interpolate_at(
        #                     volume_point[0],
        #                     volume_point[1],
        #                     volume_point[2],
        #                 )
        #             except IndexError:
        #                 subvolume[z, y, x] = 0
        # return subvolume
        pass

    def get_subvolume_nearest_neighbor(self, center, shape, normal,
                                       out_of_bounds):
        subvolume = np.zeros(shape)
        x_vec = mathutils.Vector([1, 0, 0])
        y_vec = mathutils.Vector([0, 1, 0])
        z_vec = mathutils.Vector([0, 0, 1])
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    x_offset = int(round(-1 * (shape[2] - 1) / 2.0 + x))
                    y_offset = int(round(-1 * (shape[1] - 1) / 2.0 + y))
                    z_offset = int(round(-1 * (shape[0] - 1) / 2.0 + z))

                    # Calculate the corresponding position in the
                    # volume.
                    volume_point = center \
                                   + x_offset * x_vec \
                                   + y_offset * y_vec \
                                   + z_offset * z_vec
                    try:
                        subvolume[z, y, x] = self.intensity_at(
                            (volume_point[0]),
                            (volume_point[1]),
                            (volume_point[2])
                        )
                    except IndexError:
                        subvolume[z, y, x] = 0
        return subvolume


    # def get_subvolume_using_normal(self, center_xyz, shape_zyx, normal_vec=(0, 0, 1)):
    #     """Get a subvolume oriented based on a surface normal vector.

    #     Calculate the rotation needed to align the z axis of the
    #     subvolume with the surface normal vector, and then apply that
    #     rotation to all three axes of the subvolume in order to get
    #     the vectors for the subvolume axes in the volume space.

    #     See:
    #     https://docs.blender.org/api/blender_python_api_current/mathutils.html

    #     """
    #     x_vec = mathutils.Vector([1, 0, 0])
    #     y_vec = mathutils.Vector([0, 1, 0])
    #     z_vec = mathutils.Vector([0, 0, 1])
    #     normal_vec = mathutils.Vector(normal_vec).normalized()

    #     quaternion = z_vec.rotation_difference(normal_vec)

    #     x_vec.rotate(quaternion)
    #     y_vec.rotate(quaternion)
    #     z_vec.rotate(quaternion)

    #     return self.get_subvolume(center_xyz, shape_zyx, x_vec, y_vec, z_vec)
