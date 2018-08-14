# cython: boundscheck=False
"""
Define the Volume class to represent volumetric data.
"""

cimport libc.math as math
import os
import random
import sys

import mathutils
import numpy as np
cimport numpy as cnp
from PIL import Image
import progressbar


cdef BasisVectors get_component_vectors_from_normal(Float3 n):
    """Get a subvolume oriented based on a surface normal vector.

    Calculate the rotation needed to align the z axis of the
    subvolume with the surface normal vector, and then apply that
    rotation to all three axes of the subvolume in order to get
    the vectors for the subvolume axes in the volume space.
    
    See:
    https://docs.blender.org/api/blender_python_api_current/mathutils.html

    """
    x_vec = mathutils.Vector([1, 0, 0])
    y_vec = mathutils.Vector([0, 1, 0])
    z_vec = mathutils.Vector([0, 0, 1])
    normal = mathutils.Vector([n.x, n.y, n.z]).normalized()

    quaternion = z_vec.rotation_difference(normal)

    x_vec.rotate(quaternion)
    y_vec.rotate(quaternion)
    z_vec.rotate(quaternion)

    cdef BasisVectors basis
    
    basis.x.x = x_vec[0]
    basis.x.y = x_vec[1]
    basis.x.z = x_vec[2]
    
    basis.y.x = y_vec[0]
    basis.y.y = y_vec[1]
    basis.y.z = y_vec[2]

    basis.z.x = z_vec[0]
    basis.z.y = z_vec[1]
    basis.z.z = z_vec[2]

    return basis

cdef class Volume:
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
    cdef unsigned short [:, :, :] _data_view
    cdef int shape_z, shape_y, shape_x
    
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

        data = []
        print('Loading volume slices from {}...'.format(slices_abs_path))
        bar = progressbar.ProgressBar()
        for slice_file in bar(slice_files):
            data.append(np.array(Image.open(slice_file)))
        print()
        data = np.array(data, dtype=np.uint16)
        self._data_view = data
        print('Loaded volume {} with shape (z, y, x) = {}'.format(
            slices_abs_path,
            data.shape
        ))
        self.shape_z = data.shape[0]
        self.shape_y = data.shape[1]
        self.shape_x = data.shape[2]

    def normalize(self):
        # Don't have a good answer for this right now since it would make everything floats
        # data = np.asarray(self._data_view, dtype=np.float32)
        # data = data - data.mean()
        # data = data / data.std()
        # data = np.asarray(data, dtype=np.uint16)
        # self._data_view = data
        pass

    cdef unsigned short intensity_at(self, int x, int y, int z) nogil:
        """Get the intensity value at a voxel position."""
        if x >= self.shape_x or y >= self.shape_y or z >= self.shape_z:
            return 0
        return self._data_view[z, y, x]

    cdef unsigned short interpolate_at(self, float x, float y, float z) nogil:
        """Get the intensity value at a subvoxel position.

        Values are trilinearly interpolated.

        https://en.wikipedia.org/wiki/Trilinear_interpolation

        Potential speed improvement:
        https://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

        """
        if x >= self.shape_x or y >= self.shape_y or z >= self.shape_z:
            return 0

        cdef double dx, dy, dz, x0d, y0d, z0d
        cdef int x0, y0, z0, x1, y1, z1
        cdef double c00, c10, c01, c11, c0, c1
        cdef unsigned short c
        dx = math.modf(x, &x0d)
        dy = math.modf(y, &y0d)
        dz = math.modf(z, &z0d)

        x0 = <int>(x0d)
        y0 = <int>(y0d)
        z0 = <int>(z0d)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        c00 = self.intensity_at(x0, y0, z0) * (1 - dx) + self.intensity_at(x1, y0, z0) * dx
        c10 = self.intensity_at(x0, y1, z0) * (1 - dx) + self.intensity_at(x1, y0, z0) * dx
        c01 = self.intensity_at(x0, y0, z1) * (1 - dx) + self.intensity_at(x1, y0, z1) * dx
        c11 = self.intensity_at(x0, y1, z1) * (1 - dx) + self.intensity_at(x1, y1, z1) * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        c = <unsigned short>(c0 * (1 - dz) + c1 * dz)
        return c

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
                voxel_vector.append(self.interpolate_at(x, y, z))
            return voxel_vector

        except IndexError:
            if out_of_bounds == 'all_zeros':
                return [0.0] * (length_in_each_direction * 2 + 1)
            else:
                raise IndexError

    cpdef get_subvolume_snap_to_axis_aligned(self,
                                             center,
                                             shape,
                                             normal,
                                             out_of_bounds):
        """Snap to and get the closest axis-aligned subvolume.

        Snap the normal vector to the closest axis vector (including
        in the negative directions) and get a subvolume as if that
        were the original normal vector.

        Implemented for speed, not accuracy (instead of full
        interpolation for example, which would be the opposite).

        """
        cdef int x, y, z, z_r, y_r, x_r, strongest_normal_axis
        cdef cnp.ndarray[cnp.npy_uint16, ndim=3] subvolume

        strongest_normal_axis = np.argmax(np.absolute(normal))
        x = int(round(center[0]))
        y = int(round(center[1]))
        z = int(round(center[2]))
        z_r = shape[0] // 2
        y_r = shape[1] // 2
        x_r = shape[2] // 2

        # z in subvolume space is along x in volume space
        if strongest_normal_axis == 0:
            subvolume = np.asarray(self._data_view[z-y_r:z+y_r, y-x_r:y+x_r, x-z_r:x+z_r])
            subvolume = np.rot90(subvolume, axes=(2, 0))

        # z in subvolume space is along y in volume space
        elif strongest_normal_axis == 1:
            subvolume = np.asarray(self._data_view[z-x_r:z+x_r, y-z_r:y+z_r, x-y_r:x+y_r])
            subvolume = np.rot90(subvolume, axes=(1, 0))

        # z in subvolume space is along z in volume space
        elif strongest_normal_axis == 2:
            subvolume = np.asarray(self._data_view[z-z_r:z+z_r, y-y_r:y+y_r, x-x_r:x+x_r])

        # If the normal was pointed along a negative axis, flip the
        # subvolume over
        if normal[strongest_normal_axis] < 0:
            subvolume = np.rot90(subvolume, k=2, axes=(0, 1))

        if out_of_bounds == 'all_zeros':
            if subvolume.shape[0] != tuple(shape)[0] \
               or subvolume.shape[1] != tuple(shape)[1] \
                or subvolume.shape[2] != tuple(shape)[2]:
                subvolume = np.zeros(shape, dtype=np.uint16)
        elif out_of_bounds == 'partial_zeros':
            pass
        elif out_of_bounds == 'index_error':
            pass
        else:
            raise ValueError('Out of bounds method not recognized.')

        return subvolume

    cdef void interpolated_with_basis_vectors(self, Float3 center, Int3 shape, BasisVectors basis, uint16[:,:,:] array) nogil:
        cdef int x, y, z, x_offset, y_offset, z_offset
        cdef Float3 volume_point
        cdef Int3 offset
        
        for z in range(shape.z):
            for y in range(shape.y):
                for x in range(shape.x):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    offset.x = <int>((-1 * (shape.x - 1) / 2.0 + x) + 0.5)
                    offset.y = <int>((-1 * (shape.y - 1) / 2.0 + y) + 0.5)
                    offset.z = <int>((-1 * (shape.z - 1) / 2.0 + z) + 0.5)

                    # Calculate the corresponding position in the
                    # volume.
                    volume_point.x = center.x
                    volume_point.y = center.y
                    volume_point.z = center.z
                    
                    volume_point.x += offset.x * basis.x.x
                    volume_point.y += offset.x * basis.x.y
                    volume_point.z += offset.x * basis.x.z

                    volume_point.x += offset.y * basis.y.x
                    volume_point.y += offset.y * basis.y.y
                    volume_point.z += offset.y * basis.y.z

                    volume_point.x += offset.z * basis.z.x
                    volume_point.y += offset.z * basis.z.y
                    volume_point.z += offset.z * basis.z.z
                    
                    array[z, y, x] = self.interpolate_at(
                        volume_point.x,
                        volume_point.y,
                        volume_point.z
                    )

    cdef void nearest_neighbor_with_basis_vectors(self, Float3 center, Int3 shape, BasisVectors basis, uint16[:,:,:] array) nogil:
        cdef int x, y, z, x_offset, y_offset, z_offset
        cdef Float3 volume_point
        cdef Int3 offset
        
        for z in range(shape.z):
            for y in range(shape.y):
                for x in range(shape.x):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    offset.x = <int>((-1 * (shape.x - 1) / 2.0 + x) + 0.5)
                    offset.y = <int>((-1 * (shape.y - 1) / 2.0 + y) + 0.5)
                    offset.z = <int>((-1 * (shape.z - 1) / 2.0 + z) + 0.5)

                    # Calculate the corresponding position in the
                    # volume.
                    volume_point.x = center.x
                    volume_point.y = center.y
                    volume_point.z = center.z
                    
                    volume_point.x += offset.x * basis.x.x
                    volume_point.y += offset.x * basis.x.y
                    volume_point.z += offset.x * basis.x.z

                    volume_point.x += offset.y * basis.y.x
                    volume_point.y += offset.y * basis.y.y
                    volume_point.z += offset.y * basis.y.z

                    volume_point.x += offset.z * basis.z.x
                    volume_point.y += offset.z * basis.z.y
                    volume_point.z += offset.z * basis.z.z
                    
                    array[z, y, x] = self.intensity_at(
                        <int>(volume_point.x + 0.5),
                        <int>(volume_point.y + 0.5),
                        <int>(volume_point.z + 0.5)
                    )

    cpdef get_subvolume_nearest_neighbor(self, center, shape, normal,
                                         out_of_bounds):
        cdef BasisVectors basis
        cdef Float3 n, c
        cdef Int3 s

        n.x = normal[0]
        n.y = normal[1]
        n.z = normal[2]

        c.x = center[0]
        c.y = center[1]
        c.z = center[2]

        s.z = shape[0]
        s.y = shape[1]
        s.x = shape[2]

        basis = get_component_vectors_from_normal(n)

        subvolume = np.zeros(shape, dtype=np.uint16)

        self.nearest_neighbor_with_basis_vectors(c, s, basis, subvolume)

        return subvolume

    cpdef get_subvolume_interpolated(self, center, shape, normal,
                                     out_of_bounds):
        cdef BasisVectors basis
        cdef Float3 n, c
        cdef Int3 s

        n.x = normal[0]
        n.y = normal[1]
        n.z = normal[2]

        c.x = center[0]
        c.y = center[1]
        c.z = center[2]

        s.z = shape[0]
        s.y = shape[1]
        s.x = shape[2]

        basis = get_component_vectors_from_normal(n)

        subvolume = np.zeros(shape, dtype=np.uint16)

        self.interpolated_with_basis_vectors(c, s, basis, subvolume)

        return subvolume


    def get_subvolume(self, center, shape, normal, out_of_bounds,
                      move_along_normal, jitter_max,
                      augment_subvolume, method, normalize, pad_to_shape):
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

        if normalize:
            subvolume = np.asarray(subvolume, dtype=np.float32)
            subvolume = subvolume - subvolume.mean()
            subvolume = subvolume / subvolume.std()

        if pad_to_shape is not None:
            assert pad_to_shape[0] >= shape[0]
            assert pad_to_shape[1] >= shape[1]
            assert pad_to_shape[2] >= shape[2]
            z_d = pad_to_shape[0] - shape[0]
            y_d = pad_to_shape[1] - shape[1]
            x_d = pad_to_shape[2] - shape[2]
            subvolume = np.pad(
                subvolume,
                (
                    (z_d // 2, z_d - (z_d // 2)),
                    (y_d // 2, y_d - (y_d // 2)),
                    (x_d // 2, x_d - (x_d // 2)),
                ),
                'constant'
            )
            assert subvolume.shape == tuple(pad_to_shape)
        else:
            assert subvolume.shape == tuple(shape)

        return subvolume
