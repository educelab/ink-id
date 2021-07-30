# cython: language_level=3, boundscheck=False
"""
Define the Volume class to represent volumetric data.
"""

import json
cimport libc.math as math
import logging
import os
import random
import sys
from typing import Dict

import numpy as np
cimport numpy as cnp
from PIL import Image
from tqdm import tqdm

cdef float norm(Float3 vec):
    return (vec.x**2 + vec.y**2 + vec.z**2)**(1./2)

cdef Float3 normalize(Float3 vec):
    cdef float n
    cdef Float3 normalized
    n = norm(vec)
    normalized.x = vec.x / n
    normalized.y = vec.y / n
    normalized.z = vec.z / n
    return normalized

cdef Float3 cross(Float3 vec_a, Float3 vec_b):
    cdef Float3 res
    res.x = vec_a.y * vec_b.z - vec_a.z * vec_b.y
    res.y = vec_a.z * vec_b.x - vec_a.x * vec_b.z
    res.z = vec_a.x * vec_b.y - vec_a.y * vec_b.x
    return res

cdef float dot_v3v3(Float3 a, Float3 b):
    return a.x * b.x + a.y * b.y + a.z * b.z

cdef float saasin(float fac):
    if fac <= -1.0:
        return -math.pi / 2.0
    elif fac >= 1.0:
        return float(math.pi / 2.0)
    else:
        return math.asin(fac)

cdef float len_v3v3(Float3 a, Float3 b):
    cdef Float3 d
    d = sub_v3_v3v3(b, a)
    return len_v3(d)

cdef float len_v3(Float3 a):
    return math.sqrt(dot_v3v3(a, a))

cdef Float3 sub_v3_v3v3(Float3 a, Float3 b):
    cdef Float3 r
    r.x = a.x - b.x
    r.y = a.y - b.y
    r.z = a.z - b.z
    return r

cdef negate_v3(Float3 v):
    cdef Float3 r
    r.x = -v.x
    r.y = -v.y
    r.z = -v.z
    return r

cdef float angle_normalized_v3v3(Float3 v1, Float3 v2):
    cdef Float3 v2_n
    # This is the same as acos(dot_v3v3(v1, v2)), but more accurate
    if dot_v3v3(v1, v2) >= 0.0:
        return 2.0 * saasin(len_v3v3(v1, v2) / 2.0)
    else:
        v2_n = negate_v3(v2)
        return math.pi - 2.0 * saasin(len_v3v3(v1, v2_n) / 2.0)

cdef Float3 mul_v3_v3(Float3 a, float f):
    cdef Float3 r
    r.x = a.x * f
    r.y = a.y * f
    r.z = a.z * f
    return r

cdef Float4 axis_angle_normalized_to_quat(Float3 axis, float angle):
    cdef float phi, si, co
    cdef Float4 q
    cdef Float3 t
    phi = 0.5 * angle
    si = math.sinf(phi)
    co = math.cosf(phi)
    q.a = co
    t = mul_v3_v3(axis, si)
    q.b = t.x
    q.c = t.y
    q.d = t.z
    return q

cdef Float4 unit_qt():
    cdef Float4 q
    q.a = 1.0
    q.b = 0.0
    q.c = 0.0
    q.d = 0.0
    return q

cdef int axis_dominant_v3_single(Float3 v):
    cdef float x, y, z
    x = abs(v.x)
    y = abs(v.y)
    z = abs(v.z)
    return (0 if x > z else 2) if (x > y) else (1 if y > z else 2)

# Calculates p - a perpendicular vector to v
cdef Float3 ortho_v3_v3(Float3 v):
    cdef Float3 p
    cdef int axis
    axis = axis_dominant_v3_single(v)
    if axis == 0:
        p.x = -v.y - v.z
        p.y = v.x
        p.z = v.x
    elif axis == 1:
        p.x = v.y
        p.y = -v.x - v.z
        p.z = v.y
    elif axis == 2:
        p.x = v.z
        p.y = v.z
        p.z = -v.x - v.y
    return p

cdef Float4 axis_angle_to_quat(Float3 axis, float angle):
    cdef Float4 nor
    nor = # TODO LEFT OFF

# Note: expects vectors to be normalized
cdef Float4 rotation_between_vecs_to_quat(Float3 v1, Float3 v2):
    cdef Float3 axis
    cdef float angle
    cdef Float4 q
    axis = cross(v1, v2)
    if norm(axis) > sys.float_info.epsilon:
        angle = angle_normalized_v3v3(v1, v2)
        return axis_angle_normalized_to_quat(axis, angle)
    else:
        # Degenerate case
        if dot_v3v3(v1, v2) > 0.0:
            # Same vectors, zero rotation
            return unit_qt()
        else:
            # Colinear but opposed vectors, 180 rotation
            axis = ortho_v3_v3(v1)
            return axis_angle_to_quat(axis, math.pi)


cdef Float4 vector_rotation_difference(Float3 vec_a, Float3 vec_b):
    cdef Float4 quat

    vec_a = normalize(vec_a)
    vec_b = normalize(vec_b)

    quat = rotation_between_vecs_to_quat(vec_a, vec_b)

    return quat


cpdef BasisVectors get_basis_from_square(square_corners):
    top_left, top_right, bottom_left, bottom_right = np.array(square_corners)

    x_vec = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    y_vec = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0
    z_vec = np.cross(x_vec, y_vec)

    x_vec = normalize(x_vec)
    y_vec = normalize(y_vec)
    z_vec = normalize(z_vec)

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


cpdef BasisVectors get_component_vectors_from_normal(Float3 n):
    """Get a subvolume oriented based on a surface normal vector.

    Calculate the rotation needed to align the z axis of the
    subvolume with the surface normal vector, and then apply that
    rotation to all three axes of the subvolume in order to get
    the vectors for the subvolume axes in the volume space.
    
    See:
    https://docs.blender.org/api/blender_python_api_current/mathutils.html

    """
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    z_vec = np.array([0, 0, 1])
    normal = np.array([n.x, n.y, n.z])
    normal = normalize(normal)

    quaternion = vector_rotation_difference(z_vec, normal)

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
    cdef dict _metadata
    cdef float _voxelsize

    initialized_volumes: Dict[str, Volume] = dict()

    @classmethod
    def from_path(cls, path: str) -> Volume:
        if path in cls.initialized_volumes:
            return cls.initialized_volumes[path]
        cls.initialized_volumes[path] = Volume(path)
        return cls.initialized_volumes[path]
    
    def __init__(self, slices_path):
        """Initialize a volume using a path to the slices directory.

        Get the absolute path and filename for each slice in the given
        directory. Load them into a contiguous volume in memory,
        represented as a numpy array and indexed self._data[z, y, x].

        Ignores hidden files in that directory, but will get all other
        files, so it must be a directory with only image files.

        """
        # Load metadata
        self._metadata = dict()
        metadata_filename = os.path.join(slices_path, 'meta.json')
        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(f'No volume meta.json file found in {slices_path}')
        else:
            with open(metadata_filename) as f:
                self._metadata = json.loads(f.read())
        self._voxelsize = self._metadata['voxelsize']
        self.shape_z = self._metadata['slices']
        self.shape_y = self._metadata['height']
        self.shape_x = self._metadata['width']

        # Get list of slice image filenames
        slice_files = []
        for root, dirs, files in os.walk(slices_path):
            for filename in files:
                # Make sure it is not a hidden file and it's a
                # .tif. In the future we might add other formats.
                if filename[0] != '.' and os.path.splitext(filename)[1] == '.tif':
                    slice_files.append(os.path.join(root, filename))
        slice_files.sort()
        assert len(slice_files) == self.shape_z

        # Load slice images into volume
        data, w, h, d = None, 0, 0, 0
        logging.info('Loading volume slices from {}...'.format(slices_path))
        for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
            if data is None:
                w, h = Image.open(slice_file).size
                d = len(slice_files)
                data = np.empty((d, h, w), dtype=np.uint16)
            data[slice_i, :, :] = np.array(Image.open(slice_file), dtype=np.uint16).copy()
        print()
        self._data_view = data
        logging.info('Loaded volume {} with shape (z, y, x) = {}'.format(
            slices_path,
            data.shape
        ))

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

        x0 = <int> x0d
        y0 = <int> y0d
        z0 = <int> z0d

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

        normal = normalize(normal)

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

    cdef void interpolated_with_basis_vectors(self, Float3 center, Int3 shape_voxels, Float3 shape_microns, BasisVectors basis, uint16[:,:,:] array) nogil:
        cdef int x, y, z, x_offset, y_offset, z_offset
        cdef Float3 volume_point, subvolume_voxel_size_microns, subvolume_voxel_size_volume_voxel_size_ratio
        cdef Int3 offset

        subvolume_voxel_size_microns.x = shape_microns.x / shape_voxels.x
        subvolume_voxel_size_microns.y = shape_microns.y / shape_voxels.y
        subvolume_voxel_size_microns.z = shape_microns.z / shape_voxels.z

        subvolume_voxel_size_volume_voxel_size_ratio.x = subvolume_voxel_size_microns.x / self._voxelsize
        subvolume_voxel_size_volume_voxel_size_ratio.y = subvolume_voxel_size_microns.y / self._voxelsize
        subvolume_voxel_size_volume_voxel_size_ratio.z = subvolume_voxel_size_microns.z / self._voxelsize
        
        for z in range(shape_voxels.z):
            for y in range(shape_voxels.y):
                for x in range(shape_voxels.x):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    offset.x = <int>((-1 * (shape_voxels.x - 1) / 2.0 + x) + 0.5)
                    offset.y = <int>((-1 * (shape_voxels.y - 1) / 2.0 + y) + 0.5)
                    offset.z = <int>((-1 * (shape_voxels.z - 1) / 2.0 + z) + 0.5)

                    # Calculate the corresponding position in the
                    # volume.
                    volume_point.x = center.x
                    volume_point.y = center.y
                    volume_point.z = center.z
                    
                    volume_point.x += offset.x * basis.x.x * subvolume_voxel_size_volume_voxel_size_ratio.x
                    volume_point.y += offset.x * basis.x.y * subvolume_voxel_size_volume_voxel_size_ratio.x
                    volume_point.z += offset.x * basis.x.z * subvolume_voxel_size_volume_voxel_size_ratio.x

                    volume_point.x += offset.y * basis.y.x * subvolume_voxel_size_volume_voxel_size_ratio.y
                    volume_point.y += offset.y * basis.y.y * subvolume_voxel_size_volume_voxel_size_ratio.y
                    volume_point.z += offset.y * basis.y.z * subvolume_voxel_size_volume_voxel_size_ratio.y

                    volume_point.x += offset.z * basis.z.x * subvolume_voxel_size_volume_voxel_size_ratio.z
                    volume_point.y += offset.z * basis.z.y * subvolume_voxel_size_volume_voxel_size_ratio.z
                    volume_point.z += offset.z * basis.z.z * subvolume_voxel_size_volume_voxel_size_ratio.z
                    
                    array[z, y, x] = self.interpolate_at(
                        volume_point.x,
                        volume_point.y,
                        volume_point.z
                    )

    cdef void nearest_neighbor_with_basis_vectors(self, Float3 center, Int3 shape_voxels, Float3 shape_microns, BasisVectors basis, uint16[:,:,:] array) nogil:
        cdef int x, y, z, x_offset, y_offset, z_offset
        cdef Float3 volume_point, subvolume_voxel_size_microns, subvolume_voxel_size_volume_voxel_size_ratio
        cdef Int3 offset

        subvolume_voxel_size_microns.x = shape_microns.x / shape_voxels.x
        subvolume_voxel_size_microns.y = shape_microns.y / shape_voxels.y
        subvolume_voxel_size_microns.z = shape_microns.z / shape_voxels.z

        subvolume_voxel_size_volume_voxel_size_ratio.x = subvolume_voxel_size_microns.x / self._voxelsize
        subvolume_voxel_size_volume_voxel_size_ratio.y = subvolume_voxel_size_microns.y / self._voxelsize
        subvolume_voxel_size_volume_voxel_size_ratio.z = subvolume_voxel_size_microns.z / self._voxelsize
        
        for z in range(shape_voxels.z):
            for y in range(shape_voxels.y):
                for x in range(shape_voxels.x):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    offset.x = <int>((-1 * (shape_voxels.x - 1) / 2.0 + x) + 0.5)
                    offset.y = <int>((-1 * (shape_voxels.y - 1) / 2.0 + y) + 0.5)
                    offset.z = <int>((-1 * (shape_voxels.z - 1) / 2.0 + z) + 0.5)

                    # Calculate the corresponding position in the
                    # volume.
                    volume_point.x = center.x
                    volume_point.y = center.y
                    volume_point.z = center.z

                    volume_point.x += offset.x * basis.x.x * subvolume_voxel_size_volume_voxel_size_ratio.x
                    volume_point.y += offset.x * basis.x.y * subvolume_voxel_size_volume_voxel_size_ratio.x
                    volume_point.z += offset.x * basis.x.z * subvolume_voxel_size_volume_voxel_size_ratio.x

                    volume_point.x += offset.y * basis.y.x * subvolume_voxel_size_volume_voxel_size_ratio.y
                    volume_point.y += offset.y * basis.y.y * subvolume_voxel_size_volume_voxel_size_ratio.y
                    volume_point.z += offset.y * basis.y.z * subvolume_voxel_size_volume_voxel_size_ratio.y

                    volume_point.x += offset.z * basis.z.x * subvolume_voxel_size_volume_voxel_size_ratio.z
                    volume_point.y += offset.z * basis.z.y * subvolume_voxel_size_volume_voxel_size_ratio.z
                    volume_point.z += offset.z * basis.z.z * subvolume_voxel_size_volume_voxel_size_ratio.z
                    
                    array[z, y, x] = self.intensity_at(
                        <int>(volume_point.x + 0.5),
                        <int>(volume_point.y + 0.5),
                        <int>(volume_point.z + 0.5)
                    )


    def get_subvolume(self, center, shape_voxels, shape_microns, normal,
                      out_of_bounds, move_along_normal, jitter_max,
                      augment_subvolume, method, normalize=False, square_corners=None):
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
            shape_voxels: The desired shape of the subvolume in voxels.
            shape_microns: The desired spatial extent of the sampled subvolume in microns.
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
            A np.float32 array of the requested shape.

        """
        assert len(center) == 3
        assert len(shape_voxels) == 3

        # If shape_microns not specified, fall back to the old method
        # (spatial extent based only on number of voxels and not voxel size)
        if shape_microns is None:
            shape_microns = list(np.array(shape_voxels) * self._voxelsize)

        assert len(shape_microns) == 3

        # TODO if square_corners is not None and not empty, modify basis vectors before calling (and center and normal?)
        # TODO if empty return zeros?

        if normal is None:
            normal = np.array([0, 0, 1])
        else:
            normal = normalize(normal)

        if out_of_bounds is None:
            out_of_bounds = 'all_zeros'
        assert out_of_bounds in ['all_zeros', 'partial_zeros', 'index_error']

        if move_along_normal is None:
            move_along_normal = 0

        if jitter_max is None:
            jitter_max = 0

        if augment_subvolume is None:
            augment_subvolume = False

        center = np.array(center)
        center += (move_along_normal + random.randint(-jitter_max, jitter_max)) * normal

        cdef BasisVectors basis
        cdef Float3 n, c, s_m
        cdef Int3 s_v

        n.x = normal[0]
        n.y = normal[1]
        n.z = normal[2]

        c.x = center[0]
        c.y = center[1]
        c.z = center[2]

        s_v.z = shape_voxels[0]
        s_v.y = shape_voxels[1]
        s_v.x = shape_voxels[2]

        s_m.z = shape_microns[0]
        s_m.y = shape_microns[1]
        s_m.x = shape_microns[2]

        subvolume = np.zeros(shape_voxels, dtype=np.uint16)

        if square_corners is None or len(square_corners) > 0:
            if square_corners is None:
                basis = get_component_vectors_from_normal(n)
            else:
                basis = get_basis_from_square(square_corners)
            if method is None:
                method = 'nearest_neighbor'
            assert method in ['interpolated', 'nearest_neighbor']
            if method == 'interpolated':
                self.interpolated_with_basis_vectors(c, s_v, s_m, basis, subvolume)
            elif method == 'nearest_neighbor':
                self.nearest_neighbor_with_basis_vectors(c, s_v, s_m, basis, subvolume)

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

        assert subvolume.shape == tuple(shape_voxels)

        # Convert to float
        subvolume = np.asarray(subvolume, np.float32)
        # Add singleton dimension for number of channels
        subvolume = np.expand_dims(subvolume, 0)

        return subvolume
