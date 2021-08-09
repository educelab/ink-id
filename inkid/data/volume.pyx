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

cdef float normalize_v3_v3_length(float r[3], const float a[3], const float unit_length):
    cdef float d = dot_v3v3(a, a)

    # a larger value causes normalize errors in a scaled down models with camera extreme close
    if d > 1.0e-35:
        d = math.sqrt(d)
        mul_v3_v3fl(r, a, unit_length / d)
    else:
        zero_v3(r)
        d = 0.0
    return d

cdef float normalize_v3_v3(float r[3], const float a[3]):
    return normalize_v3_v3_length(r, a, 1.0)

cdef float normalize_v3(float n[3]):
    return normalize_v3_v3(n, n)

cdef void mul_v3_v3fl(float r[3], const float a[3], float f):
    r[0] = a[0] * f
    r[1] = a[1] * f
    r[2] = a[2] * f

cdef void zero_v3(float r[3]):
    r[0] = 0.0
    r[1] = 0.0
    r[2] = 0.0

cdef void cross_v3_v3v3(float r[3], const float a[3], const float b[3]):
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]

cdef float dot_v3v3(const float a[3], const float b[3]):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

cdef float saasin(const float fac):
    if fac <= -1.0:
        return <float>-math.M_PI / 2.0
    elif fac >= 1.0:
        return <float>math.M_PI / 2.0
    else:
        return math.asin(fac)

cdef float len_v3v3(const float a[3], const float b[3]):
    cdef float d[3]

    sub_v3_v3v3(d, b, a)
    return len_v3(d)

cdef float len_v3(const float a[3]):
    return math.sqrt(dot_v3v3(a, a))

cdef void sub_v3_v3v3(float r[3], const float a[3], const float b[3]):
    r[0] = a[0] - b[0]
    r[1] = a[1] - b[1]
    r[2] = a[2] - b[2]

cdef void negate_v3_v3(float r[3], const float a[3]):
    r[0] = -a[0]
    r[1] = -a[1]
    r[2] = -a[2]

cdef float angle_normalized_v3v3(const float v1[3], const float v2[3]):
    cdef float v2_n[3]
    # This is the same as acos(dot_v3v3(v1, v2)), but more accurate
    if dot_v3v3(v1, v2) >= 0.0:
        return 2.0 * saasin(len_v3v3(v1, v2) / 2.0)
    else:
        negate_v3_v3(v2_n, v2)
        return <float>math.M_PI - 2.0 * saasin(len_v3v3(v1, v2_n) / 2.0)

cdef void axis_angle_normalized_to_quat(float q[4], const float axis[3], const float angle):
    cdef float phi = 0.5 * angle
    cdef float si = math.sin(phi)
    cdef float co = math.cos(phi)
    q[0] = co
    mul_v3_v3fl(q + 1, axis, si)

cdef void unit_qt(float q[4]):
    q[0] = 1.0
    q[1] = q[2] = q[3] = 0.0

cdef int axis_dominant_v3_single(const float vec[3]):
    cdef float x = abs(vec[0])
    cdef float y = abs(vec[1])
    cdef float z = abs(vec[2])
    return (0 if x > z else 2) if (x > y) else (1 if y > z else 2)

# Calculates p - a perpendicular vector to v
cdef void ortho_v3_v3(float out[3], const float v[3]):
    cdef int axis = axis_dominant_v3_single(v)

    if axis == 0:
        out[0] = -v[1] - v[2]
        out[1] = v[0]
        out[2] = v[0]
    elif axis == 1:
        out[0] = v[1]
        out[1] = -v[0] - v[2]
        out[2] = v[1]
    elif axis == 2:
        out[0] = v[2]
        out[1] = v[2]
        out[2] = -v[0] - v[1]

cdef void axis_angle_to_quat(float q[4], const float axis[3], const float angle):
    cdef float nor[3]
    if normalize_v3_v3(nor, axis) != 0.0:
        axis_angle_normalized_to_quat(q, nor, angle)
    else:
        unit_qt(q)

# Note: expects vectors to be normalized
cdef void rotation_between_vecs_to_quat(float q[4], const float v1[3], const float v2[3]):
    cdef float angle
    cdef float axis[3]

    cross_v3_v3v3(axis, v1, v2)

    if normalize_v3(axis) > sys.float_info.epsilon:
        angle = angle_normalized_v3v3(v1, v2)
        axis_angle_normalized_to_quat(q, axis, angle)
    else:
        # Degenerate case
        if dot_v3v3(v1, v2) > 0.0:
            # Same vectors, zero rotation
            unit_qt(q)
        else:
            # Colinear but opposed vectors, 180 rotation
            ortho_v3_v3(axis, v1)
            axis_angle_to_quat(q, axis, <float>math.M_PI)

cdef void vector_rotation_difference(float q[4], const float vec_a_in[3], const float vec_b_in[3]):
    cdef float quat[4]
    cdef float vec_a[3]
    cdef float vec_b[3]
    copy_v3_v3(vec_a, vec_a_in)
    copy_v3_v3(vec_b, vec_b_in)

    normalize_v3(vec_a)
    normalize_v3(vec_b)

    rotation_between_vecs_to_quat(quat, vec_a, vec_b)

cdef void copy_qt_qt(float q1[4], const float q2[4]):
    q1[0] = q2[0]
    q1[1] = q2[1]
    q1[2] = q2[2]
    q1[3] = q2[3]

cdef float dot_qtqt(const float q1[4], const float q2[4]):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

cdef void mul_qt_fl(float q[4], const float f):
    q[0] *= f
    q[1] *= f
    q[2] *= f
    q[3] *= f

cdef float normalize_qt(float q[4]):
    cdef float length = math.sqrt(dot_qtqt(q, q))
    if length != 0.0:
        mul_qt_fl(q, 1.0 / length)
    else:
        q[1] = 1.0
        q[0] = q[2] = q[3] = 0.0
    return length

cdef float normalize_qt_qt(float r[4], const float q[4]):
    copy_qt_qt(r, q)
    return normalize_qt(r)

cdef void quat_to_mat3(float m[3][3], const float q[4]):
    cdef double q0, q1, q2, q3, qda, qdb, qdc, qaa, qab, qac, qbb, qbc, qcc

    q0 = math.M_SQRT2 * <double>q[0]
    q1 = math.M_SQRT2 * <double>q[1]
    q2 = math.M_SQRT2 * <double>q[2]
    q3 = math.M_SQRT2 * <double>q[3]

    qda = q0 * q1
    qdb = q0 * q2
    qdc = q0 * q3
    qaa = q1 * q1
    qab = q1 * q2
    qac = q1 * q3
    qbb = q2 * q2
    qbc = q2 * q3
    qcc = q3 * q3

    m[0][0] = <float>(1.0 - qbb - qcc)
    m[0][1] = <float>(qdc + qab)
    m[0][2] = <float>(-qdb + qac)

    m[1][0] = <float>(-qdc + qab)
    m[1][1] = <float>(1.0 - qaa - qcc)
    m[1][2] = <float>(qda + qbc)

    m[2][0] = <float>(qdb + qac)
    m[2][1] = <float>(-qda + qbc)
    m[2][2] = <float>(1.0 - qaa - qbb)

cdef void quaternion_to_rmat(float rmat[3][3], const float quaternion[4]):
    cdef float tquat[4]
    normalize_qt_qt(tquat, quaternion)
    quat_to_mat3(rmat, tquat)

cdef void copy_v3_v3(float r[3], const float a[3]):
    r[0] = a[0]
    r[1] = a[1]
    r[2] = a[2]

cdef void mul_v3_m3v3(float r[3], const float m[3][3], const float a[3]):
    cdef float t[3]
    copy_v3_v3(t, a)

    r[0] = m[0][0] * t[0] + m[1][0] * t[1] + m[2][0] * t[2]
    r[1] = m[0][1] * t[0] + m[1][1] * t[1] + m[2][1] * t[2]
    r[2] = m[0][2] * t[0] + m[1][2] * t[1] + m[2][2] * t[2]

cdef void mul_m3_v3(const float m[3][3], float r[3]):
    cdef float a[3]
    copy_v3_v3(a, r)
    mul_v3_m3v3(r, m, a)

cdef void rotate(float out[3], const float vec[3], const float quaternion[4]):
    cdef float rmat[3][3]

    quaternion_to_rmat(rmat, quaternion)

    copy_v3_v3(out, vec)

    mul_m3_v3(rmat, out)

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

cpdef BasisVectors get_basis_from_square(square_corners):
    cdef BasisVectors basis
    cdef float x_vec[3]
    cdef float y_vec[3]
    cdef float z_vec[3]

    top_left, top_right, bottom_left, bottom_right = np.array(square_corners)

    x_vec_np = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    y_vec_np = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0

    x_vec[0] = x_vec_np[0]
    x_vec[1] = x_vec_np[1]
    x_vec[2] = x_vec_np[2]

    y_vec[0] = y_vec_np[0]
    y_vec[1] = y_vec_np[1]
    y_vec[2] = y_vec_np[2]

    cross_v3_v3v3(z_vec, x_vec, y_vec)

    normalize_v3(x_vec)
    normalize_v3(y_vec)
    normalize_v3(z_vec)

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


cpdef BasisVectors get_component_vectors_from_normal(const Float3 n):
    """Get a subvolume oriented based on a surface normal vector.

    Calculate the rotation needed to align the z axis of the
    subvolume with the surface normal vector, and then apply that
    rotation to all three axes of the subvolume in order to get
    the vectors for the subvolume axes in the volume space.
    
    See:
    https://docs.blender.org/api/blender_python_api_current/mathutils.html

    """
    cdef BasisVectors basis
    cdef float x_vec[3]
    cdef float y_vec[3]
    cdef float z_vec[3]
    cdef float normal[3]
    cdef float quaternion[4]

    zero_v3(x_vec)
    x_vec[0] = 1.0
    zero_v3(y_vec)
    y_vec[1] = 1.0
    zero_v3(z_vec)
    z_vec[2] = 1.0

    normal[0] = n.x
    normal[1] = n.y
    normal[2] = n.z
    normalize_v3(normal)

    vector_rotation_difference(quaternion, z_vec, normal)

    rotate(x_vec, x_vec, quaternion)
    rotate(y_vec, y_vec, quaternion)
    rotate(z_vec, z_vec, quaternion)

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
