# cython: language_level=3, boundscheck=False
"""
Define the Volume class to represent volumetric data.
"""

import json
cimport libc.math as math
import logging
import os
import random

import numpy as np
cimport numpy as cnp
from PIL import Image
from tqdm import tqdm

cimport inkid.data.mathutils as mathutils

import inkid.util

cpdef norm(vec):
    vec = np.array(vec)
    return (vec[0]**2 + vec[1]**2 + vec[2]**2)**(1./2)

cpdef normalize_fl3(vec):
    vec = np.array(vec)
    n = norm(vec)
    return vec / n

cpdef BasisVectors get_basis_from_square(square_corners):
    cdef BasisVectors basis
    cdef float x_vec[3]
    cdef float y_vec[3]
    cdef float z_vec[3]
    mathutils.zero_v3(z_vec)

    top_left, top_right, bottom_left, bottom_right = np.array(square_corners)

    x_vec_np = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    y_vec_np = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0

    x_vec[0] = x_vec_np[0]
    x_vec[1] = x_vec_np[1]
    x_vec[2] = x_vec_np[2]

    y_vec[0] = y_vec_np[0]
    y_vec[1] = y_vec_np[1]
    y_vec[2] = y_vec_np[2]

    mathutils.cross_v3_v3v3(z_vec, x_vec, y_vec)

    mathutils.normalize_v3(x_vec)
    mathutils.normalize_v3(y_vec)
    mathutils.normalize_v3(z_vec)

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

    mathutils.zero_v3(x_vec)
    x_vec[0] = 1.0
    mathutils.zero_v3(y_vec)
    y_vec[1] = 1.0
    mathutils.zero_v3(z_vec)
    z_vec[2] = 1.0

    normal[0] = n.x
    normal[1] = n.y
    normal[2] = n.z
    mathutils.normalize_v3(normal)

    mathutils.vector_rotation_difference(quaternion, z_vec, normal)

    mathutils.rotate(x_vec, x_vec, quaternion)
    mathutils.rotate(y_vec, y_vec, quaternion)
    mathutils.rotate(z_vec, z_vec, quaternion)

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
    cdef const unsigned short [:, :, :] _data_view
    cdef int shape_z, shape_y, shape_x
    cdef dict _metadata
    cdef float _voxelsize_um

    initialized_volumes = dict()  # Dict[str, Volume]

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
        self._voxelsize_um = self._metadata['voxelsize']
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
                img = Image.open(slice_file)
                w, h = img.size
                img.close()
                d = len(slice_files)
                data = np.empty((d, h, w), dtype=np.uint16)
            data[slice_i, :, :] = np.array(Image.open(slice_file), dtype=np.uint16).copy()
        print()
        self._data_view = data
        logging.info('Loaded volume {} with shape (z, y, x) = {}'.format(
            slices_path,
            data.shape
        ))

    def z_slice(self, int idx):
        if 0 <= idx < self.shape_z:
            return inkid.util.uint16_to_float32_normalized_0_1(self._data_view[idx, :, :])
        else:
            return np.zeros((self.shape_y, self.shape_x), dtype=np.float32)

    def y_slice(self, int idx):
        if 0 <= idx < self.shape_y:
            return inkid.util.uint16_to_float32_normalized_0_1(self._data_view[:, idx, :])
        else:
            return np.zeros((self.shape_z, self.shape_x), dtype=np.float32)

    def x_slice(self, int idx):
        if 0 <= idx < self.shape_x:
            return inkid.util.uint16_to_float32_normalized_0_1(self._data_view[:, :, idx])
        else:
            return np.zeros((self.shape_z, self.shape_y), dtype=np.float32)

    def shape(self):
        return self.shape_z, self.shape_y, self.shape_x

    cdef unsigned short intensity_at(self, int x, int y, int z) nogil:
        """Get the intensity value at a voxel position."""
        if 0 <= x < self.shape_x and 0 <= y < self.shape_y and 0 <= z < self.shape_z:
            return self._data_view[z, y, x]
        else:
            return 0


    cdef unsigned short interpolate_at(self, float x, float y, float z) nogil:
        """Get the intensity value at a subvoxel position.

        Values are trilinearly interpolated.

        https://en.wikipedia.org/wiki/Trilinear_interpolation

        Potential speed improvement:
        https://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

        """
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

    cpdef get_subvolume_snap_to_axis_aligned(self,
                                             center,
                                             shape,
                                             normal):
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

        if subvolume.shape[0] != tuple(shape)[0] \
           or subvolume.shape[1] != tuple(shape)[1] \
            or subvolume.shape[2] != tuple(shape)[2]:
            subvolume = np.zeros(shape, dtype=np.uint16)

        return subvolume

    cdef void interpolated_with_basis_vectors(self, Float3 center, Int3 shape_voxels, Float3 shape_microns, BasisVectors basis, uint16[:,:,:] array) nogil:
        cdef int x, y, z, x_offset, y_offset, z_offset
        cdef Float3 volume_point, subvolume_voxel_size_microns, subvolume_voxel_size_volume_voxel_size_ratio
        cdef Int3 offset

        subvolume_voxel_size_microns.x = shape_microns.x / shape_voxels.x
        subvolume_voxel_size_microns.y = shape_microns.y / shape_voxels.y
        subvolume_voxel_size_microns.z = shape_microns.z / shape_voxels.z

        subvolume_voxel_size_volume_voxel_size_ratio.x = subvolume_voxel_size_microns.x / self._voxelsize_um
        subvolume_voxel_size_volume_voxel_size_ratio.y = subvolume_voxel_size_microns.y / self._voxelsize_um
        subvolume_voxel_size_volume_voxel_size_ratio.z = subvolume_voxel_size_microns.z / self._voxelsize_um
        
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

        subvolume_voxel_size_volume_voxel_size_ratio.x = subvolume_voxel_size_microns.x / self._voxelsize_um
        subvolume_voxel_size_volume_voxel_size_ratio.y = subvolume_voxel_size_microns.y / self._voxelsize_um
        subvolume_voxel_size_volume_voxel_size_ratio.z = subvolume_voxel_size_microns.z / self._voxelsize_um
        
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
                      move_along_normal, jitter_max,
                      augment_subvolume, method, normalize=False, square_corners=None, window_min_max=None):
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
            move_along_normal: Scalar of how many units to translate
                the center point along the center vector.
            jitter_max: Jitter the center point a random amount up to
                this value in either direction along the normal vector.
            augment_subvolume: Whether to perform augmentation.
            method: String to indicate how to get the volume data.

        Returns:
            A np.float32 array of the requested shape.

        """
        assert len(center) == 3
        assert len(shape_voxels) == 3

        # If shape_microns not specified, fall back to the old method
        # (spatial extent based only on number of voxels and not voxel size)
        if shape_microns is None:
            shape_microns = list(np.array(shape_voxels) * self._voxelsize_um)

        assert len(shape_microns) == 3

        # TODO if square_corners is not None and not empty, modify basis vectors before calling (and center and normal?)
        # TODO if empty return zeros?

        if normal is None or not np.any(normal):
            normal = np.array([0, 0, 1])
        else:
            normal = normalize_fl3(normal)

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

        # Convert to float normalized to [0, 1]
        subvolume = inkid.util.uint16_to_float32_normalized_0_1(subvolume)

        # Window (contrast stretch) if requested
        if window_min_max is not None:
            window_min, window_max = window_min_max
            subvolume = inkid.util.window_0_1_array(subvolume, window_min, window_max)

        # Add singleton dimension for number of channels
        subvolume = np.expand_dims(subvolume, 0)

        return subvolume
