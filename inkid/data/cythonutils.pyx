from libc.stdint cimport uint16_t

cpdef subvol_idx_to_vol_pos(
        Py_ssize_t subvol_idx_x,
        Py_ssize_t subvol_idx_y,
        Py_ssize_t subvol_idx_z,
        Py_ssize_t subvol_shape_voxels_x,
        Py_ssize_t subvol_shape_voxels_y,
        Py_ssize_t subvol_shape_voxels_z,
        double center_x,
        double center_y,
        double center_z,
        double subvolume_voxel_size_volume_voxel_size_ratio_x,
        double subvolume_voxel_size_volume_voxel_size_ratio_y,
        double subvolume_voxel_size_volume_voxel_size_ratio_z,
        double basis_x_x,
        double basis_x_y,
        double basis_x_z,
        double basis_y_x,
        double basis_y_y,
        double basis_y_z,
        double basis_z_x,
        double basis_z_y,
        double basis_z_z,
    ):
    """Convert a subvolume index to a position in the volume space."""
    cdef double offset_x, offset_y, offset_z
    cdef double vol_pos_x, vol_pos_y, vol_pos_z
    # Convert from an index relative to an origin in the corner to a position relative to the
    # subvolume center (which may not correspond exactly to one of the subvolume voxel positions
    # if any of the side lengths are even).
    offset_x = <Py_ssize_t> ((-1 * (subvol_shape_voxels_x - 1) / 2.0 + subvol_idx_x) + 0.5)
    offset_y = <Py_ssize_t> ((-1 * (subvol_shape_voxels_y - 1) / 2.0 + subvol_idx_y) + 0.5)
    offset_z = <Py_ssize_t> ((-1 * (subvol_shape_voxels_z - 1) / 2.0 + subvol_idx_z) + 0.5)

    # Calculate the corresponding position in the volume.
    vol_pos_x = center_x
    vol_pos_y = center_y
    vol_pos_z = center_z

    vol_pos_x += (
            offset_x
            * basis_x_x
            * subvolume_voxel_size_volume_voxel_size_ratio_x
    )
    vol_pos_y += (
            offset_x
            * basis_x_y
            * subvolume_voxel_size_volume_voxel_size_ratio_x
    )
    vol_pos_z += (
            offset_x
            * basis_x_z
            * subvolume_voxel_size_volume_voxel_size_ratio_x
    )

    vol_pos_x += (
            offset_y
            * basis_y_x
            * subvolume_voxel_size_volume_voxel_size_ratio_y
    )
    vol_pos_y += (
            offset_y
            * basis_y_y
            * subvolume_voxel_size_volume_voxel_size_ratio_y
    )
    vol_pos_z += (
            offset_y
            * basis_y_z
            * subvolume_voxel_size_volume_voxel_size_ratio_y
    )

    vol_pos_x += (
            offset_z
            * basis_z_x
            * subvolume_voxel_size_volume_voxel_size_ratio_z
    )
    vol_pos_y += (
            offset_z
            * basis_z_y
            * subvolume_voxel_size_volume_voxel_size_ratio_z
    )
    vol_pos_z += (
            offset_z
            * basis_z_z
            * subvolume_voxel_size_volume_voxel_size_ratio_z
    )

    return vol_pos_x, vol_pos_y, vol_pos_z

def nearest_neighbor_with_basis_vectors(
        subvol,
        subvol_neighborhood,
        double center_x,
        double center_y,
        double center_z,
        double subvolume_voxel_size_volume_voxel_size_ratio_x,
        double subvolume_voxel_size_volume_voxel_size_ratio_y,
        double subvolume_voxel_size_volume_voxel_size_ratio_z,
        double basis_x_x,
        double basis_x_y,
        double basis_x_z,
        double basis_y_x,
        double basis_y_y,
        double basis_y_z,
        double basis_z_x,
        double basis_z_y,
        double basis_z_z,
        Py_ssize_t min_x,
        Py_ssize_t min_y,
        Py_ssize_t min_z,
    ):
    """Sample a subvolume from its axis-oriented bounding box neighborhood."""
    cdef uint16_t[:, :, :] subvol_view = subvol
    cdef const uint16_t[:, :, :] subvol_neighborhood_view = subvol_neighborhood
    cdef Py_ssize_t x, y, z
    cdef Py_ssize_t subvol_shape_x, subvol_shape_y, subvol_shape_z
    cdef Py_ssize_t subvol_neighborhood_x, subvol_neighborhood_y, subvol_neighborhood_z
    cdef double vol_pos_x, vol_pos_y, vol_pos_z

    subvol_shape_x = subvol.shape[2]
    subvol_shape_y = subvol.shape[1]
    subvol_shape_z = subvol.shape[0]

    # Iterate through subvolume indices
    for z in range(subvol_shape_z):
        for y in range(subvol_shape_y):
            for x in range(subvol_shape_x):
                # Compute the corresponding position in the volume space
                (
                    vol_pos_x,
                    vol_pos_y,
                    vol_pos_z
                ) = subvol_idx_to_vol_pos(
                    x,
                    y,
                    z,
                    subvol_shape_x,
                    subvol_shape_y,
                    subvol_shape_z,
                    center_x,
                    center_y,
                    center_z,
                    subvolume_voxel_size_volume_voxel_size_ratio_x,
                    subvolume_voxel_size_volume_voxel_size_ratio_y,
                    subvolume_voxel_size_volume_voxel_size_ratio_z,
                    basis_x_x,
                    basis_x_y,
                    basis_x_z,
                    basis_y_x,
                    basis_y_y,
                    basis_y_z,
                    basis_z_x,
                    basis_z_y,
                    basis_z_z,
                )

                # Convert that to a position in the neighborhood space
                vol_pos_x -= min_x
                vol_pos_y -= min_y
                vol_pos_z -= min_z

                # Convert to int
                subvol_neighborhood_x = <Py_ssize_t> (vol_pos_x + 0.5)
                subvol_neighborhood_y = <Py_ssize_t> (vol_pos_y + 0.5)
                subvol_neighborhood_z = <Py_ssize_t> (vol_pos_z + 0.5)

                # Sample that voxel of the subvolume from its neighborhood
                subvol_view[z, y, x] = subvol_neighborhood_view[
                    subvol_neighborhood_z,
                    subvol_neighborhood_y,
                    subvol_neighborhood_x
                ]

    return
