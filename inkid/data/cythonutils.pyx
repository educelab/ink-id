def subvol_idx_to_vol_pos(
    int subvol_idx_x,
    int subvol_idx_y,
    int subvol_idx_z,
    int subvol_shape_voxels_x,
    int subvol_shape_voxels_y,
    int subvol_shape_voxels_z,
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
    cdef int offset_x, offset_y, offset_z
    cdef double vol_pos_x, vol_pos_y, vol_pos_z
    # Convert from an index relative to an origin in the corner to a position relative to the
    # subvolume center (which may not correspond exactly to one of the subvolume voxel positions
    # if any of the side lengths are even).
    offset_x = <int> ((-1 * (subvol_shape_voxels_x - 1) / 2.0 + subvol_idx_x) + 0.5)
    offset_y = <int> ((-1 * (subvol_shape_voxels_y - 1) / 2.0 + subvol_idx_y) + 0.5)
    offset_z = <int> ((-1 * (subvol_shape_voxels_z - 1) / 2.0 + subvol_idx_z) + 0.5)

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

    vol_pos_x += 0.5
    vol_pos_y += 0.5
    vol_pos_z += 0.5

    return vol_pos_x, vol_pos_y, vol_pos_z
