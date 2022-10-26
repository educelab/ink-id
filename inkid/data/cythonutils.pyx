def subvol_idx_to_vol_pos(
    cI3 subvol_idx,
    cIShape subvol_shape_voxels,
    cD3 center,
    cDShape subvolume_voxel_size_volume_voxel_size_ratio,
    cD3x3 basis,
) -> cD3:
    cdef cI3 offset
    cdef cD3 vol_pos
    # Convert from an index relative to an origin in the corner to a position relative to the
    # subvolume center (which may not correspond exactly to one of the subvolume voxel positions
    # if any of the side lengths are even).
    offset.x = <int> ((-1 * (subvol_shape_voxels.x - 1) / 2.0 + subvol_idx.x) + 0.5)
    offset.y = <int> ((-1 * (subvol_shape_voxels.y - 1) / 2.0 + subvol_idx.y) + 0.5)
    offset.z = <int> ((-1 * (subvol_shape_voxels.z - 1) / 2.0 + subvol_idx.z) + 0.5)

    # Calculate the corresponding position in the volume.
    vol_pos.x = center.x
    vol_pos.y = center.y
    vol_pos.z = center.z

    vol_pos.x += (
            offset.x
            * basis.x.x
            * subvolume_voxel_size_volume_voxel_size_ratio.x
    )
    vol_pos.y += (
            offset.x
            * basis.x.y
            * subvolume_voxel_size_volume_voxel_size_ratio.x
    )
    vol_pos.z += (
            offset.x
            * basis.x.z
            * subvolume_voxel_size_volume_voxel_size_ratio.x
    )

    vol_pos.x += (
            offset.y
            * basis.y.x
            * subvolume_voxel_size_volume_voxel_size_ratio.y
    )
    vol_pos.y += (
            offset.y
            * basis.y.y
            * subvolume_voxel_size_volume_voxel_size_ratio.y
    )
    vol_pos.z += (
            offset.y
            * basis.y.z
            * subvolume_voxel_size_volume_voxel_size_ratio.y
    )

    vol_pos.x += (
            offset.z
            * basis.z.x
            * subvolume_voxel_size_volume_voxel_size_ratio.z
    )
    vol_pos.y += (
            offset.z
            * basis.z.y
            * subvolume_voxel_size_volume_voxel_size_ratio.z
    )
    vol_pos.z += (
            offset.z
            * basis.z.z
            * subvolume_voxel_size_volume_voxel_size_ratio.z
    )

    vol_pos.x += 0.5
    vol_pos.y += 0.5
    vol_pos.z += 0.5

    return vol_pos
