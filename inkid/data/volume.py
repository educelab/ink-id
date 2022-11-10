from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tensorstore as ts
from tqdm import tqdm

import inkid.data.cythonutils
from inkid.data.mathutils import (
    I3,
    Fl3,
    Fl3x3,
    normalize,
    get_component_vectors_from_normal,
    get_basis_from_square,
)
from inkid.util import uint16_to_float32_normalized_0_1


class Volume:
    initialized_volumes: dict[str, Volume] = dict()

    @classmethod
    def from_path(cls, path: str) -> Volume:
        if path in cls.initialized_volumes:
            return cls.initialized_volumes[path]
        cls.initialized_volumes[path] = Volume(path)
        return cls.initialized_volumes[path]

    def __init__(self, vol_path: str):
        vol_path = Path(vol_path)

        # Load metadata
        self._metadata = dict()
        metadata_filename = vol_path / "meta.json"
        if not metadata_filename.exists():
            raise FileNotFoundError(f"No volume meta.json file found in {vol_path}")
        else:
            with open(metadata_filename) as f:
                self._metadata = json.loads(f.read())
        self._voxelsize_um = self._metadata["voxelsize"]
        self.shape_z = self._metadata["slices"]
        self.shape_y = self._metadata["height"]
        self.shape_x = self._metadata["width"]

        if "zarr" in vol_path:
            chunk_size = 256
            self._data = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": str(vol_path),
                    },
                    "metadata": {
                        "shape": [self.shape_z, self.shape_y, self.shape_x],
                        "chunks": [chunk_size, chunk_size, chunk_size],
                        "dtype": "<u2",
                    },
                    "context": {
                        "cache_pool": {
                            "total_bytes_limit": 100000000,
                        }
                    },
                }
            ).result()
        else:

            # Get list of slice image filenames
            slice_files = []
            for child in vol_path.iterdir():
                if not child.is_file():
                    continue
                # Make sure it is not a hidden file and it's a .tif
                if child.name[0] != "." and child.suffix == ".tif":
                    slice_files.append(str(child))
            slice_files.sort()
            assert len(slice_files) == self.shape_z

            # Load slice images into volume
            logging.info("Loading volume slices from {}...".format(vol_path))
            vol = np.empty((self.shape_z, self.shape_y, self.shape_x), dtype=np.uint16)
            for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
                img = np.array(Image.open(slice_file), dtype=np.uint16).copy()
                vol[slice_i, :, :] = img
            print()

            self._data = ts.open(
                {"driver": "array", "array": vol, "dtype": "uint16"}
            ).result()

    def __getitem__(self, key):
        return self._data[key].read().result()

    @property
    def shape(self) -> I3:
        return self._data.shape

    def get_subvolume(
        self,
        center: Fl3,
        shape_voxels: I3,
        shape_microns: Optional[Fl3] = None,
        normal: Fl3 = (0.0, 0.0, 1.0),
        square_corners=None,
        augment_subvolume: bool = False,
        jitter_max: float = 4,
        move_along_normal: float = 0,
        normalize_subvolume: bool = False,
    ):
        assert len(center) == 3
        assert len(shape_voxels) == 3

        # If shape_microns not specified, fall back to the old method
        # (spatial extent based only on number of voxels and not voxel size)
        if shape_microns is None:
            shape_microns = tuple(np.array(shape_voxels) * self._voxelsize_um)
        assert len(shape_microns) == 3

        normal = normalize(normal)

        move_along_normal += random.uniform(-jitter_max, jitter_max)
        center = np.array(center, dtype=float)
        center += move_along_normal * np.array(normal, dtype=float)
        center = tuple(center)

        if square_corners is None:
            basis: Fl3x3 = get_component_vectors_from_normal(normal)
        else:
            basis: Fl3x3 = get_basis_from_square(square_corners)

        subvolume_voxel_size_microns: Fl3 = (
            shape_microns[0] / shape_voxels[0],
            shape_microns[1] / shape_voxels[1],
            shape_microns[2] / shape_voxels[2],
        )

        subvolume_voxel_size_volume_voxel_size_ratio: Fl3 = tuple(
            np.array(subvolume_voxel_size_microns) / self._voxelsize_um
        )

        # Compute axis oriented bounding box around subvolume
        d, h, w = shape_voxels
        subvol_corners: list[I3] = [
            (0, 0, 0),
            (0, h - 1, 0),
            (w - 1, 0, 0),
            (w - 1, h - 1, 0),
            (0, 0, d - 1),
            (0, h - 1, d - 1),
            (w - 1, 0, d - 1),
            (w - 1, h - 1, d - 1),
        ]
        vol_positions_of_subvol_corners = []
        for subvol_pos in subvol_corners:
            x, y, z = subvol_pos
            vol_pos: Fl3 = inkid.data.cythonutils.subvol_idx_to_vol_pos(
                x,
                y,
                z,
                shape_voxels[2],
                shape_voxels[1],
                shape_voxels[0],
                center[0],
                center[1],
                center[2],
                subvolume_voxel_size_volume_voxel_size_ratio[2],
                subvolume_voxel_size_volume_voxel_size_ratio[1],
                subvolume_voxel_size_volume_voxel_size_ratio[0],
                basis[0][0],
                basis[0][1],
                basis[0][2],
                basis[1][0],
                basis[1][1],
                basis[1][2],
                basis[2][0],
                basis[2][1],
                basis[2][2],
            )
            vol_positions_of_subvol_corners.append(vol_pos)
        vol_positions_of_subvol_corners = np.rint(vol_positions_of_subvol_corners).astype(int)
        min_x, min_y, min_z = np.amin(vol_positions_of_subvol_corners, axis=0)
        max_x, max_y, max_z = np.amax(vol_positions_of_subvol_corners, axis=0)
        bbox_shape: I3 = max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1

        # Compute the subvolume overlap with the actual volume bounds (in most cases complete overlap)
        # Compute overlap in volume
        overlap_min_x = min(max(min_x, 0), self.shape[2] - 1)
        overlap_max_x = min(max(max_x, 0), self.shape[2] - 1)
        overlap_min_y = min(max(min_y, 0), self.shape[1] - 1)
        overlap_max_y = min(max(max_y, 0), self.shape[1] - 1)
        overlap_min_z = min(max(min_z, 0), self.shape[0] - 1)
        overlap_max_z = min(max(max_z, 0), self.shape[0] - 1)
        # Compute overlap in subvol bbox
        subvol_overlap_min_x = min(max(overlap_min_x - min_x, 0), bbox_shape[2] - 1)
        subvol_overlap_max_x = min(max(overlap_max_x - min_x, 0), bbox_shape[2] - 1)
        subvol_overlap_min_y = min(max(overlap_min_y - min_y, 0), bbox_shape[1] - 1)
        subvol_overlap_max_y = min(max(overlap_max_y - min_y, 0), bbox_shape[1] - 1)
        subvol_overlap_min_z = min(max(overlap_min_z - min_z, 0), bbox_shape[0] - 1)
        subvol_overlap_max_z = min(max(overlap_max_z - min_z, 0), bbox_shape[0] - 1)

        # Initialize the neighborhood/bounding box to zeros
        subvol_neighborhood = np.zeros(bbox_shape, dtype=np.uint16)

        # Fill neighborhood with data from volume, where it overlaps (isn't out of bounds)
        subvol_neighborhood[
            subvol_overlap_min_z : subvol_overlap_max_z + 1,
            subvol_overlap_min_y : subvol_overlap_max_y + 1,
            subvol_overlap_min_x : subvol_overlap_max_x + 1,
        ] = self[
            overlap_min_z : overlap_max_z + 1,
            overlap_min_y : overlap_max_y + 1,
            overlap_min_x : overlap_max_x + 1,
        ]

        # Pass this filled neighborhood to the subroutine which samples it using C loops
        subvol = np.zeros(shape_voxels, dtype=np.uint16)
        inkid.data.cythonutils.nearest_neighbor_with_basis_vectors(
            subvol,
            subvol_neighborhood,
            center[0],
            center[1],
            center[2],
            subvolume_voxel_size_volume_voxel_size_ratio[2],
            subvolume_voxel_size_volume_voxel_size_ratio[1],
            subvolume_voxel_size_volume_voxel_size_ratio[0],
            basis[0][0],
            basis[0][1],
            basis[0][2],
            basis[1][0],
            basis[1][1],
            basis[1][2],
            basis[2][0],
            basis[2][1],
            basis[2][2],
            min_x,
            min_y,
            min_z
        )

        # TODO move this elsewhere
        if augment_subvolume:
            flip_direction = np.random.randint(4)
            if flip_direction == 0:
                subvol = np.flip(subvol, axis=1)  # Flip y
            elif flip_direction == 1:
                subvol = np.flip(subvol, axis=2)  # Flip x
            elif flip_direction == 2:
                subvol = np.flip(subvol, axis=1)  # Flip x and y
                subvol = np.flip(subvol, axis=2)

            rotate_direction = np.random.randint(4)
            subvol = np.rot90(subvol, k=rotate_direction, axes=(1, 2))

        # TODO move this elsewhere
        if normalize_subvolume:
            subvol = np.asarray(subvol, dtype=np.float32)
            subvol = subvol - subvol.mean()
            subvol = subvol / subvol.std()

        # TODO move this elsewhere
        # Convert to float normalized to [0, 1]
        subvol = uint16_to_float32_normalized_0_1(subvol)

        # TODO move this elsewhere?
        # Add singleton dimension for number of channels
        subvol = np.expand_dims(subvol, 0)

        return subvol


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vol-path", required=True)
    args = parser.parse_args()

    start = time.time()
    vol = Volume.from_path(args.vol_path)
    end = time.time()
    print(f"{end - start} seconds to initialize {vol.shape} volume")

    subvol_center = vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[0] // 2
    subvol_shape_voxels = 24, 80, 80

    vol.get_subvolume(subvol_center, subvol_shape_voxels)
    start = time.time()
    vol.get_subvolume(subvol_center, subvol_shape_voxels)
    end = time.time()
    print(f"{end - start} seconds to fetch {subvol_shape_voxels} subvolume")


if __name__ == "__main__":
    main()
