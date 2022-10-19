from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tensorstore as ts
from tqdm import tqdm

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

        if vol_path.suffix == ".zarr":
            chunk_size = 64
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
                    }
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
        # TODO consider adding bounds checking and return 0 if not in bounds (to match previous implementation)
        #   It would be nice to avoid that if possible (doesn't affect ML performance), though, because
        #   it breaks the intuition around the array access.
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
        # TODO removed: augment, jitter, move_along_normal, normalize_subvol
        assert len(center) == 3
        assert len(shape_voxels) == 3

        # If shape_microns not specified, fall back to the old method
        # (spatial extent based only on number of voxels and not voxel size)
        if shape_microns is None:
            shape_microns = tuple(np.array(shape_voxels) * self._voxelsize_um)
        assert len(shape_microns) == 3

        normal = normalize(normal)

        if square_corners is None:
            basis = get_component_vectors_from_normal(normal)
        else:
            basis = get_basis_from_square(square_corners)

        subvolume = self.nearest_neighbor_with_basis_vectors(
            center, shape_voxels, shape_microns, basis
        )
        assert subvolume.shape == shape_voxels

        # TODO move this elsewhere
        # Convert to float normalized to [0, 1]
        subvolume = uint16_to_float32_normalized_0_1(subvolume)

        # TODO move this elsewhere?
        # Add singleton dimension for number of channels
        subvolume = np.expand_dims(subvolume, 0)

        return subvolume

    def nearest_neighbor_with_basis_vectors(
        self,
        center: Fl3,
        shape_voxels: I3,
        shape_microns: Fl3,
        basis: Fl3x3,
    ) -> np.array:
        basis = np.array(basis)

        subvolume_voxel_shape_microns: Fl3 = (
            shape_microns[0] / shape_voxels[0],
            shape_microns[1] / shape_voxels[1],
            shape_microns[2] / shape_voxels[2],
        )

        subvolume_voxel_size_volume_voxel_size_ratio: Fl3 = tuple(
            np.array(subvolume_voxel_shape_microns) / self._voxelsize_um
        )

        # Shape is [z, y, x] but I want points to be in [x, y, z] hence the reversing with [::-1]
        points = np.array(list(np.ndindex(*shape_voxels[::-1])), dtype=float)
        # Convert from index relative to origin in the corner to a position relative to the subvolume center
        points -= np.array(
            [
                (shape_voxels[2] - 1) / 2.0 - 0.5,
                (shape_voxels[1] - 1) / 2.0 - 0.5,
                (shape_voxels[0] - 1) / 2.0 - 0.5,
            ]
        )
        xs = np.sum(basis[:, 0] * points, axis=1) * subvolume_voxel_size_volume_voxel_size_ratio[2] + center[0] + 0.5
        ys = np.sum(basis[:, 1] * points, axis=1) * subvolume_voxel_size_volume_voxel_size_ratio[1] + center[1] + 0.5
        zs = np.sum(basis[:, 2] * points, axis=1) * subvolume_voxel_size_volume_voxel_size_ratio[0] + center[2] + 0.5
        xs = np.array(xs, dtype=int)
        ys = np.array(ys, dtype=int)
        zs = np.array(zs, dtype=int)

        subvol = self[zs, ys, xs].reshape(shape_voxels)
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

    subvol = vol.get_subvolume(subvol_center, subvol_shape_voxels)
    start = time.time()
    subvol = vol.get_subvolume(subvol_center, subvol_shape_voxels)
    end = time.time()
    print(f"{end - start} seconds to fetch {subvol.shape} subvolume")
    # print(subvol)


if __name__ == "__main__":
    main()
