from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorstore as ts

from inkid.data.mathutils import I3, Fl3, Fl3x3, normalize, get_component_vectors_from_normal, get_basis_from_square
from inkid.util import uint16_to_float32_normalized_0_1


class NewVolume:
    initialized_volumes: dict[str, NewVolume] = dict()

    @classmethod
    def from_path(cls, path: str) -> NewVolume:
        if path in cls.initialized_volumes:
            return cls.initialized_volumes[path]
        cls.initialized_volumes[path] = NewVolume(path)
        return cls.initialized_volumes[path]

    def __init__(self, zarr_path: str):
        # Load metadata
        self._metadata = dict()
        metadata_filename = Path(zarr_path) / "meta.json"
        if not metadata_filename.exists():
            raise FileNotFoundError(f"No volume meta.json file found in {zarr_path}")
        else:
            with open(metadata_filename) as f:
                self._metadata = json.loads(f.read())
        self._voxelsize_um = self._metadata["voxelsize"]
        self.shape_z = self._metadata["slices"]
        self.shape_y = self._metadata["height"]
        self.shape_x = self._metadata["width"]

        chunk_size = 64
        self._data = ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(zarr_path),
                },
                "metadata": {
                    "shape": [self.shape_z, self.shape_y, self.shape_x],
                    "chunks": [chunk_size, chunk_size, chunk_size],
                    "dtype": "<u2",
                },
            }
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
    ):
        # TODO removed: augment, jitter, move_along_normal, window_min_max, normalize, method
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

        subvolume = self.nearest_neighbor_with_basis_vectors(center, shape_voxels, shape_microns, basis)
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
        array = np.zeros(shape_voxels, dtype=np.uint16)

        subvolume_voxel_size_microns: Fl3 = (
            shape_microns[0] / shape_voxels[0],
            shape_microns[1] / shape_voxels[1],
            shape_microns[2] / shape_voxels[2]
        )

        subvolume_voxel_size_volume_voxel_size_ratio: Fl3 = tuple(
            np.array(subvolume_voxel_size_microns) / self._voxelsize_um
        )

        for z in range(shape_voxels[0]):
            for y in range(shape_voxels[1]):
                for x in range(shape_voxels[2]):
                    # Convert from an index relative to an origin in
                    # the corner to a position relative to the
                    # subvolume center (which may not correspond
                    # exactly to one of the subvolume voxel positions
                    # if any of the side lengths are even).
                    offset: I3 = (
                        int((-1 * (shape_voxels[2] - 1) / 2.0 + x) + 0.5),
                        int((-1 * (shape_voxels[1] - 1) / 2.0 + y) + 0.5),
                        int((-1 * (shape_voxels[0] - 1) / 2.0 + z) + 0.5)
                    )

                    # Calculate the corresponding position in the volume.
                    vol_x: float = center[0]
                    vol_y: float = center[1]
                    vol_z: float = center[2]

                    vol_x += offset[0] * basis[0][0] * subvolume_voxel_size_volume_voxel_size_ratio[2]
                    vol_y += offset[0] * basis[0][1] * subvolume_voxel_size_volume_voxel_size_ratio[2]
                    vol_z += offset[0] * basis[0][2] * subvolume_voxel_size_volume_voxel_size_ratio[2]

                    vol_x += offset[1] * basis[1][0] * subvolume_voxel_size_volume_voxel_size_ratio[1]
                    vol_y += offset[1] * basis[1][1] * subvolume_voxel_size_volume_voxel_size_ratio[1]
                    vol_z += offset[1] * basis[1][2] * subvolume_voxel_size_volume_voxel_size_ratio[1]

                    vol_x += offset[2] * basis[2][0] * subvolume_voxel_size_volume_voxel_size_ratio[0]
                    vol_y += offset[2] * basis[2][1] * subvolume_voxel_size_volume_voxel_size_ratio[0]
                    vol_z += offset[2] * basis[2][2] * subvolume_voxel_size_volume_voxel_size_ratio[0]

                    vol_x += 0.5
                    vol_y += 0.5
                    vol_z += 0.5

                    array[z, y, x] = self[int(z), int(y), int(x)]

        return array


def main():
    vol = NewVolume.from_path(
        "/home/stephen/data/dri-datasets-drive/PHercParis3/volumes/20211026092241.zarr"
    )
    print(vol[0, 0, 0])


if __name__ == "__main__":
    main()
