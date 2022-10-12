from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorstore as ts


def norm_fl3(vec: tuple[float, float, float]) -> float:
    vec = np.array(vec)
    return (vec[0]**2 + vec[1]**2 + vec[2]**2)**(1./2)


def normalize_fl3(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    vec = np.array(vec)
    n = norm_fl3(vec)
    return tuple(vec / n)


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
        # It would be nice to avoid that if possible (doesn't affect ML performance), though, because
        # it breaks the intuition around the array access.
        return self._data[key].read().result()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    def get_subvolume(
        self,
        center: tuple[float, float, float],
        shape_voxels: tuple[int, int, int],
        shape_microns: Optional[tuple[float, float, float]] = None,
        normal: tuple[float, float, float] = (0, 0, 1),
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

        normal = normalize_fl3(normal)

        # TODO LEFT OFF implementing this stuff
        if square_corners is None:
            basis = get_component_vectors_from_normal(n)
        else:
            basis = get_basis_from_square(square_corners)

        self.nearest_neighbor_with_basis_vectors(c, s_v, s_m, basis, subvolume)

        raise NotImplementedError

    def nearest_neighbor_with_basis_vectors(self):
        raise NotImplementedError


def main():
    vol = NewVolume.from_path(
        "/home/stephen/data/dri-datasets-drive/PHercParis3/volumes/20211026092241.zarr"
    )


if __name__ == "__main__":
    main()
