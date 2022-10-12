from __future__ import annotations

import json
from pathlib import Path

import tensorstore as ts


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
        return self._data[key].read().result()


def main():
    vol = NewVolume.from_path("/home/stephen/data/dri-datasets-drive/PHercParis3/volumes/20211026092241.zarr")


if __name__ == "__main__":
    main()
