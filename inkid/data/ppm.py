# For PPM.initialized_ppms https://stackoverflow.com/a/33533514
from __future__ import annotations

import logging
import re
import struct
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import inkid.util


class PPM:
    initialized_ppms: Dict[str, PPM] = dict()

    def __init__(self, path: str, lazy_load: bool = False):
        self._path = path

        header = PPM.parse_ppm_header(path)
        self.width: int = header["width"]
        self.height: int = header["height"]
        self.dim: int = header["dim"]
        self.ordered: bool = header["ordered"]
        self.type: str = header["type"]
        self.version: str = header["version"]

        self.data: Optional[np.typing.ArrayLike] = None

        logging.info(
            f"Initialized PPM for {self._path} with width {self.width}, "
            f"height {self.height}, dim {self.dim}"
        )

        if not lazy_load:
            self.ensure_loaded()

    def is_loaded(self):
        return self.data is not None

    def ensure_loaded(self):
        if not self.is_loaded():
            self.load_ppm_data()

    @classmethod
    def from_path(cls, path: str, lazy_load: bool = False) -> PPM:
        if path in cls.initialized_ppms:
            return cls.initialized_ppms[path]
        cls.initialized_ppms[path] = PPM(path, lazy_load=lazy_load)
        return cls.initialized_ppms[path]

    @staticmethod
    def parse_ppm_header(filename):
        comments_re = re.compile("^#")
        width_re = re.compile("^width")
        height_re = re.compile("^height")
        dim_re = re.compile("^dim")
        ordered_re = re.compile("^ordered")
        type_re = re.compile("^type")
        version_re = re.compile("^version")
        header_terminator_re = re.compile("^<>$")

        width, height, dim, ordered, val_type, version = [None] * 6

        data = inkid.util.get_raw_data_from_file_or_url(filename)
        while True:
            line = data.readline().decode("utf-8")
            if comments_re.match(line):
                pass
            elif width_re.match(line):
                width = int(line.split(": ")[1])
            elif height_re.match(line):
                height = int(line.split(": ")[1])
            elif dim_re.match(line):
                dim = int(line.split(": ")[1])
            elif ordered_re.match(line):
                ordered = line.split(": ")[1].strip() == "true"
            elif type_re.match(line):
                val_type = line.split(": ")[1].strip()
                assert val_type in ["double"]
            elif version_re.match(line):
                version = line.split(": ")[1].strip()
            elif header_terminator_re.match(line):
                break
            else:
                logging.warning(
                    "PPM header contains unknown line: {}".format(line.strip())
                )

        return {
            "width": width,
            "height": height,
            "dim": dim,
            "ordered": ordered,
            "type": val_type,
            "version": version,
        }

    @staticmethod
    def write_ppm_from_data(
        path: str,
        data: np.typing.ArrayLike,
        width: int,
        height: int,
        dim: int,
        ordered: bool = True,
        version: str = "1.0",
    ):
        with open(path, "wb") as f:
            logging.info("Writing PPM to file {}...".format(path))
            f.write("width: {}\n".format(width).encode("utf-8"))
            f.write("height: {}\n".format(height).encode("utf-8"))
            f.write("dim: {}\n".format(dim).encode("utf-8"))
            f.write(
                "ordered: {}\n".format("true" if ordered else "false").encode("utf-8")
            )
            f.write("type: double\n".encode("utf-8"))
            f.write("version: {}\n".format(version).encode("utf-8"))
            f.write("<>\n".encode("utf-8"))
            for y in tqdm(range(height)):
                for x in range(width):
                    for idx in range(dim):
                        f.write(struct.pack("d", data[y, x, idx]))

    def load_ppm_data(self):
        """Read the PPM file data and store it in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3].

        """
        logging.info(
            f"Loading PPM data for {self._path} with width {self.width}, "
            f"height {self.height}, dim {self.dim}..."
        )

        self.data = np.empty((self.height, self.width, self.dim))

        data = inkid.util.get_raw_data_from_file_or_url(self._path)
        header_terminator_re = re.compile("^<>$")
        while True:
            line = data.readline().decode("utf-8")
            if header_terminator_re.match(line):
                break

        for y in tqdm(range(self.height)):
            for x in range(self.width):
                for idx in range(self.dim):
                    # Only works if dim == 6: (x, y, z, n_x, n_y, n_z)
                    if self.dim == 6:
                        self.data[y, x, idx] = struct.unpack("d", data.read(8))[0]
        print()

    def get_point_with_normal(self, ppm_x, ppm_y):
        self.ensure_loaded()
        return self.data[ppm_y][ppm_x]

    def scale_down_by(self, scale_factor):
        self.ensure_loaded()

        self.width //= scale_factor
        self.height //= scale_factor

        new_data = np.empty((self.height, self.width, self.dim))

        logging.info(
            "Downscaling PPM by factor of {} on all axes...".format(scale_factor)
        )
        for y in tqdm(range(self.height)):
            for x in range(self.width):
                for idx in range(self.dim):
                    new_data[y, x, idx] = self.data[
                        y * scale_factor, x * scale_factor, idx
                    ]

        self.data = new_data

    def translate(self, dx: int, dy: int, dz: int) -> None:
        for ppm_y in tqdm(range(self.height)):
            for ppm_x in range(self.width):
                if np.any(self.data[ppm_y, ppm_x]):  # Leave empty pixels unchanged
                    vol_x, vol_y, vol_z = self.data[ppm_y, ppm_x, 0:3]
                    self.data[ppm_y, ppm_x, 0] = vol_x + dx
                    self.data[ppm_y, ppm_x, 1] = vol_y + dy
                    self.data[ppm_y, ppm_x, 2] = vol_z + dz

    def write(self, filename):
        self.ensure_loaded()

        with open(filename, "wb") as f:
            logging.info("Writing PPM to file {}...".format(filename))
            f.write("width: {}\n".format(self.width).encode("utf-8"))
            f.write("height: {}\n".format(self.height).encode("utf-8"))
            f.write("dim: {}\n".format(self.dim).encode("utf-8"))
            f.write(
                "ordered: {}\n".format("true" if self.ordered else "false").encode(
                    "utf-8"
                )
            )
            f.write("type: double\n".encode("utf-8"))
            f.write("version: {}\n".format(self.version).encode("utf-8"))
            f.write("<>\n".encode("utf-8"))
            for y in tqdm(range(self.height)):
                for x in range(self.width):
                    for idx in range(self.dim):
                        f.write(struct.pack("d", self.data[y, x, idx]))
