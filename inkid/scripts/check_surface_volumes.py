import argparse
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of the surface volumes")
    args = parser.parse_args()

    for subdir in Path(args.root).iterdir():
        if subdir.is_dir():
            parts = subdir.name.split("_")[:2]
            name = "_".join(parts)

            extras_dir = subdir / "extras"
            assert extras_dir.exists(), f"Missing extras directory in {subdir}"
            extras_suffixes = [
                "_alignment.psd",
                "_cellmap.tif",
                "_faux_rgb.png",
                "_faux_rgb.psd",
                "_ir.png",
                ".mtl",
                ".obj",
                ".png",
                ".ppm",
                "_surface_volume_grid2x1.txt",
                "_surface_volume.json",
                "_surface_volume.ppm",
                ".tif",
                "_uvs.png",
            ]
            for extras_suffix in extras_suffixes:
                filename = name + extras_suffix
                filepath = extras_dir / filename
                assert filepath.exists(), f"Missing {filename} in {extras_dir}"
                if extras_suffix == "_faux_rgb.png":
                    img = Image.open(filepath)
                    assert img.mode == "RGB", f"Image {filepath} mode is not RGB"
                elif extras_suffix in ["_ir.png"]:
                    img = Image.open(filepath)
                    assert img.mode == "L", f"Image {filepath} mode is not L"
                elif extras_suffix in [".png"]:
                    img = Image.open(filepath)
                    assert img.mode == "I", f"Image {filepath} mode is not I"
                elif extras_suffix in ["_surface_volume.json"]:
                    with open(filepath) as f:
                        data = json.load(f)
                        keys = [
                            "volume",
                            "ppm",
                            "mask",
                            "ink_label",
                            "rgb_label",
                            "volcart_texture_label",
                        ]
                        for key in keys:
                            key_path = extras_dir / data[key]
                            assert key_path.exists(), f"{key_path} specified in _surface_volume.json does not exist"

            root_suffixes = [
                "_inklabels.png",
                "_mask.png",
            ]
            for root_suffix in root_suffixes:
                filename = name + root_suffix
                filepath = subdir / filename
                assert filepath.exists(), f"Missing {filename} in {subdir}"
                if root_suffix == "_inklabels.png":
                    img = Image.open(filepath)
                    assert img.mode == "P", f"Image {filepath} mode is not P"
                elif root_suffix == "_mask.png":
                    img = Image.open(filepath)
                    assert img.mode == "P", f"Image {filepath} mode is not L"

            surface_volume_dir = subdir / (name + "_surface_volume")
            assert surface_volume_dir.exists(), f"Missing surface volume directory in {subdir}"
            tif_files = list(surface_volume_dir.glob("*.tif"))
            assert len(tif_files) == 65, f"Expected 65 tif files in {surface_volume_dir}, found {len(tif_files)}"
            meta_json_path = surface_volume_dir / "meta.json"
            assert meta_json_path.exists(), f"Missing meta.json in {surface_volume_dir}"


if __name__ == '__main__':
    main()
