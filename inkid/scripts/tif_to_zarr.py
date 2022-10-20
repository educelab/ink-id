import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import tensorstore as ts
from tqdm import tqdm


def tif_to_zarr(slices_path: str, zarr_path: str, chunk_size: int, stage_writes: bool):
    slices_path = Path(slices_path)
    zarr_path = Path(zarr_path)

    # Load metadata
    metadata_filename = slices_path / "meta.json"
    if not metadata_filename.exists():
        raise FileNotFoundError(f"No volume meta.json file found in {slices_path}")
    else:
        with open(metadata_filename) as f:
            metadata = json.loads(f.read())
    shape_z = metadata["slices"]
    shape_y = metadata["height"]
    shape_x = metadata["width"]

    # Get list of slice image filenames
    slice_files = []
    for child in slices_path.iterdir():
        if not child.is_file():
            continue
        # Make sure it is not a hidden file and it's a .tif
        if child.name[0] != "." and child.suffix == ".tif":
            slice_files.append(str(child))
    slice_files.sort()
    assert len(slice_files) == shape_z

    dataset = ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(zarr_path),
            },
            "metadata": {
                "shape": [shape_z, shape_y, shape_x],
                "chunks": [chunk_size, chunk_size, chunk_size],
                "dtype": "<u2",
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()

    txn = ts.Transaction() if stage_writes else None

    # Write slice images into volume
    for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
        img = np.array(Image.open(slice_file), dtype=np.uint16).copy()
        # With a 600MB volume, async (staged): 24s, sync: 19m. ~40x speedup.
        if stage_writes:
            dataset.with_transaction(txn)[slice_i, :, :] = img
        else:
            dataset[slice_i, :, :] = img

    # Commit (write to disk) the staged slice writes
    if stage_writes:
        txn.commit_async().result()

    # Copy metadata file
    shutil.copy(metadata_filename, zarr_path / "meta.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-slices-dir", required=True)
    parser.add_argument("--output-zarr-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--stage-writes", action="store_true")
    args = parser.parse_args()

    tif_to_zarr(
        args.input_slices_dir, args.output_zarr_dir, args.chunk_size, args.stage_writes
    )


if __name__ == "__main__":
    main()
