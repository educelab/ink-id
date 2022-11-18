import argparse
import json
from pathlib import Path
import shutil

from tqdm import tqdm


if __name__ == "__main__":
    """
    Example based on the slabs in PHercParis1Fr39, 54KeV. These are shown here in the same order they appear in
    their parent directory.

                                       0
                                       [---------]    row_001
                                0      1548
                                [------|--]           row_002
                         0      1547
                         [------|--]                  row_003
                  0      1548
                  [------|--]                         row_004
           0      1547
           [------|--]                                row_005
    0      1548
    [------|--]                                       row_006

    We want to reverse this order so we can iterate over the slabs starting at slice 0 until the overlap index for
    the first slab and then the second and so on. So we use --reverse-slab-order and get:

    0      1548
    [------|--]                                       row_006
           0      1547
           [------|--]                                row_005
                  0      1548
                  [------|--]                         row_004
                         0      1547
                         [------|--]                  row_003
                                0      1548
                                [------|--]           row_002
                                       0
                                       [---------]    row_001

    So here the relevant arguments would be `--indices 1548 1547 1548 1547 1548 --reverse-slab-order` with result:

    Taking slices 0 to 1547 from slab 20200702135633: PHercParis1Fr39_54keV_row_006
    Taking slices 0 to 1546 from slab 20200701102508: PHercParis1Fr39_54keV_row_005
    Taking slices 0 to 1547 from slab 20200630104641: PHercParis1Fr39_54keV_row_004
    Taking slices 0 to 1546 from slab 20200629110330: PHercParis1Fr39_54keV_row_003
    Taking slices 0 to 1547 from slab 20200625101031: PHercParis1Fr39_54keV_row_002
    Taking slices 0 to 2149 from slab 20200623100550: PHercParis1Fr39_54keV_row_001
    Total slices: 9888

    """
    parser = argparse.ArgumentParser(description="Merge slabs given overlap indices")
    parser.add_argument(
        "--in-dir", "-i", required=True, help="Directory of slabs to be merged"
    )
    parser.add_argument(
        "--out-dir", "-o", required=True, help="Output merged slab directory"
    )
    parser.add_argument(
        "--indices",
        type=int,
        required=True,
        nargs="+",
        help="slab indices to indicate overlap",
    )
    parser.add_argument(
        "--reverse-slab-order",
        action="store_true",
        help="Indicate if the sorted order of the slabs should be reversed",
    )

    args = parser.parse_args()

    def vol_name_from_dir(dirname):
        with open(dirname / "meta.json", "r") as meta_f:
            return json.load(meta_f)["name"]

    # Sort the folders by the actual volume/slab name, not the folder name, since they could have been
    # processed out of order.
    slab_folders = sorted(
        Path(args.in_dir).iterdir(),
        reverse=args.reverse_slab_order,
        key=vol_name_from_dir,
    )

    if not isinstance(args.indices, list):
        args.indices = [args.indices] * (len(slab_folders) - 1)
    # The number of indices provided should be number of slabs minus one - all slices are taken from the first/last slab
    assert len(args.indices) == len(slab_folders) - 1

    # We want all the slices from the first slab
    args.indices.insert(0, len(list(slab_folders[0].glob("*.tif"))))

    # Reverse slice indices if needed
    if args.reverse_slab_order:
        args.indices = args.indices[::-1]

    all_source_slices_in_merged_volume = []

    # Iterate over the slab folders and get the filenames of the slices we are getting from each slab
    for i, slab_folder in enumerate(slab_folders):
        slab_slices = sorted(slab_folder.glob("*.tif"))
        cutoff_slice_from_this_slab = args.indices[i]
        with open(slab_folder / "meta.json", "r") as f:
            slab_volume_name = json.load(f)["name"]
        print(
            f"Taking slices 0 to {cutoff_slice_from_this_slab - 1} from slab {slab_folder.stem}: {slab_volume_name}"
        )
        all_source_slices_in_merged_volume += slab_slices[:cutoff_slice_from_this_slab]

    total_slices_in_merged_volume = len(all_source_slices_in_merged_volume)
    print(f"Total slices: {total_slices_in_merged_volume}")
    zero_pad_len = len(str(total_slices_in_merged_volume))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Copy source slices to their destination with new slice number in the merged volume
    for i, source_slice_path in tqdm(
        list(enumerate(all_source_slices_in_merged_volume))
    ):
        dest_slice_path = out_dir / (str(i).zfill(zero_pad_len) + ".tif")
        shutil.copy(source_slice_path, dest_slice_path)
