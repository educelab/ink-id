import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import inkid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="+", help="dataset file(s)")
    args = parser.parse_args()

    regions = inkid.data.flatten_data_sources_list(args.dataset)

    # get default colors in matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [matplotlib.colors.to_rgb(color) for color in colors]
    colors = np.array(colors) * 255

    # track regions by mask
    regions_by_mask = dict()
    
    for region in regions:
        with open(region) as f:
            data = json.load(f)
            mask_path = data["mask"]
            mask_path = Path(region).parent / mask_path
            mask_path = mask_path.resolve()
            if mask_path not in regions_by_mask:
                regions_by_mask[mask_path] = []
            regions_by_mask[mask_path].append(region)

    color_idx = 0
    for mask_path, regions in regions_by_mask.items():
        mask_img = Image.open(mask_path)
        mask_img = np.array(mask_img)

        output_img = np.zeros(mask_img.shape + (3,), dtype=np.uint8)

        for region in regions:
            with open(region) as f:
                data = json.load(f)
                bounding_box = data["bounding_box"]
                left, top, right, bottom = bounding_box
                color = colors[color_idx]
                color_idx = (color_idx + 1) % len(colors)
                # only paint where mask is already nonzero
                output_img[top:bottom, left:right, :] = color * mask_img[top:bottom, left:right, None]
            # save output image
            output_path = Path(mask_path).stem + "_regions.png"
            Image.fromarray(output_img).save(output_path)



if __name__ == "__main__":
    main()
