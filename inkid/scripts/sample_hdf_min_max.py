import argparse
import h5py
import numpy as np
import random

from scipy.ndimage import gaussian_filter


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-file", help="HDF file to read")
    parser.add_argument("--dataset", help="Dataset to read")
    args = parser.parse_args()

    with h5py.File(args.hdf_file, "r") as f:
        dset = f[args.dataset]
        print(f"Dataset shape: {dset.shape}")
        slices, height, width = dset.shape

        global_raw_min = np.inf
        global_raw_max = -np.inf
        global_blurred_min = np.inf
        global_blurred_max = -np.inf

        for i in range(slices):
            z = random.randint(0, slices - 1)
            img = dset[z]
            blurred_img = gaussian_filter(img, 3)

            raw_min = np.amin(img)
            raw_max = np.amax(img)
            blurred_min = np.amin(blurred_img)
            blurred_max = np.amax(blurred_img)

            update = False
            if raw_min < global_raw_min:
                global_raw_min = raw_min
                update = True
            if raw_max > global_raw_max:
                global_raw_max = raw_max
                update = True
            if blurred_min < global_blurred_min:
                global_blurred_min = blurred_min
                update = True
            if blurred_max > global_blurred_max:
                global_blurred_max = blurred_max
                update = True

            if update:
                print(f"Slices processed: {i}", flush=True)
                print(f"Raw min: {global_raw_min}", flush=True)
                print(f"Raw max: {global_raw_max}", flush=True)
                print(f"Blurred min: {global_blurred_min}", flush=True)
                print(f"Blurred max: {global_blurred_max}", flush=True)

        print(f"Raw min: {global_raw_min}")
        print(f"Raw max: {global_raw_max}")
        print(f"Blurred min: {global_blurred_min}")
        print(f"Blurred max: {global_blurred_max}")


if __name__ == "__main__":
    main()
