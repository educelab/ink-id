import argparse
import h5py
import numpy as np

from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-file", help="HDF file to read")
    parser.add_argument("--dataset", help="Dataset to read")
    args = parser.parse_args()

    with h5py.File(args.hdf_file, "r") as f:
        dset = f[args.dataset]
        print(f"Dataset shape: {dset.shape}")
        chunk_size = 10
        min_value = np.inf
        max_value = -np.inf

        for i in tqdm(range(0, dset.shape[0], chunk_size), desc="Processing HDF"):
            chunk = dset[i:i + chunk_size]
            min_value = min(min_value, chunk.min())
            max_value = max(max_value, chunk.max())

        print(f"min: {min_value}, max: {max_value}")

if __name__ == "__main__":
    main()