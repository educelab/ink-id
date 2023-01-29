import argparse
import h5py

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-file", help="HDF file to read")
    parser.add_argument("--dataset", help="Dataset to read")
    args = parser.parse_args()

    with h5py.File(args.hdf_file, "r") as f:
        dset = f[args.dataset]
        print(f"min: {dset[:].min()}")
        print(f"max: {dset[:].max()}")

if __name__ == "__main__":
    main()