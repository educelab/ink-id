"""
resample_volume.py


"""

import pickle
import numpy as np
from scipy import signal
import tifffile as tiff
from os import mkdir


def main():

    data_path = "/home/jack/devel/ink-id/small-fragment-data"
    output_path = data_path + "/resample-slices"

    try:
        mkdir(output_path)
    except Exception:
        pass

    vol = pickle.load(open("new_volume.pickled","r"))
    lengths = pickle.load(open("lengths.pickled","r"))
    max_length = max(lengths)
    num_slices = len(vol)
    slice_length = len(vol[0])
    new_vol = []

    for i in range(num_slices):
        new_slice = np.zeros((slice_length,max_length), dtype = np.int16)
        for v in range(slice_length):
            if len(vol[i][v]) != 0:
                new_slice[v] = signal.resample(vol[i][v],max_length)

        slice_name = output_path+"slice"+"0000"[:4-len(str(i))] + str(i) + ".tif"
        tiff.imsave(slice_name, new_slice)
        new_vol.append(new_slice)
        print("finished slice {} of {}".format(i,num_slices))
    
    pickle.dump(new_vol,open("resampled_volume.pickled","w"))



if __name__ == "__main__":
    main()
