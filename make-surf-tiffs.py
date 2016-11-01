'''
make-surf-tiffs.py: load the volume, extract surface and save its voxels
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
from scipy.signal import argrelmax, argrelmin
import tifffile as tiff

def extract_surface(vect,length):
    to_return = np.zeros(length,dtype=np.uint16)
    thresh_vect = np.where(vect > 21500, vect, 0)
    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]
    # surface valley: the last of all valleys until first peak
    surf_vall = argrelmin(vect[:surf_peak])[0][-1]
    to_return[surf_vall:surf_peak] = vect[surf_vall:surf_peak]
    return to_return


volume_filename = "volume-21000.npy"
volume = np.load(volume_filename)
surfs = np.zeros(volume.shape, dtype=np.uint16)


for sl in range(volume.shape[0]):
    for v in range(volume.shape[1]):
        try:
            surfs[sl][v] = extract_surface(volume[sl][v], volume.shape[2])
        except Exception:
            print("no surface found at {}, {}".format(sl, v))


# save the pictures
for sl in range(surfs.shape[0]):
    tiff.imsave("small-fragment-data/surface-only-slices/slice"
            +"0000"[:(4-len(str(sl)))]+ str(sl), surfs[sl])
