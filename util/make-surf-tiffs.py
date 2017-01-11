'''
make-surf-tiffs.py: load the volume, extract surface and save its voxels
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import os
#from scipy.signal import argrelmax, argrelmin
import tifffile as tiff

def extract_surface(vect,length,thresh):
    to_return = np.zeros(length,dtype=np.uint16)
    thresh_vect = np.where(vect > thresh, vect, 0)

    nz = np.nonzero(thresh_vect)[0][0]
    # find first peak above threshold
    #surf_peak = argrelmax(thresh_vect)[0][0]
    # surface valley: the last of all valleys until first peak
    #surf_vall = argrelmin(vect[:surf_peak])[0][-1]
    #to_return[surf_peak-1:surf_peak+2] = vect[surf_vall:surf_peak]

    to_return[nz-1:nz+2] = 65535

    return to_return


volume_filename = "volume.npy"
volume = np.load(volume_filename)
surfs = np.zeros(volume.shape, dtype=np.uint16)
threshes = [20000, 20100, 20200, 20300, 20400, 20500]
n = 2

for thresh in threshes:
    stdev = np.zeros(volume.shape[0], dtype=np.float64)
    for sl in range(volume.shape[0]):
        for v in range(volume.shape[1]):
            try:
                surf_vect = extract_surface(volume[sl][v], volume.shape[2], thresh)
                surfs[sl][v] = surf_vect
            except Exception:
                pass

    for sl in range(n, volume.shape[0]-n-1):
        inkspots = []
        for v in range(n, volume.shape[1]-n-1):
            for i in range(sl-n, sl+n+1):
                for j in range(v-n, v+n+1):
                    try:
                        for num in np.nonzero(surfs[i][j])[0]:
                            inkspots.append(num)
                    except Exception:
                        pass
        inkspots = np.array(inkspots)
        stdev[sl] += np.std(inkspots)

    # save the pictures
    print("thresh {}".format(thresh))
    print("stdev : {}".format(stdev))
    print("total stdev : {}".format(np.sum(stdev)))
    
    print("saving pictures for thresh {}".format(thresh))
    try:
        pic_dir = "small-fragment-data/surface-thresh{}-only-slices".format(thresh)
        os.mkdir(pic_dir)
    except Exception:
        pass
    for sl in range(surfs.shape[0]):
        tiff.imsave(pic_dir + "/slice"
                +"0000"[:(4-len(str(sl)))]+ str(sl), surfs[sl])
    


