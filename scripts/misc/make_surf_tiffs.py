'''
make-surf-imageios.py: load the volume, extract surface and save its voxels
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import os
#from scipy.signal import argrelmax, argrelmin
import imageio
import matplotlib.pyplot as plt

def extract_surface(vect,length,thresh):
    to_return = np.zeros(length,dtype=np.uint16)
    thresh_vect = np.where(vect > thresh, vect, 0)

    nz = np.nonzero(thresh_vect)[0][0]
    # find first peak above threshold
    #surf_peak = argrelmax(thresh_vect)[0][0]
    # surface valley: the last of all valleys until first peak
    #surf_vall = argrelmin(vect[:surf_peak])[0][-1]
    #to_return[surf_peak-1:surf_peak+2] = vect[surf_vall:surf_peak]

    #to_return[nz-1:nz+2] = 65535
    to_return[nz] = 65535

    return (to_return, nz)


volume_filename = "/home/jack/devel/ink-id/small-fragment-data/volume.npy"
volume = np.load(volume_filename)
surfs = np.zeros(volume.shape, dtype=np.uint16)
approx_surfs = np.zeros(volume.shape, dtype=np.uint16)
threshes = [20500]
n = 2
degree = 32

for thresh in threshes:
    # create the directory to save pictures
    try:
        pic_dir = "/home/jack/devel/ink-id/small-fragment-data/surf-line-slices-thresh{}-deg{}".format(
                thresh, degree)
        os.mkdir(pic_dir)
    except Exception:
        pass


    # main loop
    for sl in range(volume.shape[0]):
        x_vals = np.arange(volume.shape[1])
        y_vals = np.zeros((volume.shape[1]), dtype=np.uint16)
        x_vals = []
        y_vals = []


        # extract the surface peaks
        for v in range(volume.shape[1]):
            try:
                surf_vect, ind = extract_surface(volume[sl][v], volume.shape[2], thresh)
                surfs[sl][v] = surf_vect
                if ind > 0 and ind < volume.shape[2]:
                    x_vals.append(v)
                    y_vals.append(ind)
            except Exception:
                pass


        # create line of best fit
        surf_line = np.poly1d(np.polyfit(x_vals, y_vals, degree))
        for v in range(volume.shape[1]):
            approx = int(surf_line(v))
            approx_surfs[sl][v][max(0,min(approx,approx_surfs.shape[2]-1))] = 65535
        if sl == 220:
            _ = plt.plot(x_vals, y_vals, '.', x_vals, surf_line(x_vals), '-')
            plt.show()


        # save picture
        print("saving picture for slice {}/{}".format(sl, approx_surfs.shape[0]))
        imageio.imsave(pic_dir + "/slice"
                +"0000"[:(4-len(str(sl)))]+ str(sl), approx_surfs[sl])



    '''
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
    '''



