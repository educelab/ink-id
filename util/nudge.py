'''
nudge.py
create a nudged version of the volume
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import tifffile as tiff
from scipy.signal import argrelmax
from scipy.stats import norm

truth = np.load('/home/jack/devel/ink-id/small-fragment-data/volume-truth.npy')
vol = np.load('/home/jack/devel/ink-id/small-fragment-data/volume.npy')
output = np.zeros(vol.shape, dtype=np.uint16)
before = np.zeros(truth.shape, dtype=np.uint16)
after = np.zeros(truth.shape, dtype=np.uint16)


# parameters
loc = 0
scale = 2
increases = [16000]
neigh = 2
thresh = 20500
span = 4

# create the distribution array
distribute = [0.0] * (span+1)
for i in range(len(distribute)):
    distribute[i] = norm.pdf(i, loc, scale)


for increase in increases:
    # main loop
    for i in range(neigh, vol.shape[0] - neigh):
        for j in range(neigh, vol.shape[1] - neigh):
            vector = vol[i,j]
            truth_weight = np.mean(truth[i-neigh:i+neigh, j-neigh:j+neigh]) / 255

            # set everything below threshold to 0
            thresh_vect = np.where(vector > thresh, vector, 0)
            try:
                peak = argrelmax(thresh_vect)[0][0]
                before[i,j] = vector[peak]

                # nudge each point around the peak
                for x in range(peak - span, peak + span):
                    diff = abs(peak - x)
                    dist_weight = distribute[diff]
                    vector[x] += int(increase * truth_weight * dist_weight)

                output[i,j] = vector
                after[i,j] = vector[peak]

            except IndexError:
                pass
            
        print("finished row {} / {} for increase {}".format(i, vol.shape[0] - neigh, increase))

    np.save("volume-inc{}-span{}-scale{}".format(increase, span, scale), output)
    tiff.imsave("values-before.tif", before)
    tiff.imsave("values-after-{}.tif".format(int(distribute[0]*increase)), after)
