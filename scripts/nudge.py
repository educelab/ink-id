'''
nudge.py
create a nudged version of the volume, increasing values at ink points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import tifffile as tiff
from scipy.signal import argrelmax
from scipy.stats import norm

truth = np.load('/home/jack/devel/volcart/small-fragment-data/volume-truth.npy')
vol = np.load('/home/jack/devel/volcart/small-fragment-data/volume.npy')
output = np.zeros(vol.shape, dtype=np.uint16)
before = np.zeros(truth.shape, dtype=np.uint16)
after = np.zeros(truth.shape, dtype=np.uint16)
cap = np.iinfo(vol.dtype).max
truth_value = np.max(truth)


# parameters
loc = 0
scale = 2
increase_percentages = np.arange(0, 1.5, .05)+.05
increase_decimals = increase_percentages / 100
neigh = 4
thresh = 20500
span = 4


# create the distribution array
distribute = [0.0] * (span+1)
for i in range(len(distribute)):
    # the initial distribution
    distribute[i] = norm.pdf(i, loc, scale)



for increase in increase_decimals:
    # re-initialize everything
    vol = np.load('/home/jack/devel/volcart/small-fragment-data/volume.npy')
    outvol = np.zeros(vol.shape, dtype=np.uint16)
    before = np.zeros(truth.shape, dtype=np.uint16)
    after = np.zeros(truth.shape, dtype=np.uint16)

    target_increase = increase * cap
    increase_parameter = (target_increase / distribute[0])
    # for example if the target increase is 1.0% (.010),
    # target_increase = 65535*.01 = 655.35
    # increase_parameter = 655.35 / .19 = 3285


    # main loop
    for i in range(neigh, vol.shape[0] - neigh):
        for j in range(neigh, vol.shape[1] - neigh):
            vector = vol[i,j]
            truth_weight = np.mean(truth[i-neigh:i+neigh, j-neigh:j+neigh]) / truth_value

            # set everything below threshold to 0
            thresh_vect = np.where(vector > thresh, vector, 0)
            try:
                peak = argrelmax(thresh_vect)[0][0]
                before[i,j] = vector[peak]

                # nudge each point around the peak
                for x in range(peak - span, peak + span):
                    diff = abs(peak - x)
                    dist_weight = distribute[diff]
                    vector[x] += int(increase_parameter * truth_weight * dist_weight)

                outvol[i,j] = vector
                after[i,j] = vector[peak]

            except IndexError:
                # for when no argrelmax exists
                pass
            
        print("finished row {} / {} for increase {}".format(i, vol.shape[0] - neigh, increase))

    np.save("/home/jack/devel/volcart/small-fragment-data/volume-nudged-{:.2f}%".format(
        increase*100), outvol)
    tiff.imsave("/home/jack/devel/volcart/output/values-before-nudge-{:.2f}%.tif".format(
        increase*100), before)
    tiff.imsave("/home/jack/devel/volcart/output/values-after-nudged-{:.2f}%.tif".format(
        increase*100), after)
