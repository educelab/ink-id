'''
nudge.py
create a nudged version of the volume, increasing values at ink points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.stats import norm

print("Initializing...")
#truth_mask = tiff.imread('/home/jack/devel/fall17/predictions/3dcnn/10-28-5h-square5/ink/prediction-iteration70000-depth0.tif')
truth_mask = tiff.imread('/home/jack/Desktop/10fold-results/spliced.tif')
surface_mask = tiff.imread('/home/jack/devel/volcart/small-fragment-outline.tif')
vol = np.load('/home/jack/devel/volcart/small-fragment-data/volume.npy')
volume_directory = ('/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/')
data_files = os.listdir(volume_directory)
data_files.sort()
volume = []
print("Loading slices...")
for f in data_files:
    slice_data = np.array(tiff.imread(volume_directory+f))
    volume.append(slice_data)
vol = np.array(volume)
print("Done loading slices...")

output_dir = '/home/jack/Desktop/nudge-'
output = np.zeros(vol.shape, dtype=np.uint16)
before = np.zeros(truth_mask.shape, dtype=np.uint16)
after = np.zeros(truth_mask.shape, dtype=np.uint16)
cap = np.iinfo(vol.dtype).max
vol_min = np.min(np.where(vol > 0, vol, cap))
vol_max = np.max(vol)
print("min: {}, max: {}".format(vol_min, vol_max))
vol_range = (vol_max - vol_min)
truth_value = np.max(truth_mask)


# parameters
loc = 0
scale = 4 # how much to stretch the curve, lower = taller curve, higher = shorter/wider
increase_percentages = np.array([20,])
increase_decimals = increase_percentages / 100
neigh = 2
thresh = 20500
reach_in = 10
reach_back = 4
span = max(reach_in, reach_back)
show_demo = False
shown_demo = True # set to False to display a sample graph

# create the distribution array
distribute = [0.0] * (span+1)
for i in range(len(distribute)):
    # the initial distribution
    distribute[i] = norm.pdf(i, loc, scale)


print("Beginning loop...")
for increase in increase_decimals:
    print("Bumping with increase {:.2f}".format(increase))
    # re-initialize everything
    volume = []
    for f in data_files:
        slice_data = np.array(tiff.imread(volume_directory+f))
        volume.append(slice_data)
    vol = np.array(volume)
    outvol = np.copy(vol)
    before = np.zeros(truth_mask.shape, dtype=np.uint16)
    after = np.zeros(truth_mask.shape, dtype=np.uint16)

    target_increase = increase * vol_range
    increase_parameter = (target_increase / distribute[0])
    # for example if the target increase is 1.0% (.010),
    # target_increase = 65535*.01 = 655.35
    # increase_parameter = 655.35 / .19 = 3285


    # main loop
    print("row range:{}".format(vol.shape[0] - neigh))
    print("col range:{}".format(vol.shape[1] - neigh))
    for i in range(48+neigh, vol.shape[0] - neigh - 48):
        for j in range(48+neigh, vol.shape[1] - neigh - 48):
            # don't bump any points off the surface
            if surface_mask[i,j] == 0:
                continue

            vector = vol[i,j]
            truth_weight = np.mean(truth_mask[i-neigh:i+neigh, j-neigh:j+neigh]) / truth_value

            # set everything below threshold to 0
            thresh_vect = np.where(vector > thresh, vector, 0)
            try:
                peak = argrelmax(thresh_vect)[0][0]
                before[i,j] = vector[peak]

                # nudge each point around the peak
                for x in range(peak - reach_back, peak):
                    diff = abs(peak - x)
                    proportion = float(diff) / float(reach_back)
                    aligned_index = int(proportion * span)
                    dist_weight = distribute[aligned_index]
                    vector[x] += int(increase_parameter * truth_weight * dist_weight)
                for x in range(peak, peak + reach_in):
                    diff = abs(peak - x)
                    dist_weight = distribute[diff]
                    vector[x] += int(increase_parameter * truth_weight * dist_weight)


                outvol[i,j] = vector
                after[i,j] = vector[peak]
                if show_demo and not shown_demo and truth_weight > .9:
                    xs = np.arange(vol.shape[2])
                    plt.plot(thresh_vect, color='b')
                    plt.plot(vector, color='g')
                    plt.show()
                    shown_demo = True

            except IndexError:
                # for when no argrelmax exists
                pass

        #progress update
        if (i % int((vol.shape[0] - neigh) / 10) == 0):
            print("finished rows 0 to {} out of {} for increase {:.2f}".format(
                i, vol.shape[0] - neigh, increase))

    # output
    current_output_dir = (output_dir + "{:.2f}%".format(increase * 100) + "/")
    try:
        os.mkdir(current_output_dir)
    except Exception:
        pass

    # 1: save the volume and surface images
    np.save(current_output_dir+"volume-nudged-{:.2f}%".format(
        increase*100), outvol)
    tiff.imsave(current_output_dir+"values-before-nudge-{:.2f}%.tif".format(
        increase*100), before)
    tiff.imsave(current_output_dir+"values-after-nudged-{:.2f}%.tif".format(
        increase*100), after)

    # 2: save the slices
    slice_dir = current_output_dir + "/slices/"
    try:
        os.mkdir(slice_dir)
    except Exception:
        pass

    for sl in range(outvol.shape[0]):
        zeros = len(str(sl))
        tiff.imsave(slice_dir+"slice" + "0000"[:4-zeros] + str(sl) + ".tif", outvol[sl])

    # 3: save the planet
    #TODO
