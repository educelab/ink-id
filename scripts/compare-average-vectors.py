'''
compare-average-vectors.py
this script compares average vectors along aligned surface points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin

print("Initializing...")
fragment_threshold = 21500
step_back = 40
step_through = 90

truth_file = "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif"
ground_truth = tiff.imread(truth_file)
volume = np.load('/home/jack/devel/volcart/small-fragment-data/volume.npy')
output_pic = np.zeros(ground_truth.shape, dtype=np.uint16)

ink_vectors = []
fragment_vectors = []
ink_valls = []
fragment_valls = []

ink_count = 0
fragment_count = 0
error_count = 0

print("Iterating across fragment...")
for row in range(int(volume.shape[0])):
    for col in range(int(volume.shape[1])):
        # don't look at non-fragment points
        if np.max(volume[row, col]) < fragment_threshold:
            continue

        vector = volume[row,col]
        thresh_vect = np.where(vector > fragment_threshold, vector, 0)
        try:
            peak = argrelmax(thresh_vect)[0][0]
            valley = argrelmin(vector[:peak])[0][-1]
        except IndexError:
            # no peak/valley
            error_count += 1
            continue

        start = peak - step_back
        end = peak + step_through

        if 18200 < vector[valley]:
            output_pic[row,col] = 65535

        if start > 0 and end < volume.shape[2]:
            aligned_vect = vector[start:end]

            if ground_truth[row, col] == 0:
                fragment_vectors.append(aligned_vect)
                fragment_valls.append(vector[valley])
                fragment_count += 1
            else:
                ink_vectors.append(aligned_vect)
                ink_valls.append(vector[valley])
                ink_count += 1
    #progress
    if row % int(volume.shape[0] / 10) == 0:
        print("{:.2f} percent done".format(100*float(row)/volume.shape[0]))
        print("\t{} ink\n\t{} fragment\n\t{} nothing".format(
            ink_count, fragment_count, error_count))

fragment_vectors = np.array(fragment_vectors)
ink_vectors = np.array(ink_vectors)
print("totals: {} ink, {} fragment".format(ink_count, fragment_count))
print("fragment vectors: {}".format(fragment_vectors.shape))
print("ink vectors: {}".format(ink_vectors.shape))
avg_fragment = np.mean(fragment_vectors, axis=0)
avg_ink = np.mean(ink_vectors, axis=0)
print("avg ink shape: {}".format(avg_ink.shape))
print("avg fragment: {}".format(avg_fragment.shape))
x_vals = np.arange(0, ink_vectors.shape[1], 1)
tiff.imsave("output_pic.tif", output_pic)

f1 = plt.figure(1)
g1 = f1.add_subplot(111)
ink_hist, ink_bins = np.histogram(np.array(ink_valls), bins=np.arange(10000, 20000, 100))
centers = (ink_bins[1:] + ink_bins[:-1]) / 2
g1.plot(centers, ink_hist / sum(ink_hist), 'g.')
#g2 = f1.add_subplot(212, sharex=g1)
frag_hist, frag_bins = np.histogram(np.array(fragment_valls), bins=np.arange(10000, 20000, 100))
g1.plot(centers, frag_hist / sum(frag_hist), 'b.')

#plt.plot(x_vals, avg_fragment, color='b')
#plt.plot(x_vals, avg_ink, color='g')
plt.show()
