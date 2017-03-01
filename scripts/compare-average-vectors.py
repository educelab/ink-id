'''
compare-average-vectors.py
this script compares average vectors along aligned surface points 
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax 

fragment_threshold = 21500
step_back = 30
step_through = 30

truth_file = "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif"
ground_truth = tiff.imread(truth_file)
volume = np.load('/home/jack/devel/volcart/small-fragment-data/volume.npy')

ink_vectors = []
fragment_vectors = []

ink_count = 0
fragment_count = 0

for row in range(volume.shape[0]):
    for col in range(volume.shape[1]):
        if np.max(volume[row, col]) < fragment_threshold:
            continue
        vector = volume[row,col]
        thresh_vect = np.where(vector > fragment_threshold, vector, 0)
        try:
            peak = argrelmax(thresh_vect)[0][0]
        except:
            # no peaks
            continue

        start = peak - step_back
        end = peak + step_through

        if start > 0 and end < volume.shape[2]:
            aligned_vect = vector[start:end]

            if ground_truth[row, col] == 0:
                fragment_vectors.append(aligned_vect)
                fragment_count += 1
            else:
                ink_vectors.append(aligned_vect)
                ink_count += 1

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

plt.plot(x_vals, avg_fragment, color='b')
plt.plot(x_vals, avg_ink, color='g')
plt.show()
