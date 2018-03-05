'''
simple-surface.py
A simplified version of the surface extraction script
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from scipy.signal import argrelmin

pth = '/home/jack/devel/volcart-data/flat-scroll-fragment/training_slices/'
outpath = '/home/jack/devel/volcart-data/flat-scroll-fragment/'
THRESH = 9000
data_files = os.listdir(pth)
data_files.sort()
sample_im = np.array(Image.open(pth+data_files[0]))

# create the volume
surface_points = np.zeros((len(data_files), sample_im.shape[0]), dtype=np.uint16)
surface_valleys = np.zeros((len(data_files), sample_im.shape[0]), dtype=np.uint16)
print("Loading images...")
total_count = 0
error_count = 0
for i in range(len(data_files)):
    print("Loading image {} of {}...".format(i, len(data_files)))
    slice_data = np.array(Image.open(pth+data_files[i]))
    for col in range(slice_data.shape[0]):
        total_count += 1
        try:
            approx = np.where(slice_data[col] > THRESH)[0][0]
            surface_points[i, col] = approx
            empty = slice_data[col, :approx]
            surface_valleys[i, col] = argrelmin(empty)[0][-1]
        except:
            error_count += 1
            print("No surface at slice {}, column {}".format(i,col))
            continue

print("Surface point found for {:.3f}% of the volume".format(error_count / total_count))
print("Saving surface images...")
tiff.imsave(outpath+'surface-points.tif',surface_points)
tiff.imsave(outpath+'surface-valleys.tif',surface_valleys)
print("Done!")

