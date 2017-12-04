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

pth = '/home/jack/devel/volcart/pherc2/oriented-scaled-cropped-slices/'
outpath = '/home/jack/devel/volcart/pherc2/'
THRESH = 21000
data_files = os.listdir(pth)
data_files.sort()
sample_im = np.array(Image.open(pth+data_files[0]))

# create the volume
surface_points = np.zeros((len(data_files), sample_im.shape[0]), dtype=np.uint16)
surface_valleys = np.zeros((len(data_files), sample_im.shape[0]), dtype=np.uint16)
print("Loading images...")
for i in range(len(data_files)):
    print("Loading image {} of {}...".format(i, len(data_files)))
    slice_data = np.array(Image.open(pth+data_files[i]))
    for col in range(slice_data.shape[0]):
        try:
            approx = np.where(slice_data[col] > THRESH)[0][0]
            surface_points[i, col] = approx
            empty = slice_data[col, :approx]
            surface_valleys[i, col] = argrelmin(empty)[0][-1]
        except:
            continue

print("Saving surface images...")
tiff.imsave(outpath+'cropped-surface-points.tif',surface_points)
tiff.imsave(outpath+'cropped-surface-valleys.tif',surface_valleys)
print("Done!")
