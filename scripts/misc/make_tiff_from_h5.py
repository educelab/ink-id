'''
load the trim_herc.h5 file and save it as .tif slices
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import h5py
import imageio
import skimage.io as io
import pdb


debug = False

file = h5py.File('small_fragment_data/trim_herc.h5', 'r')  #open read-only
data = file['5-AstraReconGpu-tomo']['data']

xmin = -0.0011387261
xmax = 0.0018620082

for i in range(0,len(data)):
#for i in range(400,410):
    #xmin = data[i].min()
    #xmax = data[i].max()
    a = 0
    b = 65535
    current = data[i]
    # delete values < 0
    # scraped off too much of the fragment
    #current = np.where(current > 0, current, np.nan)
    # put values into int range
    current = np.subtract(current,xmin)
    current = np.multiply(b,current)
    current = np.divide(current,(xmax-xmin))
    current = current.astype(np.uint16)
    current = np.nan_to_num(current)
    if(i == 0) and debug: print(current)
    if(i == 0) and debug: print(current)

    print("finished slice " + str(i) + " of " + str(len(data)-1))
    #all_slices.append(current)
    zeros = '0000'
    zeros = zeros[:(4-len(str(i)))]
    imageio.imsave("small_fragment_data/smooth_slices/slice" + zeros + str(i) + '.tif', current)

#save the 3d tif
#all_slices = np.asarray(all_slices)
#imageio.imsave("3d_output.tif", all_slices)

