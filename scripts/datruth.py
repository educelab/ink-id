'''
datruth.py
use an input mask to create a numpy truth array, save it
'''

import tifffile as tiff
import numpy as np

inpath = '/home/jack/devel/ink-id/small-fragment-data/registered/'
in_filename = 'thresh-gt-mask.tif'
outpath= '/home/jack/devel/ink-id/small-fragment-data/'
out_filename = 'volume-truth'

picture = tiff.imread(inpath + in_filename)

np.save(outpath + out_filename, picture)
print("saved new ground truth file to {}".format(outpath + out_filename))
