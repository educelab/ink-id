'''
splice.py
Input: prediction images for individual grid squares
Outpu: spliced prediction images
'''

import tifffile as tiff
import numpy as np

n_squares = 10
n_rows = int(n_squares / 2)
predictions = []

for i in range(n_squares):
    predictions.append(tiff.imread('{}.tif'.format(i)))
    print("Loaded {}.tif...".format(i))

rows = predictions[0].shape[0]
cols = predictions[0].shape[1]
output = np.zeros((rows, cols), dtype=np.uint16)
for i in range(n_squares):
    start_row = int(rows/n_rows)*int(i/2)
    end_row = int(rows/n_rows)*(int(i/2)+1)

    if i%2 == 0:
        start_col = 0
        end_col = int(cols/2)
    else:
        start_col = int(cols/2)
        end_col = cols

    print("Splicing square {}...".format(i))
    print(" rows {} to {}...".format(start_row, end_row))
    print(" cols {} to {}...".format(start_col, end_col))
    output[start_row:end_row, start_col:end_col] = \
            predictions[i][start_row:end_row, start_col:end_col]

print("Saving output image!")
tiff.imsave('spliced.tif', output)
