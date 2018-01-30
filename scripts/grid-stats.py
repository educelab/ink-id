'''
grid-stats.py
Calculate statistics about a training configuration
'''

import numpy as np
from imageio import imread

gt_path = '/home/jack/devel/volcart/lunate-sigma/small-fragment-gt.tif'
mask_path = '/home/jack/devel/volcart/lunate-sigma/small-fragment-outline.tif'

n_rows = 5
n_cols = 2

gt = imread(gt_path)
mask = imread(mask_path)

row_height = gt.shape[0] / n_rows
col_width = gt.shape[1] / n_cols

for row in range(n_rows):
    for col in range(n_cols):
        cell_id = (row*n_cols)+col
        print("ID: {} ({}, {})".format(cell_id,row,col))

        col_bound_left = col*col_width
        row_bound_top = row*row_height

