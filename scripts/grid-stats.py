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

total_ink_points = np.count_nonzero(gt)
total_frag_points = np.count_nonzero(mask)
print("{} total ink points".format(total_ink_points))
print("{} total points on fragment".format(total_frag_points))
print("{:.2f} of the fragment is ink".format(total_ink_points / total_frag_points))

row_height = gt.shape[0] / n_rows
col_width = gt.shape[1] / n_cols

for row in range(n_rows):
    for col in range(n_cols):
        cell_id = (row*n_cols)+col
        #print("ID: {} ({} {})".format(cell_id,row,col))

        col_bound_left = int(col*col_width)
        col_bound_right = int(col_bound_left + col_width)
        row_bound_top = int(row*row_height)
        row_bound_bottom = int(row_bound_top + row_height)

        gt_cell = gt[row_bound_top:row_bound_bottom, col_bound_left:col_bound_right]
        mask_cell = mask[row_bound_top:row_bound_bottom, col_bound_left:col_bound_right]

        cell_ink_count = np.count_nonzero(gt_cell)
        cell_frag_count = np.count_nonzero(mask_cell)

        #print("Ink in cell: {} out: {}".format(cell_ink_count, total_ink_points - cell_ink_count))
        #print("Frag in cell: {} out: {}".format(cell_frag_count, total_frag_points - cell_frag_count))
        print("{}({:.2f}) {}({:.2f}) {}({:.2f}) {}({:.2f})".format(
            cell_ink_count, cell_ink_count / total_ink_points,
            cell_frag_count, cell_frag_count / total_frag_points,
            total_ink_points - cell_ink_count, 1-(cell_ink_count / total_ink_points),
            total_frag_points - cell_frag_count, 1-(cell_frag_count / total_frag_points)))



