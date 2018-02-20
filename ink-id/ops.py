"""
Miscellaneous operations required by the 3DCNN.
"""

import datetime
import json
import sys
import os

from jsmin import jsmin
import numpy as np
import tifffile as tiff


def save_subvolume_to_image_stack(subvolume, dirname):
    os.makedirs(dirname)
    for i in range(subvolume.shape[2]):
        image = subvolume[:, :, i]
        # image = (65535 * image).astype(np.uint16)
        image = image.astype(np.uint16)
        tiff.imsave(os.path.join(dirname, str(i) + '.tif'), image)        
    

def load_parameters_from_json(filename):
    with open(filename, 'r') as f:
        # minify to remove comments
        minified = jsmin(str(f.read()))
        return json.loads(minified)['parameters']

    
def adjustDepthForWobble(args, rowCoord, colCoord, zCoordinate, angle, axes, volume_shape):
    if set(axes) == {0,2}:
        # plane of rotation = yz
        adjacent_length =  (volume_shape[0] / 2) - rowCoord
        offset = adjacent_length * np.tan(np.deg2rad(angle))
        newZ = int(zCoordinate + offset)

    elif set(axes) == {1,2}:
        # plane of rotation = xz
        adjacent_length = colCoord - (volume_shape[1] / 2)
        offset = adjacent_length * np.tan(np.deg2rad(angle))
        newZ = int(zCoordinate + offset)

    else:
        # either no wobble or rotation plane = xy (does not affect depth)
        newZ = zCoordinate

    return newZ


def bounds(args, volume_shape, identifier, train_portion):
    y_step = int(args["y_dimension"]/2)
    x_step = int(args["x_dimension"]/2)

    if args["use_grid_training"]:
        col_bounds = [x_step, volume_shape[1]-x_step]
        row_bounds = [y_step, volume_shape[0]-y_step]

    else:
        if identifier == 0: # TOP
            col_bounds = [x_step, volume_shape[1]-x_step]
            row_bounds = [y_step, int(volume_shape[0] * train_portion)-y_step]
        elif identifier == 1: # RIGHT
            col_bounds = [int(volume_shape[1] * (1-train_portion))+x_step, volume_shape[1]-x_step]
            row_bounds = [y_step, volume_shape[0]-y_step]
        elif identifier == 2: # BOTTOM
            col_bounds = [x_step, volume_shape[1]-x_step]
            row_bounds = [int(volume_shape[0] * (1-train_portion))+y_step, volume_shape[0]-y_step]
        elif identifier == 3: # LEFT
            col_bounds = [x_step, int(volume_shape[1] * train_portion)-x_step]
            row_bounds = [y_step, volume_shape[0]-y_step]
        else:
            print("Bound identifier not recognized")
            sys.exit(0)

    return row_bounds, col_bounds


def getRandomTestCoordinate(args, volume_shape, bounds_identifier, train_portion):
    if args["use_grid_training"]:
        return getGridTestCoordinate(args, volume_shape)
    else:
        return np.random.randint(row_bounds[0], row_bounds[1]), np.random.randint(col_bounds[0], col_bounds[1])


def getTrainCoordinate(args, colBounds, rowBounds, volume_shape):
    if args["use_grid_training"]:
        return getGridTrainCoordinate(args, volume_shape)
    else:
        return np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])


def getGridTrainCoordinate(args, volume_shape):
    row_step = int(args["y_dimension"]/2)
    col_step = int(args["x_dimension"]/2)
    row, col = np.random.randint(row_step, volume_shape[0]-row_step), np.random.randint(col_step, volume_shape[1]-col_step)

    found = False
    n_rows = int(args["grid_n_squares"] / 2)
    voxels_per_row = int(volume_shape[0] / n_rows)
    voxels_per_col = int(volume_shape[1] / 2)

    row_number = int(args["grid_test_square"] / 2)
    col_number = args["grid_test_square"] % 2

    while not found:
        if row not in range(row_number*voxels_per_row, (row_number+1)*voxels_per_row) and col in range(col_number*voxels_per_col, (col_number+1)*voxels_per_col):
            found = True
        else:
            row, col = np.random.randint(row_step, volume_shape[0]-row_step), np.random.randint(col_step, volume_shape[1]-col_step)

    return row, col


def getGridTestCoordinate(args, volume_shape):
    row_step = int(args["y_dimension"]/2)
    col_step = int(args["x_dimension"]/2)

    n_rows = int(args["grid_n_squares"] / 2)
    voxels_per_row = int(volume_shape[0] / n_rows)
    row_number = int(args["grid_test_square"] / 2)
    row = np.random.randint(row_step+(voxels_per_row*row_number), (voxels_per_row*(row_number+1))-col_step)

    if args["grid_test_square"] % 2 == 0:
        # testing on the left side
        col = np.random.randint(int(args["x_dimension"]/2), int(volume_shape[1] / 2))
    else:
        # testing on the right side
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["x_dimension"]/2))

    return row,col


def isInTestSet(args, row_point, col_point, volume_shape, bounds_identifier, train_portion):
    if args["use_grid_training"]:
        n_rows = int(args["grid_n_squares"] / 2)
        voxels_per_row = int(volume_shape[0] / n_rows)
        row_number = int(args["grid_test_square"] / 2)

        if args["grid_test_square"] % 2 == 0:
            return row_point in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and col_point < (volume_shape[1]/2)
        else:
            return row_point in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and col_point > (volume_shape[1]/2)

    else:
        if bounds_identifier == 0: # train top / test bottom
            return row_point > (volume_shape[0] * train_portion)
        elif bounds_identifier == 1: # train right / test left
            return col_point < (volume_shape[1] * (1 - train_portion))
        elif bounds_identifier == 2: # train bottom / test top
            return row_point < (volume_shape[0] * (1 - train_portion))
        elif bounds_identifier == 3: # train left / test right
            return col_point > (volume_shape[1] * train_portion)


def averageTruthInSubvolume(args, row_coordinate, col_coordinate, ground_truth):
    # assume coordinate is at the center
    row_step = int(args["y_dimension"]/2)
    col_step = int(args["x_dimension"]/2)
    row_top = row_coordinate - row_step
    col_left = col_coordinate - col_step

    avg_truth = np.mean(ground_truth[row_top:row_top+args["y_dimension"], col_left:col_left+args["x_dimension"]])

    return avg_truth 


def generateCoordinatePool(args, volume_shape, ground_truth, surface_mask, train_bounds_identifier, train_portion):
    coordinates = []
    ink_count = 0
    truth_label_value = np.iinfo(ground_truth.dtype).max
    rowStep = int(args["y_dimension"]/2)
    colStep = int(args["x_dimension"]/2)

    row_bounds, col_bounds = bounds(args, volume_shape, train_bounds_identifier, train_portion)
    print(" rowbounds: {}".format(row_bounds))
    print(" colbounds: {}".format(col_bounds))

    for row in range(row_bounds[0], row_bounds[1]):
        for col in range(col_bounds[0], col_bounds[1]):
            # Dang this if chain is embarassingly large
            if args["use_grid_training"]:
                if isInTestSet(args, row, col, volume_shape, train_bounds_identifier, train_portion):
                    continue

            if args["restrict_surface"] and not isOnSurface(args, row, col, surface_mask):
                continue

            label_avg = np.mean(ground_truth[row-rowStep:row+rowStep, col-colStep:col+colStep])

            # use to exclude ambiguous samples
            if args['truth_cutoff_low']*truth_label_value < label_avg < args['truth_cutoff_high']*truth_label_value:
                # don't use ambiguous samples
                continue


            label = round(ground_truth[row,col] / truth_label_value)
            augment_seed = np.random.randint(4)
            ink_count += label # 0 if less than .9
            coordinates.append([row, col, label, augment_seed])

    ink_portion = ink_count / len(coordinates)

    return coordinates


def getRandomBrick(args, volume, xCoordinate, yCoordinate):
    v_min = np.min(volume[yCoordinate, xCoordinate])
    v_max = np.max(volume[yCoordinate, xCoordinate])
    v_median = np.median(volume[yCoordinate, xCoordinate])
    low = v_median - args["random_range"]
    high = v_median + args["random_range"]
    sample = np.random.random([args["x_dimension"], args["y_dimension"], args["z_dimension"]])
    return ((high - low) * sample) + low


def augmentSample(args, sample, seed=None):
    augmentedSample = sample
    if seed is None:
        seed = np.random.randint(4)

    # ensure equal probability for each augmentation, including no augmentation
    # for flip: original, flip left-right, flip up-down, both, or none
    if seed == 0:
        augmentedSample = np.flip(augmentedSample, axis=0)
    elif seed == 1:
        augmentedSample = np.flip(augmentedSample, axis=1)
    elif seed == 2:
        augmentedSample = np.flip(augmentedSample, axis=0)
        augmentedSample = np.flip(augmentedSample, axis=1)
    #implicit: no flip if seed == 2

    # for rotate: original, rotate 90, rotate 180, or rotate 270
    augmentedSample = np.rot90(augmentedSample, k=seed, axes=(0,1))

    return augmentedSample


def generateSurfaceApproximation(args, volume, area=3, search_increment=1):
    surface_points = np.zeros((volume.shape[0:2]), dtype=np.int)
    for row in range(1, volume.shape[0], area):
        for col in range(1, volume.shape[1], area):
            max_sum_index = 0
            max_sum = 0
            for i in range(0, volume.shape[2]-50, search_increment):
                sum_from_i = np.sum(volume[row,col,i:i+50])
                if sum_from_i > max_sum:
                    max_sum_index = i
                    max_sum = sum_from_i
            surface_points[row-int(area/2):row+round(area/2), col-int(area/2):col+round(area/2)] = max_sum_index
            # | | | |
            # | |+| |
            # | | | |

            # fill in every blank around the + with the value at +
    return surface_points


def isOnSurface(args, rowCoordinate, colCoordinate, surfaceMask):
    # alternatively, check if the maximum value in the vector crosses a threshold
    # for now, just check our mask
    rowStep = int(args["y_dimension"] / 2)
    colStep = int(args["x_dimension"] / 2)
    square = surfaceMask[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep]
    return np.size(square) > 0 and np.min(square) != 0


def minimumSurfaceInSample(args, row, col, surfaceImage):
    rowStep = int(args["y_dimension"] / 2)
    colStep = int(args["x_dimension"] / 2)

    square = surfaceImage[row-rowStep:row+rowStep, col-colStep:col+colStep]
    if np.size(square) == 0:
        return 0

    return np.min(square)


def getSpecString(args):
    tm = datetime.datetime.today().timetuple()
    tmstring = ""
    for t in range(3):
        tmstring += str(tm[t])
        tmstring+= "-"
    tmstring += str(tm[3])
    tmstring += "h"

    specstring = "{}x{}x{}-".format(args["x_dimension"], args["y_dimension"], args["z_dimension"])
    specstring = specstring + tmstring

    return specstring
