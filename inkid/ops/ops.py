"""Miscellaneous operations used in ink-id."""

import datetime
import inspect
import json
import sys
import os

from jsmin import jsmin
import numpy as np
import imageio

import inkid


def generator_from_iterator(iterator):
    """Take a Python iterator and create a generator function with it.

    E.g. given an iterator of (x, y) points this will create a
    generator function that can be passed to create a
    tf.data.Dataset.

    """
    def generator():
        for item in iterator:
            yield item
    return generator


def are_coordinates_within(p1, p2, distance):
    """Return if two points would have overlapping boxes.

    Given two (x, y) points and a distance, imagine creating squares
    with side lengths equal to that distance and centering them on
    each point. Return if the squares overlap at all.

    """
    (x1, y1) = p1
    (x2, y2) = p2
    return abs(x1 - x2) < distance and abs(y1 - y2) < distance


def save_volume_to_image_stack(volume, dirname):
    """Save a volume to a stack of .tif images.

    Given a volume as a np.array and a directory name, save the volume
    as a stack of .tif images in that directory, with filenames
    starting at 0 and going up to the z height of the volume.

    """
    os.makedirs(dirname)
    for z in range(volume.shape[0]):
        image = volume[z, :, :]
        image = image.astype(np.uint16)
        imageio.imsave(os.path.join(dirname, str(z) + '.tif'), image)        


def load_default_parameters():
    """Return the default network parameters for ink-id.

    Find the directory that the inkid package is loaded from, and then
    return the network parameters in parameters.json.

    https://stackoverflow.com/questions/247770/retrieving-python-module-path

    """
    return load_parameters_from_json(os.path.join(os.path.dirname(inspect.getfile(inkid)), 'parameters.json'))


def load_parameters_from_json(filename):
    """Return a dict of the parameters stored in a JSON file.

    Given a filename to a .json, remove the comments from that file
    and return a Python dictionary built from the JSON.

    """
    with open(filename, 'r') as f:
        # minify to remove comments
        minified = jsmin(str(f.read()))
        return json.loads(minified)['parameters']

    
def bounds(args, volume_shape, identifier, train_portion):
    """Return the bounds tuples for X and Y dimensions. Used in finding
    training vs. testing coordinates.
    """
    y_step = int(args["subvolume_shape"][1]/2)
    x_step = int(args["subvolume_shape"][0]/2)

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


def getRandomTestCoordinate(args, volume_shape):
    """DEPRECATED"""
    if args["use_grid_training"]:
        return getGridTestCoordinate(args, volume_shape)
    # else:
    #     return (np.random.randint(row_bounds[0], row_bounds[1]),
    #             np.random.randint(col_bounds[0], col_bounds[1]))

def augment_sample(sample, seed=None):
    """DEPRECATED"""
    augmented_sample = sample
    if seed is None:
        seed = np.random.randint(4)

    # ensure equal probability for each augmentation, including no augmentation
    # for flip: original, flip left-right, flip up-down, both, or none
    if seed == 0:
        augmented_sample = np.flip(augmented_sample, axis=0)
    elif seed == 1:
        augmented_sample = np.flip(augmented_sample, axis=1)
    elif seed == 2:
        augmented_sample = np.flip(augmented_sample, axis=0)
        augmented_sample = np.flip(augmented_sample, axis=1)
    elif seed == 3:
        pass

    # for rotate: original, rotate 90, rotate 180, or rotate 270
    augmented_sample = np.rot90(augmented_sample, k=seed, axes=(0,1))

    return augmented_sample



def getTrainCoordinate(args, colBounds, rowBounds, volume_shape):
    """DEPRECATED"""
    if args["use_grid_training"]:
        return getGridTrainCoordinate(args, volume_shape)
    else:
        return np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])


def getGridTrainCoordinate(args, volume_shape):
    """DEPRECATED"""
    row_step = int(args["subvolume_shape"][1]/2)
    col_step = int(args["subvolume_shape"][0]/2)
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
    """DEPRECATED"""
    row_step = int(args["subvolume_shape"][1]/2)
    col_step = int(args["subvolume_shape"][0]/2)

    n_rows = int(args["grid_n_squares"] / 2)
    voxels_per_row = int(volume_shape[0] / n_rows)
    row_number = int(args["grid_test_square"] / 2)
    row = np.random.randint(row_step+(voxels_per_row*row_number), (voxels_per_row*(row_number+1))-col_step)

    if args["grid_test_square"] % 2 == 0:
        # testing on the left side
        col = np.random.randint(int(args["subvolume_shape"][0]/2), int(volume_shape[1] / 2))
    else:
        # testing on the right side
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["subvolume_shape"][0]/2))

    return row,col


def isInTestSet(args, row_point, col_point, volume_shape, bounds_identifier, train_portion):
    """DEPRECATED"""
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
    """DEPRECATED"""
    # assume coordinate is at the center
    row_step = int(args["subvolume_shape"][1]/2)
    col_step = int(args["subvolume_shape"][0]/2)
    row_top = row_coordinate - row_step
    col_left = col_coordinate - col_step

    avg_truth = np.mean(ground_truth[row_top:row_top+args["subvolume_shape"][1], col_left:col_left+args["subvolume_shape"][0]])

    return avg_truth 


def generateCoordinatePool(args, volume_shape, ground_truth, surface_mask, train_bounds_identifier, train_portion):
    """DEPRECATED"""
    coordinates = []
    truth_label_value = np.iinfo(ground_truth.dtype).max
    rowStep = int(args["subvolume_shape"][1]/2)
    colStep = int(args["subvolume_shape"][0]/2)

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
            coordinates.append([row, col, label, augment_seed])

    return coordinates


def generateSurfaceApproximation(volume, area=3, search_increment=1):
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
    """DEPRECATED"""
    # alternatively, check if the maximum value in the vector crosses a threshold
    # for now, just check our mask
    rowStep = int(args["subvolume_shape"][1] / 2)
    colStep = int(args["subvolume_shape"][0] / 2)
    square = surfaceMask[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep]
    return np.size(square) > 0 and np.min(square) != 0


def minimumSurfaceInSample(args, row, col, surfaceImage):
    rowStep = int(args["subvolume_shape"][1] / 2)
    colStep = int(args["subvolume_shape"][0] / 2)

    square = surfaceImage[row-rowStep:row+rowStep, col-colStep:col+colStep]
    if np.size(square) == 0:
        return 0

    return np.min(square)


def getSpecString(args):
    """Return string with date, time, and subvolume size."""
    tm = datetime.datetime.today().timetuple()
    tmstring = ""
    for t in range(3):
        tmstring += str(tm[t])
        tmstring+= "-"
    tmstring += str(tm[3])
    tmstring += "h"

    specstring = "{}x{}x{}-".format(args["subvolume_shape"][0], args["subvolume_shape"][1], args["subvolume_shape"][2])
    specstring = specstring + tmstring

    return specstring
