'''
ops.py
This file provides miscellaneous operations required by the 3DCNN
Used mainly by data
'''

import numpy as np
import sys
import pdb
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




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
    yStep = int(args["y_dimension"]/2)
    xStep = int(args["x_dimension"]/2)

    if args["use_grid_training"]:
        colBounds = [xStep, volume_shape[1]-xStep]
        rowBounds = [yStep, volume_shape[0]-yStep]

    else:
        if identifier == 0: # TOP
            colBounds = [xStep, volume_shape[1]-xStep]
            rowBounds = [yStep, int(volume_shape[0] * train_portion)-yStep]
        elif identifier == 1: # RIGHT
            colBounds = [int(volume_shape[1] * train_portion), volume_shape[1]-xStep]
            rowBounds = [yStep, volume_shape[0]-yStep]
        elif identifier == 2: # BOTTOM
            colBounds = [xStep, volume_shape[1]-xStep]
            rowBounds = [int(volume_shape[0] * train_portion), volume_shape[0]-yStep]
        elif identifier == 3: # LEFT
            colBounds = [xStep, int(volume_shape[1] * train_portion)-xStep]
            rowBounds = [yStep, volume_shape[0]-yStep]
        else:
            print("Bound identifier not recognized")
            sys.exit(0)

    return rowBounds, colBounds



def findRandomCoordinate(args, colBounds, rowBounds, groundTruth, surfaceImage, surfaceMask, volume_shape, testSet=False):
    max_truth = np.iinfo(groundTruth.dtype).max

    rowStep = int(args["y_dimension"]/2)
    colStep = int(args["x_dimension"]/2)
    if args["use_grid_training"]:
        if testSet:
            rowCoordinate, colCoordinate = getGridTestCoordinate(args, colBounds, rowBounds, volume_shape)
        else:
            rowCoordinate, colCoordinate = getGridTrainCoordinate(args, colBounds, rowBounds, volume_shape)
    else:
        rowCoordinate, colCoordinate = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])


    if args['restrict_surface']:
        while np.min(surfaceMask[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep]) == 0:
            if args["use_grid_training"]:
                if testSet:
                    rowCoordinate, colCoordinate = getGridTestCoordinate(args, colBounds, rowBounds, volume_shape)
                else:
                    rowCoordinate, colCoordinate = getGridTrainCoordinate(args, colBounds, rowBounds, volume_shape)
            else:
                rowCoordinate, colCoordinate = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])

    label_avg = np.mean(groundTruth[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep])
    zCoordinate = max(0, surfaceImage[rowCoordinate, colCoordinate] - args["surface_cushion"])
    return rowCoordinate, colCoordinate, zCoordinate, label_avg

    '''
    # each coordinate should have equal chance of being ink or not ink
    if np.random.randint(2) == 1: # make it INK
        # make sure 90% of the ground truth in this block is ink
        while label_avg < (.9*max_truth):
                #np.min(np.max(volume[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep, :], axis=2)) < args["surface_threshold"]:
            if testSet:
                rowCoordinate, colCoordinate = getTestCoordinate(args, colBounds, rowBounds, volume.shape)
            else:
                rowCoordinate, colCoordinate = getTrainCoordinate(args, colBounds, rowBounds, volume.shape)
            label_avg = np.mean(groundTruth[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep])

    else: # make it NON-INK
        # make sure 90% of the ground truth in this block is NON-ink
        while label_avg > (.1*max_truth) or \
                (args["restrict_surface"] and np.min(surfaceMask[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep]) == 0):
            if testSet:
                rowCoordinate, colCoordinate = getTestCoordinate(args, colBounds, rowBounds, volume.shape)
            else:
                rowCoordinate, colCoordinate = getTrainCoordinate(args, colBounds, rowBounds, volume.shape)
            label_avg = np.mean(groundTruth[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep])
    '''



def getTestCoordinate(args, colBounds, rowBounds, volume_shape):
    if args["use_grid_training"]:
        return getGridTestCoordinate(args, colBounds, rowBounds, volume_shape)
    else:
        return np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])



def getTrainCoordinate(args, colBounds, rowBounds, volume_shape):
    if args["use_grid_training"]:
        return getGridTrainCoordinate(args, colBounds, rowBounds, volume_shape)
    else:
        return np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])




def getGridTrainCoordinate(args, colBounds, rowBounds, volume_shape):
    row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
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
            row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])

    return row, col


def getGridTestCoordinate(args, colBounds, rowBounds, volume_shape):
    #TODO check validity of test square
    n_rows = int(args["grid_n_squares"] / 2)
    voxels_per_row = int(volume_shape[0] / n_rows)
    row_number = int(args["grid_test_square"] / 2)

    row = np.random.randint(int(args["y_dimension"]/2)+(voxels_per_row*row_number), (voxels_per_row*(row_number+1)))

    if args["grid_test_square"] % 2 == 0:
        # testing on the left side
        col = np.random.randint(int(args["x_dimension"]/2), int(volume_shape[1] / 2))
    else:
        # testing on the right side
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["x_dimension"]/2))

    return row,col




def isInTestSet(args, rowPoint, colPoint, volume_shape, train_bounds, train_portion):
    if args["use_grid_training"]:
        n_rows = int(args["grid_n_squares"] / 2)
        voxels_per_row = int(volume_shape[0] / n_rows)
        row_number = int(args["grid_test_square"] / 2)

        if args["grid_test_square"] % 2 == 0:
            return rowPoint in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and colPoint < (volume_shape[1]/2)
        else:
            return rowPoint in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and colPoint > (volume_shape[1]/2)

    else:
        if train_bounds == 0: # train top / test bottom
            return rowPoint > (volume_shape[0] * train_portion)
        elif train_bounds == 1: # train right / test left
            return colPoint < (volume_shape[1] * (1 - train_portion))
        elif train_bounds == 2: # train bottom / test top
            return rowPoint < (volume_shape[0] * (1 - train_portion))
        elif train_bounds == 3: # train left / test right
            return colPoint > (volume_shape[1] * train_portion)



def generateCoordinatePool(args, volume, rowBounds, colBounds, groundTruth, surfaceMask, train_bounds, train_portion):
    coordinates = []
    ink_count = 0
    truth_label_value = np.iinfo(groundTruth.dtype).max
    rowStep = int(args["y_dimension"]/2)
    colStep = int(args["x_dimension"]/2)

    print(" rowbounds: {}".format(rowBounds))
    print(" colbounds: {}".format(colBounds))

    for row in range(rowBounds[0], rowBounds[1]):
        for col in range(colBounds[0], colBounds[1]):
            # Dang this if chain is embarassingly large
            if args["use_grid_training"]:
                if isInTestSet(args,row,col, volume.shape, train_bounds, train_portion):
                    continue

            if args["restrict_surface"] and not isOnSurface(args, row, col, surfaceMask):
                continue

            label_avg = np.mean(groundTruth[row-rowStep:row+rowStep, col-colStep:col+colStep])

            # use to exclude ambiguous samples
            if .1*truth_label_value < label_avg < .9*truth_label_value:
                continue


            label = round(groundTruth[row,col] / truth_label_value)
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
            for i in range(0, volume.shape[2]-10, search_increment):
                sum_from_i = np.sum(volume[row,col,i:i+10])
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
