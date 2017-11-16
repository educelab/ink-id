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



def bounds(args, shape, identifier):
    yStep = int(args["y_Dimension"]/2)
    xStep = int(args["x_Dimension"]/2)

    if identifier == 0: # TOP
        colBounds = [xStep, shape[1]-xStep]
        rowBounds = [yStep, int(shape[0] * args["train_portion"])-yStep]
    elif identifier == 1: # RIGHT
        colBounds = [int(shape[1] * args["train_portion"]), shape[1]-xStep]
        rowBounds = [yStep, shape[0]-yStep]
    elif identifier == 2: # BOTTOM
        colBounds = [xStep, shape[1]-xStep]
        rowBounds = [int(shape[0] * args["train_portion"]), shape[0]-yStep]
    elif identifier == 3: # LEFT
        colBounds = [xStep, int(shape[1] * args["train_portion"])-xStep]
        rowBounds = [yStep, shape[0]-yStep]
    else:
        print("Bound identifier not recognized")
        sys.exit(0)
    return rowBounds, colBounds



def findRandomCoordinate(args, colBounds, rowBounds, groundTruth, surfaceImage, volume):
    truth_label_value = np.iinfo(groundTruth.dtype).max
    col = np.random.randint(colBounds[0], colBounds[1])
    row = np.random.randint(rowBounds[0], rowBounds[1])
    rowStep = int(args["y_Dimension"]/2)
    colStep = int(args["x_Dimension"]/2)

    #uncomment for quadrant training
    #col = np.random.randint(colStep, int(volume.shape[1]/2))
    #row = np.random.randint(rowStep, int(volume.shape[0]/2))
    zCoordinate = np.maximum(0, surfaceImage[row+rowStep, col+colStep] - args["surface_cushion"])
    label = np.around(groundTruth[row-rowStep:row+rowStep, col-colStep:col+colStep] / truth_label_value).astype(np.int32)
    label_avg = np.mean(label)

    '''
    # in some cases, each coordinate should have equal chance of being ink or not ink
    if np.random.randint(2) == 1: # make it INK
        # make sure 90% of the ground truth in this block is ink
        while label_avg < (.5):
                #np.min(np.max(volume[row-rowStep:row:rowStep, col-colStep:col+colStep, :], axis=2)) < args["surface_threshold"]:
            col = np.random.randint(colBounds[0], colBounds[1])
            row = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = surfaceImage[row+rowStep, col+colStep] - args["surface_cushion"]
            label = np.around(groundTruth[row-rowStep:row+rowStep, col-colStep:col+colStep] / truth_label_value).astype(np.int32)
            label_avg = np.mean(label)

    else: # make it NON-INK
        # make sure 90% of the ground truth in this block is NON-ink
        while 0.5 > label_avg > (.1) or \
                (args["restrict_surface"] and np.min(np.max(volume[row-rowStep:row+rowStep, col-colStep:col+colStep, :], axis=2)) < args["surface_threshold"]):
            col = np.random.randint(colBounds[0], colBounds[1])
            row = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = surfaceImage[row+rowStep, col+colStep] - args["surface_cushion"]
            label = np.around(groundTruth[row-rowStep:row+rowStep, col-colStep:col+colStep] / truth_label_value).astype(np.int32)
            label_avg = np.mean(label)
            '''
    return row, col, zCoordinate, label



def generateCoordinatePool(args, volume, rowBounds, colBounds, groundTruth):
    print("Generating coordinate pool...")
    coordinates = []
    ink_count = 0
    truth_label_value = np.iinfo(groundTruth.dtype).max
    int_labels = np.around(groundTruth/ truth_label_value).astype(np.int32)
    rowStep = int(args["y_Dimension"]/2)
    colStep = int(args["x_Dimension"]/2)
    surf_maxes = np.max(volume, axis=2)
    on_surface = np.greater_equal(surf_maxes, args["surface_threshold"])

    for row in range(rowBounds[0], rowBounds[1]):
        for col in range(colBounds[0], colBounds[1]):
            if args["restrict_surface"] and False in on_surface[row-rowStep:row+rowStep, col-colStep:col+colStep]:
                continue

            label = int_labels[row-rowStep:row+rowStep, col-colStep:col+colStep]
            augment_seed = np.random.randint(4)
            ink_count += np.mean(label)
            coordinates.append([row, col, label, augment_seed])

    ink_portion = ink_count / len(coordinates)

    print("Final coordinate pool is {:.3f} ink samples".format(ink_count / len(coordinates)))
    return coordinates



def getRandomBrick(args, volume, xCoordinate, yCoordinate):
    v_min = np.min(volume[yCoordinate, xCoordinate])
    v_max = np.max(volume[yCoordinate, xCoordinate])
    v_median = np.median(volume[yCoordinate, xCoordinate])
    low = v_median - args["random_range"]
    high = v_median + args["random_range"]
    sample = np.random.random([args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
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


    '''# reverse surface, i.e. flip in the z direction, 50% of the time
    if seed % 2 == 0:
        augmentedSample = np.flip(augmentedSample, axis=2)
    '''
    # for rotate: original, rotate 90, rotate 180, or rotate 270
    augmentedSample = np.rot90(augmentedSample, k=seed, axes=(0,1))

    return augmentedSample


def getSpecString(args):
    tm = datetime.datetime.today().timetuple()
    tmstring = ""
    for t in range(3):
        tmstring += str(tm[t])
        tmstring+= "-"
    tmstring += str(tm[3])
    tmstring += "h"

    specstring = "{}x{}x{}-".format(args["x_Dimension"], args["y_Dimension"], args["z_Dimension"])
    specstring = specstring + tmstring

    return specstring


def grabSample(args, coordinates, volume, surfaceImage, trainingImage, trainingSamples, groundTruth, index, label_avg):
    #TODO
    return trainingSamples, groundTruth
