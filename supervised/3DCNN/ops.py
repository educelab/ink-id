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

def graph(args, epoch, test_accs, test_losses, train_accs, train_losses, test_fps, train_fps):
    n_values_in_session = int(args["predict_step"] / args["display_step"])
    n_v = n_values_in_session

    plt.figure(1)
    plt.figure(figsize=(16,8))
    plt.clf()
    raw_loss = plt.subplot(321) # losses
    plt.title("Losses")
    axes = plt.gca()
    axes.set_ylim([0,np.median(test_losses)+ 2*(np.std(test_losses))])
    xs = np.arange(len(train_accs))
    plt.plot(train_losses, 'k.')
    plt.plot(test_losses, 'g.')
    plt.subplot(322, sharey=raw_loss) # losses, best-fit
    plt.title("Losses, fit")
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_losses, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_losses, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_losses, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_losses, 1))(xs), color='g')
    # the line of best fit for the last training portion
    #plt.plot(xs[:-n_v], np.poly1d(np.polyfit(xs[:-n_v], test_losses[:-n_v], 1))(xs[:-n_v]), color='g')
    raw_acc = plt.subplot(323) # accuracies
    plt.title("Accuracies")
    plt.plot(train_accs, 'k.')
    plt.plot(test_accs, 'g.')
    plt.subplot(324, sharey=raw_acc)
    plt.title("Accuracies, fit")
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 1))(xs), color='g')
    #plt.plot(xs[:-n_v], np.poly1d(np.polyfit(xs[:-n_v], test_accs[:-n_v], 1))(xs[:-n_v]), color='g')
    raw_prec = plt.subplot(325) # false positives
    plt.title("Ink Precision")
    plt.plot(train_fps, 'k.')
    plt.plot(test_fps, 'g.')
    plt.subplot(326, sharey=raw_prec)
    plt.title("Ink Precision, fit")
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_fps, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_fps, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_fps, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_fps, 1))(xs), color='g')
    #plt.plot(xs[:-n_v], np.poly1d(np.polyfit(xs[:-n_v], test_fps[:-n_v], 1))(xs[:-n_v]), color='g')
    plt.savefig(args["output_path"]+"/plots-{}.png".format(epoch))
    #plt.show()


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


def bounds(args, shape, identifier):
    yStep = int(args["y_Dimension"]/2)
    xStep = int(args["x_Dimension"]/2)

    if args["use_grid_training"]:
        colBounds = [xStep, shape[1]-xStep]
        rowBounds = [yStep, shape[0]-yStep]

    else:
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



def findRandomCoordinate(args, colBounds, rowBounds, groundTruth, surfaceImage, surfaceMask, volume, testSet=False):
    max_truth = np.iinfo(groundTruth.dtype).max
    if testSet:
        rowCoordinate, colCoordinate = getTestCoordinate(args, colBounds, rowBounds, volume.shape)
    else:
        rowCoordinate, colCoordinate = getTrainCoordinate(args, colBounds, rowBounds, volume.shape)

    rowStep = int(args["y_Dimension"]/2)
    colStep = int(args["x_Dimension"]/2)

    zCoordinate = 0
    label_avg = np.mean(groundTruth[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep])

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


    zCoordinate = max(0,surfaceImage[rowCoordinate+rowStep, colCoordinate+colStep] - args["surface_cushion"])
    return rowCoordinate, colCoordinate, zCoordinate, label_avg



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


'''
def getQuadTrainCoordinate(args, colBounds, rowBounds, volume_shape):
    row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
    found = False

    if args["train_quadrant"] == 0:
        # test top left
        while not found:
            if row < (volume_shape[0] / 2) and col < (volume_shape[1] / 2):
                row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
            else:
                found = True

    elif args["train_quadrant"] == 1:
        # test top right
        while not found:
            if row < (volume_shape[0] / 2) and col > (volume_shape[1] / 2):
                row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
            else:
                found = True

    elif args["train_quadrant"] == 2:
        # test bottom left
        while not found:
            if row > (volume_shape[0] / 2) and col < (volume_shape[1] / 2):
                row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
            else:
                found = True

    elif args["train_quadrant"] == 3:
        # test bottom right
        while not found:
            if row > (volume_shape[0] / 2) and col > (volume_shape[1] / 2):
                row, col = np.random.randint(rowBounds[0], rowBounds[1]), np.random.randint(colBounds[0], colBounds[1])
            else:
                found = True

    return row, col
'''


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

    row = np.random.randint(int(args["y_Dimension"]/2)+(voxels_per_row*row_number), (voxels_per_row*(row_number+1)))

    if args["grid_test_square"] % 2 == 0:
        # testing on the left side
        col = np.random.randint(int(args["x_Dimension"]/2), int(volume_shape[1] / 2))
    else:
        # testing on the right side
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["x_Dimension"]/2))

    return row,col


'''
def getQuadTestCoordinate(args, volume_shape):
    if args["train_quadrant"] == 0:
        # test top left
        row = np.random.randint(int(args["y_Dimension"]/2), int(volume_shape[0] / 2))
        col = np.random.randint(int(args["x_Dimension"]/2), int(volume_shape[1] / 2))

    elif args["train_quadrant"] == 1:
        # test top right
        row = np.random.randint(int(args["y_Dimension"]/2), int(volume_shape[0] / 2))
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["x_Dimension"]/2))

    elif args["train_quadrant"] == 2:
        # test bottom left
        row = np.random.randint(int(volume_shape[0] / 2), volume_shape[0]-int(args["y_Dimension"]/2))
        col = np.random.randint(int(args["x_Dimension"]/2), int(volume_shape[1] / 2))

    elif args["train_quadrant"] == 3:
        # test bottom right
        row = np.random.randint(int(volume_shape[0] / 2), volume_shape[0]-int(args["y_Dimension"]/2))
        col = np.random.randint(int(volume_shape[1] / 2), volume_shape[1]-int(args["x_Dimension"]/2))

    return row,col
'''


def isInTestSet(args, rowPoint, colPoint, volume_shape):
    if args["use_grid_training"]:
        n_rows = int(args["grid_n_squares"] / 2)
        voxels_per_row = int(volume_shape[0] / n_rows)
        row_number = int(args["grid_test_square"] / 2)

        if args["grid_test_square"] % 2 == 0:
            return rowPoint in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and colPoint < (volume_shape[1]/2)
        else:
            return rowPoint in range(voxels_per_row*row_number, voxels_per_row*(row_number+1)) and colPoint > (volume_shape[1]/2)
            '''
    elif args["use_quadrant_training"]:
        if args["train_quadrant"] == 0:
            # test top left
            return (rowPoint < int(volume_shape[0] / 2)) and (colPoint < int(volume_shape[1] / 2))

        elif args["train_quadrant"] == 1:
            # test top right
            return (rowPoint < int(volume_shape[0] / 2)) and (colPoint > int(volume_shape[1] / 2))

        elif args["train_quadrant"] == 2:
            # test bottom left
            return (rowPoint > int(volume_shape[0] / 2)) and (colPoint < int(volume_shape[1] / 2))

        elif args["train_quadrant"] == 3:
            # test bottom right
            return (rowPoint > int(volume_shape[0] / 2)) and (colPoint > int(volume_shape[1] / 2))
            '''
    else:
        if args["train_bounds"] == 0: # train top / test bottom
            return rowPoint > (volume_shape[0] * args["train_portion"])
        elif args["train_bounds"] == 1: # train right / test left
            return colPoint < (volume_shape[1] * (1 - args["train_portion"]))
        elif args["train_bounds"] == 2: # train bottom / test top
            return rowPoint < (volume_shape[0] * (1 - args["train_portion"]))
        elif args["train_bounds"] == 3: # train left / test right
            return colPoint > (volume_shape[1] * args["train_portion"])



def generateCoordinatePool(args, volume, rowBounds, colBounds, groundTruth, surfaceMask):
    print("Generating coordinate pool...")
    coordinates = []
    ink_count = 0
    truth_label_value = np.iinfo(groundTruth.dtype).max
    rowStep = int(args["y_Dimension"]/2)
    colStep = int(args["x_Dimension"]/2)

    for row in range(rowBounds[0], rowBounds[1]):
        for col in range(colBounds[0], colBounds[1]):
            # Dang this if chain is embarassingly large
            if args["use_grid_training"]:
                if isInTestSet(args,row,col, volume.shape):
                    continue

            if args["restrict_surface"] and not isOnSurface(args, row, col, surfaceMask):
                continue

            label_avg = np.mean(groundTruth[row-rowStep:row+rowStep, col-colStep:col+colStep])
            if .1*truth_label_value < label_avg < .9*truth_label_value:
                continue

            label = int(groundTruth[row,col] / truth_label_value)
            augment_seed = np.random.randint(4)
            ink_count += label # 0 if less than .9
            coordinates.append([row, col, label, augment_seed])

    ink_portion = ink_count / len(coordinates)

    # old, slow method of balancing samples
    '''
    if args["balance_samples"]:
        print("Initial coordinate pool is {:.3f} ink samples...".format(ink_portion))
        while ink_portion < .49:
            # delete random non-ink samples until balanced
            index = np.random.randint(len(coordinates))
            if coordinates[index][2] == 0:
                del(coordinates[index])
                ink_portion = ink_count / len(coordinates)
    '''
    print("Final pool coordinate is {:.3f} ink samples".format(ink_count / len(coordinates)))
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

    # for rotate: original, rotate 90, rotate 180, or rotate 270
    augmentedSample = np.rot90(augmentedSample, k=seed, axes=(0,1))

    return augmentedSample


def isOnSurface(args, rowCoordinate, colCoordinate, surfaceMask):
    # alternatively, check if the maximum value in the vector crosses a threshold
    # for now, just check our mask
    rowStep = int(args["y_Dimension"] / 2)
    colStep = int(args["x_Dimension"] / 2)
    square = surfaceMask[rowCoordinate-rowStep:rowCoordinate+rowStep, colCoordinate-colStep:colCoordinate+colStep]
    return np.size(square) > 0 and np.min(square) != 0


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
