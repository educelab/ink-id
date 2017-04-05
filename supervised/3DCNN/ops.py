'''
ops.py
This file provides miscellaneous operations required by the 3DCNN
Used mainly by data
'''

import numpy as np
import sys
import pdb
import datetime
import matplotlib.pyplot as plt

def graph(args, epoch, test_accs, test_losses, train_accs, train_losses, test_fps, train_fps):
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
    plt.savefig(args["savePredictionFolder"]+"plots-{}.png".format(epoch))
    #plt.show()



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
    max_truth = np.iinfo(groundTruth.dtype).max
    xCoordinate = np.random.randint(colBounds[0], colBounds[1])
    yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
    yStep = int(args["y_Dimension"]/2)
    xStep = int(args["x_Dimension"]/2)
    zCoordinate = 0
    label_avg = np.mean(groundTruth[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep])

    # each coordinate should have equal chance of being ink or not ink
    if np.random.randint(2) == 1: # make it INK
        # make sure 90% of the ground truth in this block is ink
        while label_avg < (.9*max_truth):
                #np.min(np.max(volume[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep, :], axis=2)) < args["surfaceThresh"]:
            xCoordinate = np.random.randint(colBounds[0], colBounds[1])
            yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = surfaceImage[yCoordinate+yStep, xCoordinate+xStep] - args["surfaceCushion"]
            label_avg = np.mean(groundTruth[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep])

    else: # make it NON-INK
        # make sure 90% of the ground truth in this block is NON-ink
        while label_avg > (.1*max_truth) or \
                (args["restrictSurface"] and np.min(np.max(volume[yCoordinate-yStep:yCoordinate+yStep, xCoordinate-xStep:xCoordinate+xStep, :], axis=2)) < args["surfaceThresh"]):
            xCoordinate = np.random.randint(colBounds[0], colBounds[1])
            yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = surfaceImage[yCoordinate+yStep, xCoordinate+xStep] - args["surfaceCushion"]
            if args["predictDepth"] > 1:
                # any zCoordinate on a non-ink point can still be classified as non-ink
                zCoordinate = np.random.randint(0, volume.shape[2] - args["z_Dimension"])
            label_avg = np.mean(groundTruth[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep])

    return xCoordinate, yCoordinate, zCoordinate, label_avg



def getRandomBrick(args, volume, xCoordinate, yCoordinate):
    v_min = np.min(volume[yCoordinate, xCoordinate])
    v_max = np.max(volume[yCoordinate, xCoordinate])
    v_median = np.median(volume[yCoordinate, xCoordinate])
    low = v_median - args["randomRange"]
    high = v_median + args["randomRange"]
    sample = np.random.random([args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
    return ((high - low) * sample) + low



def augmentSample(args, sample):
    augmentedSample = sample
    # ensure equal probability for each augmentation, including no augmentation

    # for flip: original, flip left-right, flip up-down, or both
    flip = np.random.randint(3)
    if flip == 0:
        augmentedSample = np.flip(augmentedSample, axis=0)
    elif flip == 1:
        augmentedSample = np.flip(augmentedSample, axis=1)
    elif flip == 2:
        augmentedSample = np.flip(augmentedSample, axis=0)
        augmentedSample = np.flip(augmentedSample, axis=1)

    # for rotate: original, rotate 90, rotate 180, or rotate 270
    rotate = np.random.randint(4)
    if rotate == 0:
        augmentedSample = np.rot90(augmentedSample, k=1, axes=(0,1))
    elif rotate == 1:
        augmentedSample = np.rot90(augmentedSample, k=2, axes=(0,1))
    elif rotate == 2:
        augmentedSample = np.rot90(augmentedSample, k=3, axes=(0,1))

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
