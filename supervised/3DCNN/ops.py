'''
ops.py
This file provides miscellaneous operations required by the 3DCNN
Used mainly by data
'''

import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt

def graph(args, epoch, test_accs, test_losses, train_accs, train_losses, test_fps, train_fps):
    plt.figure(1)
    plt.figure(figsize=(16,8))
    plt.clf()
    plt.subplot(321) # losses
    plt.title("Losses")
    axes = plt.gca()
    axes.set_ylim([0,np.median(test_losses)+ 2*(np.std(test_losses))])
    xs = np.arange(len(train_accs))
    plt.plot(train_losses, 'k.')
    plt.plot(test_losses, 'g.')
    plt.subplot(322) # losses, best-fit
    plt.title("Losses, fit")
    axes = plt.gca()
    axes.set_ylim([0,np.median(test_losses)+1])
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_losses, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_losses, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_losses, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_losses, 1))(xs), color='g')
    plt.subplot(323) # accuracies
    plt.title("Accuracies")
    plt.plot(train_accs, 'k.')
    plt.plot(test_accs, 'g.')
    plt.subplot(324)
    plt.title("Accuracies, fit")
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 1))(xs), color='g')
    plt.subplot(325) # false positives
    plt.title("False positives")
    plt.plot(train_fps, 'k.')
    plt.plot(test_fps, 'g.')
    plt.subplot(326)
    plt.title("False positives, fit")
    #plt.plot(xs, np.poly1d(np.polyfit(xs, train_fps, 2))(xs), color='k')
    #plt.plot(xs, np.poly1d(np.polyfit(xs, test_fps, 2))(xs), color='g')
    plt.plot(xs, np.poly1d(np.polyfit(xs, train_fps, 1))(xs), color='k')
    plt.plot(xs, np.poly1d(np.polyfit(xs, test_fps, 1))(xs), color='g')
    plt.savefig(args["savePredictionFolder"]+"plots-{}.png".format(epoch))
    #plt.show()



def bounds(args, shape, identifier):
    if identifier == 0: # TOP
        colBounds = [0, shape[1]-args["x_Dimension"]]
        rowBounds = [0, int(shape[0] * args["train_portion"])-args["y_Dimension"]]
    elif identifier == 1: # RIGHT
        colBounds = [int(shape[1] * args["train_portion"]), shape[1]-args["x_Dimension"]]
        rowBounds = [0, shape[0]-args["y_Dimension"]]
    elif identifier == 2: # BOTTOM
        colBounds = [0, shape[1]-args["x_Dimension"]]
        rowBounds = [int(shape[0] * args["train_portion"]), shape[0]-args["y_Dimension"]]
    elif identifier == 3: # LEFT
        colBounds = [0, int(shape[1] * args["train_portion"])-args["x_Dimension"]]
        rowBounds = [0, shape[0]-args["y_Dimension"]]
    else:
        print("Bound identifier not recognized")
        sys.exit(0)
    return rowBounds, colBounds



def findRandomCoordinate(args, colBounds, rowBounds, groundTruth, volume):
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
            label_avg = np.mean(groundTruth[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep])


    else: # make it NON-INK
        # make sure 90% of the ground truth in this block is NON-ink
        while label_avg > (.1*max_truth):
                #np.min(np.max(volume[yCoordinate-yStep:yCoordinate:yStep, xCoordinate-xStep:xCoordinate+xStep, :], axis=2)) < args["surfaceThresh"]:
            xCoordinate = np.random.randint(colBounds[0], colBounds[1])
            yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
            if args["predict3d"]:
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



def addRandomNoise(args, volume, coordinates, trainingSamples, groundTruth, index):
    #TODO
    return trainingSamples, groundTruth


def grabSample(args, coordinates, volume, surfaceImage, trainingImage, trainingSamples, groundTruth, index, label_avg):
    #TODO
    return trainingSamples, groundTruth
