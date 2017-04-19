import numpy as np
import pdb
import math
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def graph(args, iteration, test_accs, test_losses, train_accs, train_losses, test_fps, train_fps):
    # n_values_in_session = int(args["predictStep"] / args["displayStep"])
    # n_v = n_values_in_session
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
    plt.savefig(args["savePredictionPath"]+"plots-{}.png".format(iteration))
    #plt.show()

    np.savetxt(args["savePredictionPath"]+"test_accs.csv", test_accs, delimiter=",")
    np.savetxt(args["savePredictionPath"]+"train_accs.csv", train_accs, delimiter=",")
    np.savetxt(args["savePredictionPath"]+"test_losses.csv", test_losses, delimiter=",")
    np.savetxt(args["savePredictionPath"]+"train_losses.csv", train_losses, delimiter=",")
    np.savetxt(args["savePredictionPath"]+"test_precs.csv", test_fps, delimiter=",")
    np.savetxt(args["savePredictionPath"]+"train_precs.csv", train_fps, delimiter=",")

def edge(coordinate, subVolumeShape, shape):
    if coordinate < subVolumeShape: return True
    if coordinate > (shape - subVolumeShape): return True
    return False

def findEdgeSubVolume(args, xCoordinate, xCoordinate2, yCoordinate, yCoordinate2, zCoordinate, zCoordinate2, volume, i):
    x = math.ceil(args["x_Dimension"]/args["scalingFactor"])
    y = math.ceil(args["y_Dimension"]/args["scalingFactor"])
    z = math.ceil(args["z_Dimension"]/args["scalingFactor"])
    sample = np.zeros((x, y, z))

    if edge(xCoordinate, x, volume.shape[1]) and not edge(yCoordinate, y, volume.shape[2]):
        if xCoordinate < x:
            xCoordinate2 = int(xCoordinate - math.ceil(float(xCoordinate) * float(1/args["scalingFactor"])))
            unscaledSample = volume[i,0:xCoordinate2,yCoordinate:yCoordinate2,zCoordinate:zCoordinate2]
            sample[x - xCoordinate2: x, :, :] = unscaledSample
        else:
            unscaledSample = volume[i, xCoordinate:volume.shape[1], yCoordinate:yCoordinate2, zCoordinate:zCoordinate2]
            sample[0:volume.shape[1]-xCoordinate,:,:] = unscaledSample

    elif not edge(xCoordinate, x, volume.shape[1]) and edge(yCoordinate, y, volume.shape[2]):
        if yCoordinate < y:
            yCoordinate2 = int(yCoordinate - math.ceil(float(yCoordinate) * float(1/args["scalingFactor"])))
            unscaledSample = volume[i,xCoordinate:xCoordinate2,0:yCoordinate2,zCoordinate:zCoordinate2]
            sample[:, y-yCoordinate2:y, :] = unscaledSample
        else:
            unscaledSample = volume[i, xCoordinate:xCoordinate2, yCoordinate:volume.shape[2], zCoordinate:zCoordinate2]
            sample[:,0:volume.shape[2]-yCoordinate,:] = unscaledSample

    else:
        if xCoordinate < x and yCoordinate < y: # TOP LEFT corner
            xCoordinate2 = int(xCoordinate - math.ceil(float(xCoordinate) * float(1/args["scalingFactor"])))
            yCoordinate2 = int(yCoordinate - math.ceil(float(yCoordinate) * float(1/args["scalingFactor"])))
            unscaledSample = volume[i,0:xCoordinate2,0:yCoordinate2,zCoordinate:zCoordinate2]
            sample[x-xCoordinate2:x, y-yCoordinate2:y, :] = unscaledSample

        elif xCoordinate < x and yCoordinate > y: # BOTTOM LEFT corner
            xCoordinate2 = int(xCoordinate - math.ceil(float(xCoordinate) * float(1/args["scalingFactor"])))
            unscaledSample = volume[i, 0:xCoordinate2, yCoordinate:volume.shape[2], zCoordinate:zCoordinate2]
            sample[x-xCoordinate2:x, 0:volume.shape[2]-yCoordinate, :] = unscaledSample

        elif xCoordinate > x and yCoordinate < y: # TOP RIGHT corner
            yCoordinate2 = int(yCoordinate - math.ceil(float(yCoordinate) * float(1/args["scalingFactor"])))
            unscaledSample = volume[i, xCoordinate:volume.shape[1], 0:yCoordinate2, zCoordinate:zCoordinate2]
            sample[0:volume.shape[1]-xCoordinate, y-yCoordinate2:y,:] = unscaledSample

        else: # BOTTOM RIGHT corner
            unscaledSample = volume[i, xCoordinate:volume.shape[1], yCoordinate:volume.shape[2], zCoordinate:zCoordinate2]
            sample[0:volume.shape[1]-xCoordinate, 0:volume.shape[2]-yCoordinate,:] = unscaledSample


    sample = scipy.ndimage.interpolation.zoom(sample, args["scalingFactor"])
    sample = splice(sample, args)
    return sample

def bounds(args, shape, identifier):
    if identifier == 0: # TOP
        xBounds = [0, shape[0]]
        yBounds = [0, int(shape[1]/2)]
    elif identifier == 1: # RIGHT
        xBounds = [int(shape[0]/2), shape[0]]
        yBounds = [0, shape[1]]
    elif identifier == 2: # BOTTOM
        xBounds = [0, shape[0]]
        yBounds = [int(shape[1]/2), shape[1]]
    elif identifier == 3: # LEFT
        xBounds = [0, int(shape[0]/2)]
        yBounds = [0, shape[1]]
    else:
        print("Bound identifier not recognized")
        sys.exit(0)
    return xBounds, yBounds

def findRandomCoordinates(args, xBounds, yBounds, volume, groundTruth):
    xCoordinate = np.random.randint(xBounds[0], xBounds[1])
    yCoordinate = np.random.randint(yBounds[0], yBounds[1])
    zCoordinate = 0 # TODO this will need to change when we do a per voxel prediction
    label_avg = np.mean(groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                yCoordinate:yCoordinate+args["y_Dimension"]])
    # use this loop to only train on the surface
    # and make sure 90% of the ground truth in the area is the same
    # while (np.min(np.max(self.volume[yCoordinate:yCoordinate+args["y_Dimension"], xCoordinate:xCoordinate+args["x_Dimension"]], axis=2)) < args["surfaceThresh"] or \
    while label_avg in range(int(.1*255), int(.9*255)):
        xCoordinate = np.random.randint(xBounds[0], xBounds[1])
        yCoordinate = np.random.randint(yBounds[0], yBounds[1])
        label_avg = np.mean(groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                yCoordinate:yCoordinate+args["y_Dimension"]])
    return xCoordinate, yCoordinate, zCoordinate, label_avg

def splice(sample, args):
    if sample.shape[0] != args["x_Dimension"]:
        sample = sample[0:args["x_Dimension"],:,:]
    if sample.shape[1] != args["y_Dimension"]:
        sample = sample[:,0:args["y_Dimension"],:]
    if sample.shape[2] != args["z_Dimension"]:
        sample = sample[:,:,0:args["z_Dimension"]]
    return sample

def customReshape(args, batchX):
    num_batches = int(batchX.shape[0] / args["numVolumes"])

    out_batch = np.zeros((num_batches, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numVolumes"]))
    count = 0
    for i in range(0,batchX.shape[0],args["numVolumes"]):
        for j in range(args["numVolumes"]):
            out_batch[count,:,:,:,j] = batchX[i+j,:,:,:,0]
        count += 1

    return out_batch
