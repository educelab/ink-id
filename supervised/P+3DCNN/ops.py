import numpy as np
import pdb
import math
import scipy.ndimage

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
