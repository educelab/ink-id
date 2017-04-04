import numpy as np
import pdb

def edge(coordinate, subVolumeShape, shape):
    if coordinate < subVolumeShape: return True
    if coordinate > (shape - subVolumeShape): return True
    return False

def findEdgeSubVolume(args, xCoordinate, yCoordinate, volume, i):
    sample = np.zeros((args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    if edge(xCoordinate, args["x_Dimension"], volume.shape[1]) and not edge(yCoordinate, args["y_Dimension"], volume.shape[2]):
        if xCoordinate < args["x_Dimension"]:
            sample[args["x_Dimension"]-xCoordinate:args["x_Dimension"],:,:] = volume[i,0:xCoordinate,yCoordinate:yCoordinate+args["y_Dimension"],:]
        else:
            sample[0:volume.shape[1]-xCoordinate,:,:] = volume[i,xCoordinate:volume.shape[1],yCoordinate:yCoordinate+args["y_Dimension"],:]
    elif not edge(xCoordinate, args["x_Dimension"], volume.shape[1]) and edge(yCoordinate, args["y_Dimension"], volume.shape[2]):
        if yCoordinate < args["y_Dimension"]:
            try:
                sample[:,args["y_Dimension"]-yCoordinate:args["y_Dimension"],:] = volume[i,xCoordinate:xCoordinate+args["x_Dimension"],0:yCoordinate,:]
            except:
                pdb.set_trace()
                print
        else:
            sample[:,0:volume.shape[2]-yCoordinate,:] = volume[i,xCoordinate:xCoordinate+args["x_Dimension"],yCoordinate:volume.shape[2],:]
    else:
        if xCoordinate < args["x_Dimension"] and yCoordinate < args["y_Dimension"]: # TOP LEFT corner
            sample[args["x_Dimension"]-xCoordinate:args["x_Dimension"],\
                args["y_Dimension"]-yCoordinate:args["y_Dimension"] ,:]\
                    = volume[i,0:xCoordinate,0:yCoordinate,:]

        elif xCoordinate < args["x_Dimension"] and yCoordinate > args["y_Dimension"]: # BOTTOM LEFT corner
            sample[args["x_Dimension"]-xCoordinate:args["x_Dimension"],\
                0:volume.shape[2]-yCoordinate, :]\
                    = volume[i,0:xCoordinate,yCoordinate:volume.shape[2],:]

        elif xCoordinate > args["x_Dimension"] and yCoordinate < args["y_Dimension"]: # TOP RIGHT corner
            sample[0:volume.shape[1]-xCoordinate,\
                args["x_Dimension"]-yCoordinate:args["y_Dimension"], :]\
                    = volume[i,xCoordinate:volume.shape[1],0:yCoordinate,:]

        else: # BOTTOM RIGHT corner
            sample[0:volume.shape[1]-xCoordinate,\
                0:volume.shape[2]-yCoordinate, :]\
                    = volume[i,xCoordinate:volume.shape[1],yCoordinate:volume.shape[2],:]

    return sample

def bounds(args, shape, identifier):
    if identifier == 0: # TOP
        xBounds = [0, shape]
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
