import numpy as np
import pdb
import os
from PIL import Image
import math

def readData(args):
    files = os.listdir(args["dataPath"])
    files.sort()
    volume = []
    for f in files:
        sliceData = np.array(Image.open(args["dataPath"]+f))
        volume.append(sliceData)
    volume = np.array(volume)

    x = int(math.floor(int(volume[0,0,:].shape[0])/args["x_Dimension"]))
    y = int(math.floor(int(volume[0,:,0].shape[0])/args["y_Dimension"]))
    z = int(math.floor(int(volume[:,0,0].shape[0])/args["z_Dimension"]))
    n_cubes =  z * x * y
    inputSubVolumes = np.empty((n_cubes, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    cubeCount = 0
    for i in range(0, volume.shape[0]-args["x_Dimension"], args["x_Dimension"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"], args["y_Dimension"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"], args["z_Dimension"]):
                inputSubVolumes[cubeCount,:,:,:] = volume[i+args["x_Dimension"], j+args["y_Dimension"], k+args["z_Dimension"]]
                cubeCount += 1

    return inputSubVolumes
