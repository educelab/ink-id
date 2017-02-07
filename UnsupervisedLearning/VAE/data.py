import numpy as np
import pdb
import os
from PIL import Image
import math
import re

def readData(args):
    files = os.listdir(args["dataPath"])
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    volume = []
    for f in files:
        sliceData = np.array(Image.open(args["dataPath"]+f))
        volume.append(sliceData)
    volume = np.transpose(volume, (2,1,0))
    volume = (volume - np.min(volume)) / (np.amax(volume) - np.min(volume))

    x = int(math.floor(int(volume[:,0,0].shape[0])/args["x_Dimension"]))
    y = int(math.floor(int(volume[0,:,0].shape[0])/args["y_Dimension"]))
    z = int(math.floor(int(volume[0,0,:].shape[0])/args["z_Dimension"]))
    n_cubes =  z * x * y
    inputSubVolumes = np.empty((n_cubes, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    cubeCount = 0
    for i in range(0, volume.shape[0]-args["x_Dimension"], args["x_Dimension"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"], args["y_Dimension"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"], args["z_Dimension"]):
                inputSubVolumes[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                cubeCount += 1

    return x, y, z, inputSubVolumes

def saveSamples(args, sampleList):
    for i in range(sampleList.shape[0]): # iterate through layers
        for j in range(sampleList[i].shape[3]): # iterate through samples
            for k in range(sampleList[i].shape[2]): # iterate through slices
                sliceNumber = str(k).zfill(4)
                f = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/" + sliceNumber + ".jpg"
                if (np.amax(sampleList[i]) - np.min(sampleList[i])) != 0:
                    sliceIm = 255 * ((sampleList[i][:,:,k,j] - np.min(sampleList[i])) / (np.amax(sampleList[i]) - np.min(sampleList[i])))
                else:
                    sliceIm = sampleList[i][:,:,k,j]
                sliceIm = np.transpose(sliceIm, (1,0))
                im = Image.fromarray(sliceIm)
                im.convert('RGB').save(f)


# NOTE: code snippet for reconstructing input data -- for verification


    # cubeCount = 0
    # l1_x_Dimension = []
    # for i in range(x):
    #     l1_y_Dimension = []
    #     for j in range(y):
    #         l1_z_Dimension = []
    #         for k in range(z):
    #             cube = inputSubVolumes[cubeCount,:,:,:]
    #             l1_z_Dimension.append(cube)
    #             cubeCount += 1
    #             # pdb.set_trace()
    #         l1_y_Dimension.append(np.concatenate(l1_z_Dimension, axis=2))
    #     l1_x_Dimension.append(np.concatenate(l1_y_Dimension, axis=1))
    #
    # entireVolume = np.concatenate(l1_x_Dimension, axis=0)
    # entireVolume = np.transpose(entireVolume, (1,0,2))
    # for k in range(entireVolume.shape[2]): # iterate through slices
    #     sliceNumber = str(k).zfill(4)
    #     f = "/home/volcart/Desktop/test/" + sliceNumber + ".jpg"
    #     # pdb.set_trace()
    #     sliceIm = 255 * ( (entireVolume[:,:,k] - np.min(entireVolume)) / (np.amax(entireVolume) - np.min(entireVolume)) )
    #     im = Image.fromarray(sliceIm)
    #     im.convert('RGB').save(f)
    # pdb.set_trace()
