import numpy as np
import pdb
import os
from PIL import Image
import math

def inputRawData(args):
    dataFiles = os.listdir(args["trainingDataPath"])
    dataFiles.sort(key=lambda f: int(filter(str.isdigit, f)))

    volume = []
    for f in dataFiles:
        sliceData = np.array(Image.open(args["trainingDataPath"]+f))
        volume.append(sliceData)
    volume = np.transpose(volume, (2,1,0))

    x = int(math.floor(int(volume[:,0,0].shape[0]/2)/args["x_Dimension"]))
    y = int(math.floor(int(volume[0,:,0].shape[0])/args["y_Dimension"]))
    z = int(math.floor(int(volume[0,0,:].shape[0])/args["z_Dimension"]))
    n_cubes =  z * x * y

    coordinates = np.empty((n_cubes*2, 3), dtype=np.uint16)
    coordinateCount = 0

    dataSamples = np.empty((n_cubes, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    cubeCount = 0
    for i in range(0, int(volume.shape[0]/2)-args["x_Dimension"]+1, args["x_Dimension"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"]+1, args["y_Dimension"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"]+1, args["z_Dimension"]):
                dataSamples[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                coordinates[coordinateCount,0], coordinates[coordinateCount,1], coordinates[coordinateCount,2] = i, j, k
                cubeCount += 1
                coordinateCount += 1

    predictionSamples = np.empty((n_cubes, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    cubeCount = 0
    for i in range(int(volume.shape[0]/2), volume.shape[0]-args["x_Dimension"]+1, args["x_Dimension"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"]+1, args["y_Dimension"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"]+1, args["z_Dimension"]):
                predictionSamples[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                coordinates[coordinateCount,0], coordinates[coordinateCount,1], coordinates[coordinateCount,2] = i, j, k
                cubeCount += 1
                coordinateCount += 1

    # return dataSamples, predictionSamples, coordinates, volume.shape
    return predictionSamples, dataSamples, coordinates, volume.shape

def inputGroundTruth(args):

    # groundTruthSlice = np.genfromtxt(args["groundTruthFile"], delimiter=",")
    groundTruthSlice = np.array(Image.open(args["groundTruthFile"]))
    groundTruthSlice[groundTruthSlice == 255] = 1.0
    groundTruthSlice = np.fliplr(np.transpose(groundTruthSlice, (1,0)))

    x = int(math.floor(int(groundTruthSlice[:,0].shape[0]/2)/args["x_Dimension"]))
    y = int(math.floor(int(groundTruthSlice[0,:].shape[0])/args["y_Dimension"]))
    n_cubes =  x * y

    #TODO implement subVolumeStepSize
    dataGroundTruth = np.zeros((n_cubes, args["n_Classes"]))
    cubeCount = 0
    for i in range(0, int(groundTruthSlice.shape[0]/2)-args["x_Dimension"]+1, args["x_Dimension"]):
        for j in range(0, groundTruthSlice.shape[1]-args["y_Dimension"]+1, args["y_Dimension"]):
            if np.where(groundTruthSlice[i:i+args["x_Dimension"],j:j+args["y_Dimension"]] == 1.0)[0].shape[0] > (args["x_Dimension"]*args["y_Dimension"]*0.85):
                dataGroundTruth[cubeCount,0] = 1.0
            else:
                dataGroundTruth[cubeCount,1] = 1.0
            cubeCount = cubeCount + 1

    predictionGroundTruth = np.zeros((n_cubes, args["n_Classes"]))
    cubeCount = 0
    for i in range(int(groundTruthSlice.shape[0]/2), groundTruthSlice.shape[0]-args["x_Dimension"]+1, args["x_Dimension"]):
        for j in range(0, groundTruthSlice.shape[1]-args["y_Dimension"]+1, args["y_Dimension"]):
            if np.where(groundTruthSlice[i:i+args["x_Dimension"],j:j+args["y_Dimension"]] == 1.0)[0].shape[0] > (args["x_Dimension"]*args["y_Dimension"]*0.85):
                predictionGroundTruth[cubeCount,0] = 1.0
            else:
                predictionGroundTruth[cubeCount,1] = 1.0
            cubeCount = cubeCount + 1

    # return dataGroundTruth, predictionGroundTruth
    return predictionGroundTruth, dataGroundTruth

def inputData(args):

    dataSamples, predictionSamples, coordinates, volumeShape = inputRawData(args)
    dataGroundTruth, predictionGroundTruth = inputGroundTruth(args)

    randomIndices = np.arange(dataSamples.shape[0])
    np.random.shuffle(randomIndices)
    coordinatesIndices = np.concatenate((randomIndices, randomIndices+dataSamples.shape[0]))

    return dataSamples[randomIndices], predictionSamples[randomIndices], dataGroundTruth[randomIndices], predictionGroundTruth[randomIndices], coordinates[coordinatesIndices], volumeShape
