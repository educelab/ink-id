import numpy as np
import pdb
import os
from PIL import Image
import math
import tifffile as tiff


class Volume:
    def __init__(self, args):
        dataFiles = os.listdir(args["trainingDataPath"])
        dataFiles.sort()
        #dataFiles.sort(key=lambda f: int(filter(str.isdigit, f)))

        volume = []
        for f in dataFiles:
            sliceData = np.array(Image.open(args["trainingDataPath"]+f))
            volume.append(sliceData)
        self.volume = np.transpose(volume, (2,1,0))
        self.groundTruth = tiff.imread(args["groundTruthFile"])
        self.predictionImage = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)


    def getTrainingSample(self, args):
        xBounds=[0,int(self.volume.shape[0]/2)-args["x_Dimension"]]
        yBounds=[0,self.volume.shape[1]-args["y_Dimension"]]
        trainingSamples = []
        groundTruth = []
        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(xBounds[0], xBounds[1])
            yCoordinate = np.random.randint(yBounds[0], yBounds[1])
            zCoordinate = 0

            trainingSamples.append(self.volume[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

            # NOTE: this volume class assumes that there are three classifications
            #TODO: clean up
            no_ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] < 85)[0])
            papyrus = len(np.where(np.logical_and(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] > 85, \
                        self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] < 171))[0])
            ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] > 171)[0])

            classification = np.argmax([no_ink, papyrus, ink])
            gt = [0,0,0]
            gt[classification] = 1.0
            groundTruth.append(gt)

        return np.array(trainingSamples), np.array(groundTruth)


    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        xCoordinate = startingCoordinates[0]
        yCoordinate = startingCoordinates[1]
        zCoordinate = startingCoordinates[2]
        predictionSamples = []
        coordinates = []
        count = 0
        while count < args["numCubes"]:
            if (xCoordinate + args["x_Dimension"]) > self.volume.shape[0]:
                xCoordinate = 0
                yCoordinate += args["overlapStep"]
            if (yCoordinate + args["y_Dimension"]) > self.volume.shape[1]:
                # yCoordinate = 0
                break
            predictionSamples.append(self.volume[xCoordinate:xCoordinate+args["x_Dimension"], \
                    yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            coordinates.append([xCoordinate, yCoordinate])
            xCoordinate += args["overlapStep"]
            count += 1

        return np.array(predictionSamples), np.array(coordinates), [xCoordinate, yCoordinate, zCoordinate]


    def reconstruct(self, args, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
            if np.argmax(samples[i,:]) == 2:
                self.predictionImage[coordinates[i,0]:coordinates[i,0]+args["x_Dimension"], \
                        coordinates[i,1]:coordinates[i,1]+args["y_Dimension"]] += 1.0


    def savePredictionImage(self, args):
        predictionImage = 65535 * ( (self.predictionImage.copy() - np.min(self.predictionImage)) / (np.amax(self.predictionImage) - np.min(self.predictionImage)) )
        predictionImage = np.array(predictionImage, dtype=np.uint16)
        tiff.imsave(args["savePredictionPath"] + "image.tif", predictionImage)
