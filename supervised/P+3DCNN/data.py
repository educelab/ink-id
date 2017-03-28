import numpy as np
import pdb
import os
from PIL import Image
import math
import cv2
import re

class Volume:
    def __init__(self, args):
        if args["mulitpower"]:
            allDirectories = os.listdir(args["trainingDataPath"])
            layerDirectories = []
            for d in allDirectories:
                if "layers" in d:
                    layerDirectories.append(args["trainingDataPath"]+d)
        else:
            layerDirectories = [args["trainingDataPath"]]

        volume = []
        for l in layerDirectories:
            layerFiles = os.listdir(l)
            layerFiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

            currentVolume = []
            for f in layerFiles:
                sliceData = cv2.cvtColor(cv2.imread(l+"/"+f), cv2.COLOR_RGB2GRAY)
                currentVolume.append(sliceData)
            volume.append(np.array(currentVolume))

        self.volume = np.transpose(np.array(volume), (0, 3, 2, 1))

        self.groundTruth = cv2.cvtColor(cv2.imread(args["groundTruthFile"]), cv2.COLOR_RGB2GRAY)
        self.groundTruth = np.transpose(self.groundTruth, (1,0))

        self.predictionImage = np.zeros((int(self.volume.shape[1]/args["stride"]), int(self.volume.shape[2]/args["stride"])), dtype=np.uint8)

    def getTrainingSample(self, args):
        # grab training sample at random
        trainingSamples = []
        groundTruth = []
        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(self.volume.shape[1]-args["x_Dimension"])
            yCoordinate = np.random.randint(int(self.volume.shape[2]/2)-args["y_Dimension"])
            # yCoordinate = np.random.randint(self.volume.shape[1]-args["y_Dimension"])
            zCoordinate = 0

            groupedSamples = []
            for j in range(self.volume.shape[0]):
                groupedSamples.append(self.volume[j, xCoordinate:xCoordinate+args["x_Dimension"], \
                            yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

            trainingSamples.append(groupedSamples)

            no_ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] == 0)[0])
            ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] == 255)[0])

            classification = np.argmax([no_ink, ink])
            gt = [0,0]
            gt[classification] = 1.0
            groundTruth.append(gt)

        return np.transpose(np.array(trainingSamples), (0, 2, 3, 4, 1)), np.array(groundTruth)

    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        xCoordinate = startingCoordinates[0]
        yCoordinate = startingCoordinates[1]
        zCoordinate = startingCoordinates[2]
        predictionSamples = []
        coordinates = []
        count = 0
        while count < args["numCubes"]:
            if (xCoordinate + args["x_Dimension"]) > self.volume.shape[1]:
                xCoordinate = 0
                yCoordinate += args["stride"]
            if (yCoordinate + args["y_Dimension"]) > self.volume.shape[2]:
                # yCoordinate = 0
                break

            groupedSamples = []
            for i in range(self.volume.shape[0]):
                groupedSamples.append(self.volume[i, xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            predictionSamples.append(groupedSamples)

            coordinates.append([xCoordinate, yCoordinate])
            xCoordinate += args["stride"]
            count += 1

        return np.transpose(np.array(predictionSamples), (0, 2, 3, 4, 1)), np.array(coordinates), [xCoordinate, yCoordinate, zCoordinate]

    def emptyPredictionVolume(self, args):
        self.predictionImage = np.zeros((int(self.volume.shape[1]/args["stride"]), int(self.volume.shape[2]/args["stride"])), dtype=np.uint8)

    def reconstruct(self, args, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
            if np.argmax(samples[i,:]) == 1:
                # self.predictionImage[coordinates[i,0]+int(args["x_Dimension"]/2), \
                #         coordinates[i,1]+int(args["y_Dimension"]/2)] = 1.0
                self.predictionImage[int(coordinates[i,0]/args["stride"]), int(coordinates[i,1]/args["stride"])] = 255

    def savePredictionImage(self, args, epoch):
        # if (np.amax(self.predictionImage) - np.min(self.predictionImage)) != 0:
            # predictionImage = 65535 * ( (self.predictionImage.copy() - np.min(self.predictionImage)) / (np.amax(self.predictionImage) - np.min(self.predictionImage)) )
            # predictionImage = np.array(predictionImage, dtype=np.uint16)
        cv2.imwrite(args["savePredictionPath"] + str(epoch) + ".png", self.predictionImage)
