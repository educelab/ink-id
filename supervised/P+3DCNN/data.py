import numpy as np
import pdb
import os
from PIL import Image
import math
import cv2
import re

import ops

class Volume:
    def __init__(self, args):
        if args["mulitpower"] == "true":
            allDirectories = os.listdir(args["trainingDataPath"])
            layerDirectories = []
            for d in allDirectories:
                if "layers" in d:
                    layerDirectories.append(args["trainingDataPath"]+d)
        else:
            layerDirectories = [args["singleScanPath"]]

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

        if args["cropX_low"] and args["cropX_high"]:
            self.volume = self.volume[:,args["cropX_low"]:args["cropX_high"],:,:]
            self.groundTruth = self.groundTruth[args["cropX_low"]:args["cropX_high"],:]
        if args["cropY_low"] and args["cropY_high"]:
            self.volume = self.volume[:,:,args["cropY_low"]:args["cropY_high"],:]
            self.groundTruth = self.groundTruth[:,args["cropY_low"]:args["cropY_high"]]

        self.predictionImage = np.zeros((int(self.volume.shape[1]/args["stride"]), int(self.volume.shape[2]/args["stride"])), dtype=np.uint8)

    def getTrainingSample(self, args, testSet=False, bounds=0):
        # grab training sample at random

        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        if testSet:
            xBounds, yBounds = ops.bounds(args, [self.volume.shape[1], self.volume.shape[2]], (bounds+2)%4)
        else:
            xBounds, yBounds = ops.bounds(args, [self.volume.shape[1], self.volume.shape[2]], bounds)

        trainingSamples = []
        groundTruth = []

        for i in range(args["numCubes"]):
            xCoordinate, yCoordinate, zCoordinate, label_avg = ops.findRandomCoordinates(args, xBounds, yBounds, self.volume, self.groundTruth)

            spectralSamples = []
            if ops.edge(xCoordinate, args["x_Dimension"], self.volume.shape[1]) or ops.edge(yCoordinate, args["y_Dimension"], self.volume.shape[2]):
                for j in range(self.volume.shape[0]):
                    sample = ops.findEdgeSubVolume(args, xCoordinate, yCoordinate, self.volume, j)
                    spectralSamples.append(sample)
            else:
                for j in range(self.volume.shape[0]):
                    spectralSamples.append(self.volume[j, xCoordinate:xCoordinate+args["x_Dimension"], \
                                yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

            trainingSamples.append(spectralSamples)

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
        xCoordinate = startingCoordinates[0]
        yCoordinate = startingCoordinates[1]
        zCoordinate = startingCoordinates[2]

        predictionSamples = np.zeros((args["predictBatchSize"], args["numChannels"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        coordinates = np.zeros((args["predictBatchSize"], 2), dtype=np.int)
        count = 0
        while count < args["predictBatchSize"]:
            if xCoordinate > self.volume.shape[1]:
                xCoordinate = 0
                yCoordinate += args["stride"]
            if yCoordinate > self.volume.shape[2]:
                # yCoordinate = 0
                break

            spectralSamples = []
            if ops.edge(xCoordinate, args["x_Dimension"], self.volume.shape[1]) or ops.edge(yCoordinate, args["y_Dimension"], self.volume.shape[2]):
                for i in range(self.volume.shape[0]):
                    sample = ops.findEdgeSubVolume(args, xCoordinate, yCoordinate, self.volume, i)
                    spectralSamples.append(sample)
            else:
                for i in range(self.volume.shape[0]):
                    spectralSamples.append(self.volume[i, xCoordinate:xCoordinate+args["x_Dimension"], \
                            yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

            predictionSamples[count,:,:,:,:] = spectralSamples
            coordinates[count] = [xCoordinate, yCoordinate]

            xCoordinate += args["stride"]
            count += 1

        return np.transpose(predictionSamples, (0,2,3,4,1)), coordinates, [xCoordinate, yCoordinate, zCoordinate]

    def totalPredictions(self, args):
        xSlides = (self.volume.shape[1] - args["x_Dimension"]) / args["stride"]
        ySlides = (self.volume.shape[2] - args["y_Dimension"]) / args["stride"]
        return int(xSlides * ySlides)

    def emptyPredictionVolume(self, args):
        self.predictionImage = np.zeros((int(self.volume.shape[1]/args["stride"]), int(self.volume.shape[2]/args["stride"])), dtype=np.uint8)

    def reconstruct(self, args, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
            if np.argmax(samples[i,:]) == 1:
                try:
                    self.predictionImage[int(coordinates[i,0]/args["stride"]), int(coordinates[i,1]/args["stride"])] = 255
                except:
                    pass

    def savePredictionImage(self, args, epoch):
        cv2.imwrite(args["savePredictionPath"] + str(epoch) + ".png", self.predictionImage)
