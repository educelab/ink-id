import numpy as np
import pdb
import os
from PIL import Image
import math
import cv2
import scipy.ndimage
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
            layerDirectories.sort()
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

        # for i in range(self.volume.shape[0]):
        #     self.volume[i,:,:,:] = scipy.ndimage.interpolation.zoom(self.volume[i,:,:,:], args["scalingFactor"])
        # self.groundTruth = scipy.ndimage.interpolation.zoom(self.groundTruth, args["scalingFactor"])

    def getTrainingCoordinates(self, args, bounds=0, shuffle=True, testSet=False):
        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        if testSet:
            xBounds, yBounds = ops.bounds(args, [self.volume.shape[1], self.volume.shape[2]], (bounds+2)%4)
        else:
            xBounds, yBounds = ops.bounds(args, [self.volume.shape[1], self.volume.shape[2]], bounds)

        coordinates = []
        for x in range(xBounds[0], xBounds[1]):
            for y in range(yBounds[0], yBounds[1]):
                coordinates.append([x,y])

        if shuffle:
            np.random.shuffle(coordinates)
        return np.array(coordinates)

    def getRandomTestCoordinates(self, args, bounds=0):
        xBounds, yBounds = ops.bounds(args, [self.volume.shape[1], self.volume.shape[2]], (bounds+2)%4)
        coordinates = []
        for x in range(xBounds[0], xBounds[1]):
            for y in range(yBounds[0], yBounds[1]):
                coordinates.append([x,y])
        np.random.shuffle(coordinates)
        return np.array(coordinates)[0:args["batchSize"],:]

    def getPredictionCoordinates(self):
        x_resolution = self.volume.shape[1]
        y_resolution = self.volume.shape[2]

        coordinates = []
        for x in range(x_resolution):
            for y in range(y_resolution):
                coordinates.append([x,y])

        return np.array(coordinates)

    def getSamples(self, args, coordinates):
        trainingSamples = []
        groundTruth = []

        for i in range(coordinates.shape[0]):

            xCoordinate = coordinates[i][0]
            yCoordinate = coordinates[i][1]
            zCoordinate = 0
            xCoordinate2 = int(xCoordinate + math.ceil(float(args["x_Dimension"]) * float(1/args["scalingFactor"])))
            yCoordinate2 = int(yCoordinate + math.ceil(float(args["y_Dimension"]) * float(1/args["scalingFactor"])))
            zCoordinate2 = int(zCoordinate + math.ceil(float(args["z_Dimension"]) * float(1/args["scalingFactor"])))

            spectralSamples = []
            x = math.ceil(args["x_Dimension"]/args["scalingFactor"])
            y = math.ceil(args["y_Dimension"]/args["scalingFactor"])
            if ops.edge(xCoordinate, x, self.volume.shape[1]) or ops.edge(yCoordinate, y, self.volume.shape[2]):
                for j in range(self.volume.shape[0]):
                    sample = ops.findEdgeSubVolume(args, xCoordinate, xCoordinate2, yCoordinate, yCoordinate2, zCoordinate, zCoordinate2, self.volume, j)
                    if args["experimentType"] == "multipower-single-channel":
                        trainingSamples.append(sample)
                    else:
                        spectralSamples.append(sample)
            else:
                for j in range(self.volume.shape[0]):
                    sample = self.volume[j, xCoordinate:xCoordinate2, \
                                yCoordinate:yCoordinate2, zCoordinate:zCoordinate2]
                    sample = scipy.ndimage.interpolation.zoom(sample, args["scalingFactor"])
                    sample = ops.splice(sample, args)
                    if args["experimentType"] == "multipower-single-channel":
                        trainingSamples.append(sample)
                    else:
                        spectralSamples.append(sample)

            if args["experimentType"] != "multipower-single-channel":
                trainingSamples.append(spectralSamples)

            no_ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] == 0)[0])
            ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"]] == 255)[0])

            gt = [0,0]
            classification = np.argmax([no_ink, ink])
            gt[classification] = 1.0

            if args["experimentType"] == "multipower-single-channel":
                for j in range(self.volume.shape[0]):
                    groundTruth.append(gt)
            else:
                groundTruth.append(gt)

        if args["experimentType"] == "multipower-single-channel":
            return np.expand_dims(np.array(trainingSamples), axis=4), np.array(groundTruth)
        else:
            return np.transpose(np.array(trainingSamples), (0, 2, 3, 4, 1)), np.array(groundTruth)

    def totalPredictions(self, args):
        xSlides = (self.volume.shape[1] - args["x_Dimension"]) / args["stride"]
        ySlides = (self.volume.shape[2] - args["y_Dimension"]) / args["stride"]
        return int(xSlides * ySlides)

    def initPredictionImages(self, args, num_images):
        self.predictionImages = []
        for i in range(num_images):
            self.predictionImages.append(np.zeros((int(self.volume.shape[1]/args["stride"]), int(self.volume.shape[2]/args["stride"])), dtype=np.uint8))

    def reconstruct(self, args, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(len(self.predictionImages)):
            for j in range(coordinates.shape[0]):
                if np.argmax(samples[i][j,:]) == 1:
                    try:
                        self.predictionImages[i][int(coordinates[j,0]/args["stride"]), int(coordinates[j,1]/args["stride"])] = 255
                    except:
                        pass

    def savePredictionImages(self, args, epoch):
        for i in range(len(self.predictionImages)):
            cv2.imwrite(args["savePredictionPath"] + "volume-" + str(i) + "-epoch-" + str(epoch) + ".png", self.predictionImages[i])
