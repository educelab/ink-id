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
    def __init__(self, config):
        if config["mulitpower"] == True:
            allDirectories = os.listdir(config["trainingDataPath"])
            layerDirectories = []
            for d in allDirectories:
                if "layers" in d:
                    layerDirectories.append(config["trainingDataPath"]+d)
            layerDirectories.sort()
        else:
            layerDirectories = [config["trainingDataPath"]]

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

        self.groundTruth = cv2.cvtColor(cv2.imread(config["groundTruthFile"]), cv2.COLOR_RGB2GRAY)
        self.groundTruth = np.transpose(self.groundTruth, (1,0))

        if config["surface_segmentation"]:
            self.surfaceImage = cv2.cvtColor(cv2.imread(config["surfaceDataFile"]), cv2.COLOR_RGB2GRAY)
            self.surfaceImage = np.transpose(self.surfaceImage, (1,0))

        if config["crop"]:
            self.volume = self.volume[:,config["cropX_low"]:config["cropX_high"],config["cropY_low"]:config["cropY_high"],:]
            self.groundTruth = self.groundTruth[config["cropX_low"]:config["cropX_high"],config["cropY_low"]:config["cropY_high"]]
            if config["surface_segmentation"]:
                self.surfaceImage = self.surfaceImage[config["cropX_low"]:config["cropX_high"], config["cropY_low"]:config["cropY_high"]]

        # NOTE: to resample the entire volume & ground truth, uncomment the lines below
        # for i in range(self.volume.shape[0]):
        #     self.volume[i,:,:,:] = scipy.ndimage.interpolation.zoom(self.volume[i,:,:,:], config["scalingFactor"])
        # self.groundTruth = scipy.ndimage.interpolation.zoom(self.groundTruth, config["scalingFactor"])


    def getTrainingCoordinates(self, config, bounds=0, shuffle=True, testSet=False):
        #TODO set up for quadrant training
        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        if testSet:
            xBounds, yBounds = ops.bounds(config, [self.volume.shape[1], self.volume.shape[2]], (bounds+2)%4)
        else:
            xBounds, yBounds = ops.bounds(config, [self.volume.shape[1], self.volume.shape[2]], bounds)

        coordinates = []
        truth_label_value = np.amax(self.groundTruth)
        for x in range(xBounds[0], xBounds[1], config["stride"]):
            for y in range(yBounds[0], yBounds[1], config["stride"]):
                x_range = math.ceil(config["x_Dimension"]/config["scalingFactor"])
                y_range = math.ceil(config["y_Dimension"]/config["scalingFactor"])
                if not ops.edge(x, x_range, self.volume.shape[1]) and not ops.edge(y, y_range, self.volume.shape[2]):
                    xCoordinate2 = int(x + math.ceil(float(config["x_Dimension"]) * float(1/config["scalingFactor"])))
                    yCoordinate2 = int(y + math.ceil(float(config["y_Dimension"]) * float(1/config["scalingFactor"])))
                    label_avg = np.mean(self.groundTruth[x:xCoordinate2, y:yCoordinate2])
                    if 0.1*truth_label_value < label_avg < 0.9*truth_label_value:
                        continue
                    coordinates.append([x,y])

        if shuffle:
            np.random.shuffle(coordinates)
        return np.array(coordinates)


    def getRandomTestCoordinates(self, config, bounds=0):
        #TODO set up for quadrant training
        xBounds, yBounds = ops.bounds(config, [self.volume.shape[1], self.volume.shape[2]], (bounds+2)%4)
        coordinates = []
        truth_label_value = np.amax(self.groundTruth)
        for x in range(xBounds[0], xBounds[1], config["stride"]):
            for y in range(yBounds[0], yBounds[1], config["stride"]):
                x_range = math.ceil(config["x_Dimension"]/config["scalingFactor"])
                y_range = math.ceil(config["y_Dimension"]/config["scalingFactor"])
                if not ops.edge(x, x_range, self.volume.shape[1]) and not ops.edge(y, y_range, self.volume.shape[2]):
                    xCoordinate2 = int(x + math.ceil(float(config["x_Dimension"]) * float(1/config["scalingFactor"])))
                    yCoordinate2 = int(y + math.ceil(float(config["y_Dimension"]) * float(1/config["scalingFactor"])))
                    label_avg = np.mean(self.groundTruth[x:xCoordinate2, y:yCoordinate2])
                    if 0.1*truth_label_value < label_avg < 0.9*truth_label_value:
                        continue
                    coordinates.append([x,y])
        np.random.shuffle(coordinates)
        return np.array(coordinates)[0:config["batchSize"],:]


    def get2DPredictionCoordinates(self, config):
        x_resolution = self.volume.shape[1]
        y_resolution = self.volume.shape[2]

        coordinates = []
        for x in range(0,x_resolution,config["stride"]):
            for y in range(0,y_resolution,config["stride"]):
                coordinates.append([x,y])

        return np.array(coordinates)


    def get3DPredictionCoordinates(self, config):
        x_resolution = self.volume.shape[1]
        y_resolution = self.volume.shape[2]
        z_resolution = self.volume.shape[3]

        coordinates = []
        for x in range(0, x_resolution, config["stride"]):
            for y in range(0, y_resolution, config["stride"]):
                for z in range(0, z_resolution, config["stride"]):
                    coordinates.append([x,y,z])

        return np.array(coordinates)


    def getSamples(self, config, coordinates):
    ''' given a list of coordinates, retrieve samples from those coordinates'''
        trainingSamples = []
        groundTruth = []

        for i in range(coordinates.shape[0]):
            # a list of the samples at this coordinate, 1 for a single scan
            spectralSamples = []

            xCoordinate = coordinates[i][0]
            yCoordinate = coordinates[i][1]
            if coordinates[0,:].shape == 3: # NOTE: case where an [x,y,z] coordinate has been passed
                zCoordinate = coordinates[i,2]
            else:
                if config["surface_segmentation"]:
                    zCoordinate = self.surfaceImage[xCoordinate, yCoordinate] - config["surfaceCushion"]
                    if config["useJitter"]:
                        zCoordinate = np.maximum(0, zCoordinate +  np.random.randint(config["jitterRange"][0], config["jitterRange"][1]))
                else:
                    zCoordinate = 0

            xCoordinate2 = int(xCoordinate + math.ceil(float(config["x_Dimension"]) * float(1/config["scalingFactor"])))
            yCoordinate2 = int(yCoordinate + math.ceil(float(config["y_Dimension"]) * float(1/config["scalingFactor"])))
            zCoordinate2 = int(zCoordinate + math.ceil(float(config["z_Dimension"]) * float(1/config["scalingFactor"])))

            x = math.ceil(config["x_Dimension"]/config["scalingFactor"])
            y = math.ceil(config["y_Dimension"]/config["scalingFactor"])
            if ops.edge(xCoordinate, x, self.volume.shape[1]) or ops.edge(yCoordinate, y, self.volume.shape[2]):
                for j in range(self.volume.shape[0]):
                    sample = ops.findEdgeSubVolume(config, xCoordinate, xCoordinate2, yCoordinate, yCoordinate2, zCoordinate, zCoordinate2, self.volume, j)
                    if config["addAugmentation"]:
                        sample = ops.augmentSample(sample)
                    spectralSamples.append(sample)
            else:
                if config["addRandom"] and (np.random.randint(config["randomStep"]) == 0):
                    #TODO test set should never have random bricks
                    for j in range(self.volume.shape[0]):
                        sample = ops.getRandomBrick(config, np.median(self.volume[j, xCoordinate:xCoordinate2, yCoordinate:yCoordinate2, zCoordinate:zCoordinate2]))
                        spectralSamples.append(sample)
                    trainingSamples.append(spectralSamples)
                    groundTruth.append([1.0,0.0])
                    continue
                for j in range(self.volume.shape[0]):
                    sample = self.volume[j, xCoordinate:xCoordinate2, \
                                yCoordinate:yCoordinate2, zCoordinate:zCoordinate2]
                    sample = scipy.ndimage.interpolation.zoom(sample, config["scalingFactor"])
                    sample = ops.splice(sample, config)
                    if config["addAugmentation"]:
                        sample = ops.augmentSample(sample)
                    spectralSamples.append(sample)

            trainingSamples.append(spectralSamples)

            no_ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+config["x_Dimension"], \
                        yCoordinate:yCoordinate+config["y_Dimension"]] == 0)[0])
            ink = len(np.where(self.groundTruth[xCoordinate:xCoordinate+config["x_Dimension"], \
                        yCoordinate:yCoordinate+config["y_Dimension"]] == 255)[0])

            gt = [0,0]
            classification = np.argmax([no_ink, ink])
            gt[classification] = 1.0
            groundTruth.append(gt)

        return np.transpose(np.array(trainingSamples), (0, 2, 3, 4, 1)), np.array(groundTruth)


    def totalPredictions(self, config):
        xSlides = (self.volume.shape[1] - config["x_Dimension"]) / config["stride"]
        ySlides = (self.volume.shape[2] - config["y_Dimension"]) / config["stride"]
        return int(xSlides * ySlides)


    def initPredictionImages(self, config, num_images):
        self.predictionImages = []
        for i in range(num_images):
            self.predictionImages.append(np.zeros((int(self.volume.shape[1]/config["stride"]), int(self.volume.shape[2]/config["stride"])), dtype=np.uint8))


    def initPredictionVolumes(self, config, num_volumes):
        self.predictionVolumes = []
        for i in range(num_volumes):
            self.predictionVolumes.append(np.zeros((int(self.volume.shape[1]/config["stride"]), int(self.volume.shape[2]/config["stride"]), int(self.volume.shape[3]/config["stride"])), dtype=np.uint8))


    def reconstruct2D(self, config, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(len(self.predictionImages)):
            for j in range(coordinates.shape[0]):
                if np.argmax(samples[i][j,:]) == 1:
                    try:
                        #TODO use actual prediction value instead of 255
                        self.predictionImages[i][int(coordinates[j,0]/config["stride"]), int(coordinates[j,1]/config["stride"])] = 255
                    except:
                        pass


    def reconstruct3D(self, config, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(len(self.predictionVolumes)):
            for j in range(coordinates.shape[0]):
                if np.argmax(samples[i][j,:]) == 1:
                    try:
                        #TODO use actual prediction value instead of 255
                        self.predictionVolumes[i][int(coordinates[j,0]/config["stride"]), int(coordinates[j,1]/config["stride"]), int(coordinates[j,2]/config["stride"])] = 255
                    except:
                        pass


    def savePredictionImages(self, config, epoch):
        for i in range(len(self.predictionImages)):
            cv2.imwrite(config["savePredictionPath"] + "volume-" + str(i) + "-epoch-" + str(epoch) + ".png", self.predictionImages[i])


    def savePredictionVolumes(self, config, epoch):
        for i in range(len(self.predictionVolumes)):
            for j in range(self.predictionVolumes[i].shape[2]):
                cv2.imwrite(config["savePredictionPath"] + "volume-" + str(i) + "-slice-" + str(j) + "-epoch-" + str(epoch) + ".png", self.predictionVolumes[i][:,:,j])
