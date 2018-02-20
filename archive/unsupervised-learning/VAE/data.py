'''
data.py:
    - input training data
    - save captured data from network
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"

import numpy as np
import pdb
import os
from PIL import Image
import math
import re
import cv2

class Volume:
    def __init__(self, args, networkType="unsupervised"):
        # input the volume
        dataFiles = os.listdir(args["trainingDataPath"])
        dataFiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        volume = []
        for f in dataFiles:
            sliceData = np.array(Image.open(args["trainingDataPath"]+f))
            volume.append(sliceData)
        self.volume = np.transpose(volume, (2,1,0))
        self.volume = (self.volume - np.min(self.volume)) / (np.amax(self.volume) - np.min(self.volume))

    def getVolumeShape(self):
        return self.volume.shape

    def getTrainingSample(self, args):
        trainingSamples = []
        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(self.volume.shape[0]-args["x_Dimension"])
            yCoordinate = np.random.randint(self.volume.shape[1]-args["y_Dimension"])
            zCoordinate = np.random.randint(self.volume.shape[2]-args["z_Dimension"])

            trainingSamples.append(self.volume[xCoordinate:xCoordinate+args["x_Dimension"], \
                        yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

        return np.array(trainingSamples)

    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        xCoordinate = startingCoordinates[0]
        yCoordinate = startingCoordinates[1]
        zCoordinate = startingCoordinates[2]
        predictionSamples = []
        coordinates = []
        for i in range(args["numCubes"]):
            if (xCoordinate + args["x_Dimension"]) > self.volume.shape[0]:
                xCoordinate = 0
                yCoordinate += args["overlapStep"]
            if (yCoordinate + args["y_Dimension"]) > self.volume.shape[1]:
                yCoordinate = 0
                zCoordinate += args["overlapStep"]
            if (zCoordinate + args["z_Dimension"] > self.volume.shape[2]):
                break
            predictionSamples.append(self.volume[xCoordinate:xCoordinate+args["x_Dimension"], \
                    yCoordinate:yCoordinate+args["y_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            coordinates.append([xCoordinate, yCoordinate, zCoordinate])
            xCoordinate += args["overlapStep"]

        return np.array(predictionSamples), np.array(coordinates), [xCoordinate, yCoordinate, zCoordinate]


class PredictionVolume:
    def __init__(self, args, volume):
        self.shape = volume.getVolumeShape()
        self.predictionVolume = np.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype=np.float64)
        self.counter = np.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype=np.float64)

    def reconstruct(self, args, samples, coordinates, sampleRate=1):
        for i in range(coordinates.shape[0]):
            x = int(coordinates[i,0] / sampleRate)
            y = int(coordinates[i,1] / sampleRate)
            z = int(coordinates[i,2] / sampleRate)
            self.predictionVolume[x:x+samples.shape[1],y:y+samples.shape[2],z:z+samples.shape[3]] += samples[i,:,:,:]
            self.counter[x:x+samples.shape[1],y:y+samples.shape[2],z:z+samples.shape[3]] += 1

    def trimZeros(self):
        xClip, yClip, zClip = self.shape[0], self.shape[1], self.shape[2]
        for i in range(self.counter.shape[0]):
            if self.counter[i,0,0] == 0.0:
                xClip = i
                break
        for i in range(self.counter.shape[1]):
            if self.counter[0,i,0] == 0.0:
                yClip = i
                break
        for i in range(self.counter.shape[2]):
            if self.counter[0,0,i] == 0.0:
                zClip = i
                break

        self.predictionVolume = self.predictionVolume[0:xClip-1,0:yClip-1,0:zClip-1]
        self.counter = self.counter[0:xClip-1,0:yClip-1,0:zClip-1]
        print("Prediction volume shape = " + str(self.predictionVolume.shape))

    def savePredictionVolume(self, args):
        # TODO: find library to save 3D images
        predictionVolume = self.predictionVolume / self.counter
        predictionVolume = np.array(predictionVolume, dtype=np.uint16)
        pdb.set_trace()
        cv2.imwrite(args["save3DVolumePath"]+"3DPrediction.jpg", predictionVolume)

    def savePredictionSlices(self, args, layerNum, sampleNum):
        predictionVolume = self.predictionVolume / self.counter
        predictionVolume = 255*(predictionVolume - np.min(predictionVolume)) / (np.amax(predictionVolume) - np.min(predictionVolume))
        predictionVolume = np.array(predictionVolume, dtype=np.uint8)
        try:
            os.mkdir(args["saveSamplesPath"] + str(layerNum))
        except:
            pass
        self.slicesPath = args["saveSamplesPath"] + str(layerNum) + "/" + str(sampleNum)
        os.mkdir(self.slicesPath)
        for i in range(predictionVolume.shape[2]):
            predictionSlice = predictionVolume[:,:,i]
            sliceNumber = str(i).zfill(4)
            cv2.imwrite(self.slicesPath+"/"+str(sliceNumber)+".jpg", predictionSlice)

    def savePredictionVideo(self, args):
        # TODO: os.system() returning false for some reason?
        videoFile = self.slicesPath + "/0.mp4"
        sliceFiles = self.slicesPath + "/%04d.jpg"
        os.system("ffmpeg -y -framerate 10 -start_number 0 -i " + sliceFiles + " -vcodec mpeg4 " + videoFile)
