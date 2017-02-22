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

def readData(args):
    files = os.listdir(args["dataPath"])
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    volume = []
    for f in files:
        sliceData = np.array(Image.open(args["dataPath"]+f))
        volume.append(sliceData)
    volume = np.transpose(volume, (2,1,0))

    # uncomment below line to normalize input (0,1)
    # volume = (volume - np.min(volume)) / (np.amax(volume) - np.min(volume))

    nCubesX = int(math.floor(int(volume.shape[0] - args["x_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesY = int(math.floor(int(volume.shape[1] - args["y_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesZ = int(math.floor(int(volume.shape[2] - args["z_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesTotal = nCubesX * nCubesY * nCubesZ
    inputSubVolumes = np.empty((nCubesTotal, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    overlappingCoordinates = np.empty((nCubesTotal, 3))
    cubeCount = 0
    for i in range(0, volume.shape[0]-args["x_Dimension"]+1, args["subVolumeStepSize"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"]+1, args["subVolumeStepSize"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"]+1, args["subVolumeStepSize"]):
                inputSubVolumes[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                overlappingCoordinates[cubeCount,0] = i
                overlappingCoordinates[cubeCount,1] = j
                overlappingCoordinates[cubeCount,2] = k
                cubeCount += 1

    return overlappingCoordinates, inputSubVolumes

def saveSamples(args, overlappingCoordinates, sampleList):
    sampleRates = [1, 2, 4, 8, 4, 2]
    for i in range(len(sampleList)): # iterate through layers
        print("Saving layer number: " + str(i+1))
        for j in range(sampleList[i].shape[4]): # iterate through samples
            print("Saving sample number: " + str(j+1))
            sample = np.zeros((2000,2000,300), dtype=np.float32) # NOTE: calculating individual sample sizes has not been developed yet, so I declare a large empty volume and clip later
            counter = np.zeros((2000,2000,300), dtype=np.float32)
            for k in range(sampleList[i].shape[0]): # iterate through sub volumes
                xCoordinate = int(overlappingCoordinates[k,0] / sampleRates[i])
                yCoordinate = int(overlappingCoordinates[k,1] / sampleRates[i])
                zCoordinate = int(overlappingCoordinates[k,2] / sampleRates[i])
                sample[xCoordinate:xCoordinate+sampleList[i].shape[1], yCoordinate:yCoordinate+sampleList[i].shape[2], \
                        zCoordinate:zCoordinate+sampleList[i].shape[3]] += sampleList[i][k,:,:,:,j]
                counter[xCoordinate:xCoordinate+sampleList[i].shape[1], yCoordinate:yCoordinate+sampleList[i].shape[2], \
                        zCoordinate:zCoordinate+sampleList[i].shape[3]] += 1

            # find the true size of the sample
            for k in range(2000):
                if counter[k,0,0] == 0.0:
                    clipXCoordinate = k
                    break
            for k in range(2000):
                if counter[0,k,0] == 0.0:
                    clipYCoordinate = k
                    break
            for k in range(200):
                if counter[0,0,k] == 0.0:
                    clipZCoordinate = k
                    break

            # clip the sample
            sample = sample[0:clipXCoordinate, 0:clipYCoordinate, 0:clipZCoordinate]
            counter = counter[0:clipXCoordinate, 0:clipYCoordinate, 0:clipZCoordinate]
            sample /= counter # average each voxel

            # save 2D slices
            path = args["saveSamplePath"] + str(i+1) + "/"
            try:
                os.mkdir(path)
            except:
                pass
            path = path + "sample" + str(j+1) + "/"
            os.mkdir(path)
            for k in range(sample.shape[2]):
                sliceNumber = str(k).zfill(4)
                f = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/" + sliceNumber + ".jpg"
                if (np.amax(sample) - np.min(sample)) != 0:
                    sliceIm = 255 * ((sample[:,:,k] - np.min(sample)) / (np.amax(sample) - np.min(sample)))
                else:
                    sliceIm = sample[:,:,k]
                sliceIm = np.transpose(sliceIm, (1,0))
                im = Image.fromarray(sliceIm)
                im.convert('RGB').save(f)

            # save video of sample
            videoFile = args["saveVideoPath"] + "layer-" + str(i+1) + "-sample-" + str(j+1) + ".avi"
            sliceFiles = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/%04d.jpg"
            os.system("ffmpeg -y -framerate 10 -start_number 0 -i " + sliceFiles + " -vcodec mpeg4 " + videoFile)

    # save one video that spans all neurons
    videoFiles = os.listdir(args["saveVideoPath"])
    videoFiles.sort()
    if len(videoFiles) > 0:
        del videoFiles[0] # delete the video allSamples.avi that already exists TODO: search if file exists and then delete
    ffmpegConcatFileData = []
    for f in videoFiles:
        ffmpegConcatFileData.append("file " + args["saveVideoPath"] + f)

    # write the list of video files to a config file -- needed for ffmpeg
    ffmpegFileList = open(args["ffmpegFileListPath"], 'w')
    for line in ffmpegConcatFileData:
        ffmpegFileList.write(line+"\n")
    ffmpegFileList.close()

    # generate the video
    outputConcatVideo = args["saveVideoPath"] + "allSamples.avi"
    os.system("ffmpeg -y -f concat -safe 0 -i " + args["ffmpegFileListPath"] + " -c copy " + outputConcatVideo)

    # TODO:
        # evntually: test out adding text to videos with neuron info?



# TODO for class Volume:
    # def init()
        # read volume
        # if param="unsupervised":
        #     initialize prediction volume of all zeros -- trim later
            # initialize coordinate counter same shape
        # else:
        #     initialize prediction volume same shape as input volume

    # def groundTruth()  # TODO

    # def getTrainingSample()
        # take x and y bounds as parameter

    # def getPredictionSample()
        # begin returning samples along iterated loop
        # return coordinates as well

    # def reconstruct()
    #     args:
    #         sample predictions
    #         coordinates
    #         sampleRate=1

    # def trimPredictionVolume()
        # trim zeros from prediction volume

    # def savePredictionVolume():
        # save output prediction as a volume

    # def savePredictionSlices():
        # save output prediction as series of slices

    # def savePredictionAsVideo():
    #     save the prediction as a video
    #
    # outside of class:
    #
    # def saveAllSamplesAsVideo():
    #     concatenate all videos into one video
    #     takes a list of Volume's as parameter


import numpy as np
import pdb
import os
from PIL import Image
import math
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
