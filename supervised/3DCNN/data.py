import numpy as np
import pdb
import os
from PIL import Image
import math
import datetime
import tifffile as tiff
from sklearn.metrics import confusion_matrix

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
        self.trainingImage = np.zeros(self.predictionImage.shape, dtype=np.uint16)

        self.all_truth = []
        self.all_preds = []


    def getTrainingSample(self, args):
        # restrict training to left half
        xBounds=[0,int(self.volume.shape[0]/2)-args["x_Dimension"]]
        yBounds=[0,self.volume.shape[1]-args["y_Dimension"]]
        trainingSamples = []
        groundTruth = []
        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(xBounds[0], xBounds[1])
            yCoordinate = np.random.randint(yBounds[0], yBounds[1])
            zCoordinate = 0
            #pdb.set_trace()
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
            self.trainingImage[xCoordinate,yCoordinate] = int(65534/2 * classification)
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
        while count < args["predictBatchSize"]:
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
                xpoint = coordinates[i,0]
                ypoint = coordinates[i,1]

                # all_truth array for confusion matrix, 1=ink, 0=noink
                if(self.groundTruth[xpoint,ypoint,0]) > 200:
                    self.all_truth.append(1.0)
                else:
                    self.all_truth.append(0.0)

                if np.argmax(samples[i,:]) == 2:
                    self.predictionImage[xpoint:xpoint+args["overlapStep"],ypoint:ypoint+args["overlapStep"]] = 1.0
                    self.all_preds.append(1.0)
                else:
                    self.all_preds.append(0.0)


    def savePredictionImage(self, args):
        print("preparing predictionImage with shape {}".format(self.predictionImage.shape))
        predictionImage = 65535 * ( (self.predictionImage.copy() - np.min(self.predictionImage)) / (np.amax(self.predictionImage) - np.min(self.predictionImage)) )
        predictionImage = np.array(predictionImage, dtype=np.uint16)
        print("saving predictionImage with shape {}".format(predictionImage.shape))
        tm = datetime.datetime.today().timetuple()
        tmstring = ""
        for t in range(3):
            tmstring += str(tm[t])
            tmstring+= "-"
        for t in range(3,5):
            tmstring += str(tm[t])
        tmstring += "h"
        specstring = "{}x{}-".format(args["x_Dimension"], args["y_Dimension"])
        specstring = specstring + tmstring

        tiff.imsave(args["savePredictionPath"] + "prediction-{}.tif".format(specstring), predictionImage)
        tiff.imsave(args["savePredictionPath"] + "training-{}.tif".format(specstring), self.trainingImage)
        confusion = confusion_matrix(self.all_truth, self.all_preds)
        print(confusion)
        np.savetxt(args["savePredictionPath"] + "confusion-{}.txt".format(specstring), confusion, fmt='%1.0f')
