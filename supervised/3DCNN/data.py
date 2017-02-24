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
        self.volume = np.array(volume)
        self.groundTruth = tiff.imread(args["groundTruthFile"])
        self.predictionImageInk = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.predictionImageSurf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.trainingImage = np.zeros(self.predictionImageInk.shape, dtype=np.uint16)
        self.surfaceImage = tiff.imread(args["surfaceDataFile"])
        self.all_truth = []
        self.all_preds = []


    def getTrainingSample(self, args):
        # allocate an empty array with appropriate size
        trainingSamples = np.zeros((args["numCubes"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["numCubes"], args["n_Classes"]), dtype=np.float32)
        # restrict training to left half
        #colBounds=[int(self.volume.shape[1]*args["train_portion"]), self.volume.shape[1] - args["x_Dimension"]]
        colBounds=[0,int(self.volume.shape[1]*args["train_portion"])]
        rowBounds=[0,self.volume.shape[0]-args["y_Dimension"]]

        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(colBounds[0], colBounds[1])
            yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = 0
            if np.max(self.volume[yCoordinate, xCoordinate]) > 21500:
                zCoordinate = np.max(0,self.surfaceImage[yCoordinate,xCoordinate] - args["surfaceCushion"])

            # add sample to array, with appropriate shape
            sample = (self.volume[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample

            # NOTE: this volume class assumes that there are three classifications
            #TODO: clean up
            no_ink = len(np.where(self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]] < 85)[0])
            papyrus = len(np.where(np.logical_and(self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]] > 85, \
                        self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]] < 171))[0])
            ink = len(np.where(self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]] > 171)[0])

            classification = np.argmax([no_ink, papyrus, ink])
            gt = [0,0,0]
            gt[classification] = 1.0
            self.trainingImage[yCoordinate,xCoordinate] = int(65534/2 * classification)
            groundTruth[i] = gt

        #pdb.set_trace()
        return trainingSamples, groundTruth


    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        rowCoordinate = startingCoordinates[0]
        colCoordinate = startingCoordinates[1]
        zCoordinate = self.surfaceImage[rowCoordinate,colCoordinate] - args["surfaceCushion"]

        predictionSamples = np.zeros((args["predictBatchSize"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        coordinates = np.zeros((args["predictBatchSize"], 2), dtype=np.int)
        count = 0
        while count < args["predictBatchSize"]:
            if (colCoordinate + args["x_Dimension"]) > self.volume.shape[1]:
                colCoordinate = 0
                rowCoordinate += args["overlapStep"]
            if (rowCoordinate + args["y_Dimension"]) > self.volume.shape[0]:
                # yCoordinate = 0
                break
            zCoordinate = self.surfaceImage[rowCoordinate,colCoordinate] - args["surfaceCushion"]
            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_Dimension"], \
                    colCoordinate:colCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            predictionSamples[count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            coordinates[count] = [rowCoordinate, colCoordinate]

            colCoordinate += args["overlapStep"]
            count += 1

        return (predictionSamples), (coordinates), [rowCoordinate, colCoordinate]


    def reconstruct(self, args, samples, coordinates):
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
                xpoint = coordinates[i,0]
                ypoint = coordinates[i,1]

                # all_truth array for confusion matrix, 1=ink, 0=noink
                if(self.groundTruth[xpoint,ypoint]) > 200:
                    self.all_truth.append(2.0)
                elif(100 < self.groundTruth[xpoint,ypoint] < 200):
                    self.all_truth.append(1.0)
                else:
                    self.all_truth.append(0.0)

                if np.argmax(samples[i,:]) == 2:
                    self.predictionImageInk[xpoint:xpoint+args["overlapStep"],ypoint:ypoint+args["overlapStep"]] = samples[i,2]
                    self.all_preds.append(2.0)
                elif np.argmax(samples[i,:]) == 1:
                    self.predictionImageSurf[xpoint:xpoint+args["overlapStep"],ypoint:ypoint+args["overlapStep"]] = samples[i,1]
                    self.all_preds.append(1.0)
                else:
                    self.all_preds.append(0.0)


    def savePredictionImage(self, args):
        predictionImageInk = 65535 * ( (self.predictionImageInk.copy() - np.min(self.predictionImageInk)) / (np.amax(self.predictionImageInk) - np.min(self.predictionImageInk)) )
        predictionImageInk = np.array(predictionImageInk, dtype=np.uint16)
        predictionImageSurf = 65535 * ( (self.predictionImageSurf.copy() - np.min(self.predictionImageSurf)) / (np.amax(self.predictionImageSurf) - np.min(self.predictionImageSurf)) )
        predictionImageSurf = np.array(predictionImageSurf, dtype=np.uint16)

        tm = datetime.datetime.today().timetuple()
        tmstring = ""
        for t in range(3):
            tmstring += str(tm[t])
            tmstring+= "-"
        for t in range(3,5):
            tmstring += str(tm[t])
        tmstring += "h"
        specstring = "{}x{}x{}-".format(args["x_Dimension"], args["y_Dimension"], args["z_Dimension"])
        specstring = specstring + tmstring

        output_path = args["savePredictionPath"] + specstring
        os.mkdir(output_path)

        description = ""
        for arg in args:
            description += arg+": " + str(args[arg]) + "\n"

        tiff.imsave(output_path + "/predictionInk-{}.tif".format(specstring), predictionImageInk)
        tiff.imsave(output_path + "/predictionSurf-{}.tif".format(specstring), predictionImageSurf)
        tiff.imsave(output_path + "/training-{}.tif".format(specstring), self.trainingImage)
        confusion = confusion_matrix(self.all_truth, self.all_preds)
        norm_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix, before normalization: \n{}".format(confusion))
        print("Normalized confusion matrix: \n{}".format(norm_confusion))
        np.savetxt(output_path + "/confusion-{}.txt".format(specstring), confusion, fmt='%1.0f')
        np.savetxt(output_path + "/confusion-norm-{}.txt".format(specstring), norm_confusion, fmt='%1.4f')
        np.savetxt(output_path +'/description.txt', [description], delimiter=' ', fmt="%s")


    def totalPredictions(self, args):
        xSlides = (self.volume.shape[0] - args["x_Dimension"]) / args["overlapStep"]
        ySlides = (self.volume.shape[1] - args["y_Dimension"]) / args["overlapStep"]
        return int(xSlides * ySlides)
