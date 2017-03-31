import numpy as np
import pdb
import os
from PIL import Image
import math
import datetime
import tifffile as tiff
import ops
from sklearn.metrics import confusion_matrix
import shutil


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
        self.all_truth, self.all_preds = [], []
        self.test_truth, self.test_preds = [], []
        self.test_results, self.test_results_norm = [], []
        self.all_results, self.all_results_norm = [], []



    def getTrainingSample(self, args, testSet=False, bounds=3):
        # allocate an empty array with appropriate size
        trainingSamples = np.zeros((args["numCubes"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["numCubes"], args["n_Classes"]), dtype=np.float32)

        # restrict training to TOP portion by default
        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        if testSet:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], (bounds+2)%4)
        else:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], bounds)


        for i in range(args["numCubes"]):
            xCoordinate, yCoordinate, zCoordinate, label_avg = ops.findRandomCoordinate(args, colBounds, rowBounds, self.groundTruth, self.volume)

            if args["addRandom"] and not testSet and label_avg < .1 and np.random.randint(5) == 4:
                # make 20% of the non-ink samples random data labeled as non-ink
                sample = getRandomBrick(args, self.volume, xCoordinate, yCoordinate)
                groundTruth[i] = [1.0,0.0]
                continue


            jitter_range = args["jitterRange"]
            jitter = np.random.randint(jitter_range[0],jitter_range[1])
            if args["useJitter"]:
                zCoordinate = np.maximum(0, self.surfaceImage[yCoordinate,xCoordinate] - args["surfaceCushion"] + jitter)

            # add sample to array, with appropriate shape
            sample = (self.volume[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])

            # twelve total possible augmentations, ensure equal probability
            # for flip: original, flip left-right, flip up-down
            if jitter_range[0] < jitter < jitter_range[0] / 3:
                sample = np.flip(sample, axis=0)
            elif jitter_range[0] / 3 < jitter < jitter_range[1] / 3:
                sample = np.flip(sample, axis=1)
            # for rotate: original, rotate 90, rotate 180, or rotate 270
            if jitter_range[1] / 2 < jitter < jitter_range[1]:
                sample = np.rot90(sample, k=1, axes=(0,1))
            elif 0 < jitter < jitter_range[1] / 2:
                sample = np.rot90(sample, k=2, axes=(0,1))
            elif jitter_range[0] / 2 < jitter < 0:
                sample = np.rot90(sample, k=3, axes=(0,1))

            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample

            max_truth = np.iinfo(self.groundTruth.dtype).max
            if label_avg > (.9 * max_truth):
                gt = [0.0,1.0]
                self.trainingImage[yCoordinate,xCoordinate] = int(65534)
            else:
                gt = [1.0,0.0]
                self.trainingImage[yCoordinate,xCoordinate] = int(65534/2)
            groundTruth[i] = gt

        return trainingSamples, groundTruth



    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        rowCoordinate = startingCoordinates[0]
        colCoordinate = startingCoordinates[1]
        zCoordinate = self.surfaceImage[rowCoordinate+int(args["y_Dimension"]/2), colCoordinate+int(args["x_Dimension"]/2)] - args["surfaceCushion"]

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

            # don't predict on it if it's not on the fragment
            if np.max(self.volume[rowCoordinate, colCoordinate]) < args["surfaceThresh"]:
                colCoordinate += args["overlapStep"]
                continue

            zCoordinate = self.surfaceImage[rowCoordinate,colCoordinate] - args["surfaceCushion"]
            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_Dimension"], \
                    colCoordinate:colCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            predictionSamples[count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            coordinates[count] = [rowCoordinate, colCoordinate]

            colCoordinate += args["overlapStep"]
            count += 1

        return (predictionSamples), (coordinates), [rowCoordinate, colCoordinate]



    def reconstruct(self, args, samples, coordinates):
        center_step = int(round(args["overlapStep"] / 2))
        inks = 0
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
                xpoint = coordinates[i,0] + int(args["x_Dimension"] / 2)
                ypoint = coordinates[i,1] + int(args["y_Dimension"] / 2)

                # all_truth array for confusion matrix, 1=ink, 0=fragment
                #pdb.set_trace()
                if(self.groundTruth[xpoint,ypoint]) > .9*np.iinfo(self.groundTruth.dtype).max:
                    inks += 1
                    self.all_truth.append(1.0)
                else:
                    self.all_truth.append(0.0)

                if(center_step > 0):
                    self.predictionImageInk[xpoint-center_step:xpoint+center_step, ypoint-center_step:ypoint+center_step] = samples[i,1]
                    self.predictionImageSurf[xpoint-center_step:xpoint+center_step, ypoint-center_step:ypoint+center_step] = samples[i,0]
                else:
                    self.predictionImageInk[xpoint,ypoint] = samples[i,1]
                    self.predictionImageSurf[xpoint,ypoint] = samples[i,0]

                self.all_preds.append(samples[i,:])

                # test batch (left side)
                if xpoint < int(self.volume.shape[1]*args["train_portion"]):
                    self.test_preds.append(self.all_preds[-1])
                    self.test_truth.append(self.all_truth[-1])



    def savePredictionImage(self, args, iteration):
        predictionImageInk = 65535 * ( (self.predictionImageInk.copy() - np.min(self.predictionImageInk)) / (np.amax(self.predictionImageInk) - np.min(self.predictionImageInk)) )
        predictionImageInk = np.array(predictionImageInk, dtype=np.uint16)
        predictionImageSurf = 65535 * ( (self.predictionImageSurf.copy() - np.min(self.predictionImageSurf)) / (np.amax(self.predictionImageSurf) - np.min(self.predictionImageSurf)) )
        predictionImageSurf = np.array(predictionImageSurf, dtype=np.uint16)

        tm = datetime.datetime.today().timetuple()
        tmstring = ""
        for t in range(3):
            tmstring += str(tm[t])
            tmstring+= "-"
        tmstring += str(tm[3])
        tmstring += "h"

        specstring = "{}x{}x{}-".format(args["x_Dimension"], args["y_Dimension"], args["z_Dimension"])
        specstring = specstring + tmstring

        output_path = args["savePredictionFolder"]
        folders = ['ink', 'ink-no-train', 'surface',]
        for folder in folders:
            try:
                os.makedirs(output_path + "{}/".format(folder))
            except:
                pass

        description = ""
        for arg in sorted(args.keys()):
            description += arg+": " + str(args[arg]) + "\n"

        #save the ink predictions
        tiff.imsave(output_path + "{}/predictionInk-{}.tif".format(folders[0], iteration), predictionImageInk)
        predictionImageInk[:, int(self.volume.shape[1]*args["train_portion"]):] = 0
        tiff.imsave(output_path + "{}/predictionInkTest-{}.tif".format(folders[1], iteration), predictionImageInk)
        #save the surface predictions
        tiff.imsave(output_path + "{}/predictionSurf-{}.tif".format(folders[2], iteration), predictionImageSurf)
        tiff.imsave(output_path + "training.tif", self.trainingImage)
        #create confusion matrices
        all_confusion = confusion_matrix(self.all_truth, self.all_preds)
        test_confusion = confusion_matrix(self.test_truth, self.test_preds)
        all_confusion_norm = all_confusion.astype('float') / all_confusion.sum(axis=1)[:, np.newaxis]
        test_confusion_norm = test_confusion.astype('float') / test_confusion.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix for ALL points: \n{}".format(all_confusion_norm))
        print("Normalized confusion matrix for TEST points: \n{}".format(test_confusion_norm))
        #save confusion matrices in csv
        column_names = 'true positive papyrus, false positive ink, false positive papyrus, true positive ink'
        self.test_results_norm.append(test_confusion_norm.reshape(4))
        self.test_results.append(test_confusion.reshape(4))
        self.all_results_norm.append(all_confusion_norm.reshape(4))
        self.all_results.append(all_confusion.reshape(4))
        np.savetxt(output_path + "confusion-all.csv", self.all_results, fmt='%1.0f', header=column_names, delimiter=',')
        np.savetxt(output_path + "confusion-all-normalized.csv", self.all_results_norm, fmt='%1.4f', header=column_names, delimiter=',')
        np.savetxt(output_path + "confusion-test.csv", self.test_results, fmt='%1.0f', header=column_names, delimiter=',')
        np.savetxt(output_path + "confusion-test-normalized.csv", self.test_results_norm, fmt='%1.4f', header=column_names, delimiter=',')

        np.savetxt(output_path +'description.txt', [description], delimiter=' ', fmt="%s")
        #TODO shutil.copy model
        # zero-out predictions & images so next output is correct
        self.predictionImageInk = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.predictionImageSurf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.all_truth = []
        self.all_preds = []
        self.test_truth = []
        self.test_preds = []


    def totalPredictions(self, args):
        #TODO three-dimensions
        xSlides = (self.volume.shape[0] - args["x_Dimension"]) / args["overlapStep"]
        ySlides = (self.volume.shape[1] - args["y_Dimension"]) / args["overlapStep"]
        return int(xSlides * ySlides)
