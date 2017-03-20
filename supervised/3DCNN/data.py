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
        self.test_truth = []
        self.test_preds = []


    def getTrainingSample(self, args, testSet=False):
        # allocate an empty array with appropriate size
        trainingSamples = np.zeros((args["numCubes"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["numCubes"], args["n_Classes"]), dtype=np.float32)

        # restrict training to RIGHT portion
        #colBounds=[int(self.volume.shape[1]*args["train_portion"]), self.volume.shape[1] - args["x_Dimension"]]
        colBounds=[int(args["train_portion"] * self.volume.shape[1]), self.volume.shape[1] - args["x_Dimension"]]
        rowBounds=[0, int(self.volume.shape[0] - args["y_Dimension"])]
        if testSet:
            # get samples from "the other side" for test set
            colBounds=[0, int(args["train_portion"] * (self.volume.shape[1]-args["x_Dimension"]))]

        for i in range(args["numCubes"]):
            xCoordinate = np.random.randint(colBounds[0], colBounds[1])
            yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
            zCoordinate = 0
            label_avg = np.mean(self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]])

            # use this loop to only train on the surface
            # and make sure 90% of the ground truth in the area is the same
            # while (np.min(np.max(self.volume[yCoordinate:yCoordinate+args["y_Dimension"], xCoordinate:xCoordinate+args["x_Dimension"]], axis=2)) < args["surfaceThresh"] or \
            while label_avg in range(int(.1*255), int(.9*255)) or\
                    np.min(self.volume[yCoordinate:yCoordinate+args["y_Dimension"], xCoordinate:xCoordinate+args["x_Dimension"]]) == 0:
                xCoordinate = np.random.randint(colBounds[0], colBounds[1])
                yCoordinate = np.random.randint(rowBounds[0], rowBounds[1])
                label_avg = np.mean(self.groundTruth[yCoordinate:yCoordinate+args["y_Dimension"], \
                        xCoordinate:xCoordinate+args["x_Dimension"]])


            if args["addRandom"] and np.random.randint(20) == 8:
                # make 5% of the training samples random data labeled as non-ink
                v_min = np.min(self.volume[yCoordinate, xCoordinate])
                v_max = np.max(self.volume[yCoordinate, xCoordinate])
                v_median = np.median(self.volume[yCoordinate, xCoordinate])
                low = v_median - args["randomRange"]
                high = v_median + args["randomRange"]
                sample = np.random.random([args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
                trainingSamples[i, :, :, :] = ((high - low) * sample) + low
                groundTruth[i] = [1.0,0.0]
                continue


            jitter_range = args["jitterRange"]
            jitter = np.random.randint(jitter_range[0],jitter_range[1])
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

            if label_avg > (.9 * 255):
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
                if(self.groundTruth[xpoint,ypoint]) > 200:
                    inks += 1
                    self.all_truth.append(1.0)
                else:
                    self.all_truth.append(0.0)

                if np.argmax(samples[i,:]) == 1:
                    self.all_preds.append(1.0)
                    if(center_step > 0):
                        self.predictionImageInk[xpoint-center_step:xpoint+center_step, ypoint-center_step:ypoint+center_step] = samples[i,1]
                    else:
                        self.predictionImageInk[xpoint,ypoint] = samples[i,1]

                else:
                    self.all_preds.append(0.0)
                    if(center_step > 0):
                        self.predictionImageSurf[xpoint-center_step:xpoint+center_step, ypoint-center_step:ypoint+center_step] = samples[i,0]
                    else:
                        self.predictionImageSurf[xpoint,ypoint] = samples[i,0]

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
        try:
            #os.mkdir(output_path)
            os.makedirs(output_path + "{}/".format(iteration))
        except:
            pass

        description = ""
        for arg in sorted(args.keys()):
            description += arg+": " + str(args[arg]) + "\n"

        tiff.imsave(output_path + "{}/predictionInk-{}.tif".format(iteration, specstring), predictionImageInk)
        predictionImageInk[:, int(self.volume.shape[1]*args["train_portion"]):] = 0
        tiff.imsave(output_path + "{}/predictionInkTest-{}.tif".format(iteration, specstring), predictionImageInk)
        tiff.imsave(output_path + "{}/predictionSurf-{}.tif".format(iteration, specstring), predictionImageSurf)
        tiff.imsave(output_path + "{}/training-{}.tif".format(iteration, specstring), self.trainingImage)
        all_confusion = confusion_matrix(self.all_truth, self.all_preds)
        test_confusion = confusion_matrix(self.test_truth, self.test_preds)
        all_norm_confusion = all_confusion.astype('float') / all_confusion.sum(axis=1)[:, np.newaxis]
        test_norm_confusion = test_confusion.astype('float') / test_confusion.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix for ALL points: \n{}".format(all_norm_confusion))
        print("Normalized confusion matrix for TEST points: \n{}".format(test_norm_confusion))
        np.savetxt(output_path + "{}/all-confusion-{}.txt".format(iteration, specstring), all_confusion, fmt='%1.0f')
        np.savetxt(output_path + "{}/all-confusion-norm-{}.txt".format(iteration, specstring), all_norm_confusion, fmt='%1.4f')
        np.savetxt(output_path + "{}/test-confusion-{}.txt".format(iteration, specstring), test_confusion, fmt='%1.0f')
        np.savetxt(output_path + "{}/test-confusion-norm-{}.txt".format(iteration, specstring), test_norm_confusion, fmt='%1.4f')

        np.savetxt(output_path +'description.txt', [description], delimiter=' ', fmt="%s")


    def totalPredictions(self, args):
        xSlides = (self.volume.shape[0] - args["x_Dimension"]) / args["overlapStep"]
        ySlides = (self.volume.shape[1] - args["y_Dimension"]) / args["overlapStep"]
        return int(xSlides * ySlides)
