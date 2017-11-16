import numpy as np
import pdb
import os
from PIL import Image
import math
import datetime
import tifffile as tiff
import ops
from sklearn.metrics import confusion_matrix, recall_score, precision_score
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
        self.predictionVolume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)

        self.groundTruth = tiff.imread(args["groundTruthFile"])
        self.max_truth = np.iinfo(self.groundTruth.dtype).max
        self.predictionImageInk = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.predictionImageSurf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.trainingImage = np.zeros(self.predictionImageInk.shape, dtype=np.uint16)
        self.surfaceImage = tiff.imread(args["surfaceDataFile"])
        self.all_truth, self.all_preds = [], []
        self.test_truth, self.test_preds = [], []
        self.test_results, self.test_results_norm = [], []
        self.all_results, self.all_results_norm = [], []

        self.coordinate_pool = []
        self.train_index = 0
        self.epoch = 0



    def getTrainingBatch(self, args, bounds=3):
        if len(self.coordinate_pool) == 0: # initialization
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], bounds)
            self.coordinate_pool = ops.generateCoordinatePool(args, self.volume, rowBounds, colBounds, self.groundTruth)
            np.random.shuffle(self.coordinate_pool)
        if self.train_index + args["batch_size"] > len(self.coordinate_pool): # end of epoch
            print("finished epoch")
            self.train_index = 0
            self.trainingImage = np.zeros(self.predictionImageInk.shape, dtype=np.uint16)
            self.epoch += 1
            np.random.shuffle(self.coordinate_pool)

        trainingSamples = np.zeros((args["batch_size"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["batch_size"], args["x_Dimension"], args["y_Dimension"]), dtype=np.int32)
        rowStep = int(args["y_Dimension"]/2)
        colStep = int(args["x_Dimension"]/2)

        # populate the samples and labels
        for i in range(args["batch_size"]):
            if args["balance_samples"] and (i > args["batch_size"] / 2):
                if np.sum(groundTruth[:,1] / i) > .5:
                    # more than 50% ink samples
                    self.moveToNextNegativeSample(args)
                else:
                    # fewer than 50% ink samples
                    self.moveToNextPositiveSample(args)

            rowCoord, colCoord, label, augment_seed = self.coordinate_pool[self.train_index]
            zCoord = self.surfaceImage[rowCoord, colCoord] - args["surface_cushion"]

            if args["use_jitter"]:
                zCoord = np.maximum(0, zCoord +  np.random.randint(args["jitter_range"][0], args["jitter_range"][1]))

            if args["add_random"] and label < .1 and np.random.randint(args["random_step"]) == 0:
                # make this non-ink sample random data labeled as non-ink
                sample = ops.getRandomBrick(args, self.volume, colCoord, rowCoord)
                groundTruth[i] = [1.0,0.0]
                continue

            sample = self.volume[rowCoord-rowStep:rowCoord+rowStep, colCoord-colStep:colCoord+colStep, zCoord:zCoord+args["z_Dimension"]]

            if args["add_augmentation"]:
                sample = ops.augmentSample(args, sample, augment_seed)
                # change the augment seed for the next time around
                self.coordinate_pool[self.train_index][3] = (augment_seed+1) % 4

            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            groundTruth[i] = label
            self.trainingImage[rowCoord,colCoord] = int(65534/2) +  int((65534/2)*np.mean(label))
            # if label_avg is greater than .9*255, then groundTruth=[0, 1]
            self.train_index += 1

        return trainingSamples, groundTruth, self.epoch



    def getTrainingSample(self, args, testSet=False, bounds=3):
        # allocate an empty array with appropriate size
        trainingSamples = np.zeros((args["num_cubes"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        groundTruth = np.zeros((args["num_cubes"], args["x_Dimension"], args["y_Dimension"]), dtype=np.int32)
        rowStep = int(args["y_Dimension"]/2)
        colStep = int(args["x_Dimension"]/2)

        # restrict training to TOP portion by default
        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        if testSet:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], (args["train_bounds"]+2)%4)
        else:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], args["train_bounds"])


        for i in range(args["num_cubes"]):
            rowCoord, colCoord, zCoord, label = ops.findRandomCoordinate(args, colBounds, rowBounds, self.groundTruth, self.surfaceImage, self.volume)

            if args["add_random"] and not testSet and label_avg < .1 and np.random.randint(args["random_step"]) == 0:
                # make this non-ink sample random data labeled as non-ink
                sample = ops.getRandomBrick(args, self.volume, xCoordinate, yCoordinate)
                groundTruth[i] = [1.0,0.0]
                continue

            if args["use_jitter"]:
                zCoord = np.maximum(0, zCoord +  np.random.randint(args["jitter_range"][0], args["jitter_range"][1]))

            sample = self.volume[rowCoord-rowStep:rowCoord+rowStep, colCoord-colStep:colCoord+colStep, zCoord:zCoord+args["z_Dimension"]]

            if args["add_augmentation"]:
                sample = ops.augmentSample(args, sample)

            label_avg = np.mean(label)
            self.trainingImage[rowCoord,colCoord] = int(label_avg*65534)

            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            groundTruth[i] = label

        return trainingSamples, groundTruth



    def getPredictionSample(self, args, startingCoordinates):
        # return the prediction sample along side of coordinates
        rowCoordinate = startingCoordinates[0]
        colCoordinate = startingCoordinates[1]
        zCoordinate = self.surfaceImage[rowCoordinate+int(args["y_Dimension"]/2), colCoordinate+int(args["x_Dimension"]/2)] - args["surface_cushion"]

        predictionSamples = np.zeros((args["prediction_batch_size"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        coordinates = np.zeros((args["prediction_batch_size"], 2), dtype=np.int)
        count = 0
        while count < args["prediction_batch_size"]:
            if (colCoordinate + args["x_Dimension"]) > self.volume.shape[1]:
                colCoordinate = 0
                rowCoordinate += args["overlap_step"]
            if (rowCoordinate + args["y_Dimension"]) > self.volume.shape[0]:
                break

            # don't predict on it if it's not on the fragment
            if np.max(self.volume[rowCoordinate, colCoordinate]) < args["surface_threshold"]:
                colCoordinate += args["overlap_step"]
                continue

            zCoordinate = self.surfaceImage[rowCoordinate+int(args["y_Dimension"]/2), colCoordinate+int(args["x_Dimension"]/2)] - args["surface_cushion"]
            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_Dimension"], \
                    colCoordinate:colCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            predictionSamples[count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            coordinates[count] = [rowCoordinate, colCoordinate]

            colCoordinate += args["overlap_step"]
            count += 1

        return (predictionSamples), (coordinates), [rowCoordinate, colCoordinate]



    def getPredictionSample3D(self, args, startingCoordinates):
        rowCoordinate = startingCoordinates[0]
        colCoordinate = startingCoordinates[1]
        depthCoordinate = startingCoordinates[2]

        predictionSamples = np.zeros((args["prediction_batch_size"], args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]), dtype=np.float32)
        coordinates = np.zeros((args["prediction_batch_size"], 3), dtype=np.int)

        sample_count = 0
        while sample_count < args["prediction_batch_size"]:
            if (colCoordinate + args["x_Dimension"]) > self.volume.shape[1]:
                colCoordinate = 0
                rowCoordinate += args["overlap_step"]
            if (rowCoordinate + args["y_Dimension"]) > self.volume.shape[0]:
                colCoordinate = 0
                rowCoordinate = 0
                depthCoordinate += 1
            if depthCoordinate >= args["predict_depth"]:
                break

            # don't predict on it if it's not on the fragment
            if np.max(self.volume[rowCoordinate, colCoordinate]) < args["surface_threshold"]:
                colCoordinate += args["overlap_step"]
                continue

            # grab the sample and place it in output
            zCoordinate = self.surfaceImage[rowCoordinate+int(args["y_Dimension"]/2), colCoordinate+int(args["x_Dimension"]/2)] - args["surface_cushion"]
            if args["predict_depth"] > 1:
                #TODO this z-mapping mapping will eventually be something more intelligent
                zCoordinate += (depthCoordinate)
                #zCoordinate = depthCoordinate * int((self.volume.shape[2] - args["z_Dimension"]) / args["predict_depth"])

            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_Dimension"], \
                    colCoordinate:colCoordinate+args["x_Dimension"], zCoordinate:zCoordinate+args["z_Dimension"]])
            predictionSamples[sample_count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            coordinates[sample_count] = [rowCoordinate, colCoordinate, depthCoordinate]

            # increment variables for next iteration
            colCoordinate += args["overlap_step"]
            sample_count += 1

        return (predictionSamples), (coordinates), [rowCoordinate, colCoordinate, depthCoordinate]



    def reconstruct(self, args, samples, coordinates):
        center_step = int(round(args["overlap_step"] / 2))
        inks = 0
        # reconstruct prediction volume one prediction sample at a time
        for i in range(coordinates.shape[0]):
            xpoint = coordinates[i,0] + int(args["x_Dimension"] / 2)
            ypoint = coordinates[i,1] + int(args["y_Dimension"] / 2)

            # all_truth array for confusion matrix, 1=ink, 0=fragment
            #pdb.set_trace()
            if(self.groundTruth[xpoint,ypoint]) > .9*self.max_truth:
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

            self.all_preds.append(np.argmax(samples[i,:]))

            # test batch (right side)
            #TODO  make this correspond to the specified train side
            if xpoint > int(self.volume.shape[1]*args["train_portion"]):
                self.test_preds.append(self.all_preds[-1])
                self.test_truth.append(self.all_truth[-1])



    def reconstruct3D(self, args, predictionValues, coordinates):
        center_step = int(round(args["overlap_step"] / 2))
        rowstep = int(args["y_Dimension"] / 2)
        colstep = int(args["x_Dimension"] / 2)
        for i in range(coordinates.shape[0]):
            rowpoint = coordinates[i,0] + rowstep
            colpoint = coordinates[i,1] + colstep
            zpoint = coordinates[i,2]
            predictionValue = np.mean(predictionValues[i, :, :, 1])

            if(center_step > 0):
                self.predictionVolume[rowpoint-center_step:rowpoint+center_step, colpoint-center_step:colpoint+center_step, zpoint] = predictionValue
            else:
                self.predictionVolume[rowpoint, colpoint, zpoint] = predictionValue




    def savePrediction3D(self, args, iteration):
        # save individual pictures
        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.predictionVolume[:,:,d], predictionName='ink', depth=d)
        # save the average prediction across depths if depth is more than one
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.predictionVolume, axis=2), predictionName='ink-average')

        # save the output for samples not trained on
        rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], args["train_bounds"])
        self.predictionVolume[rowBounds[0]:rowBounds[1], colBounds[0]:colBounds[1]] = 0
        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.predictionVolume[:,:,d], predictionName='ink-no-train', depth=d)
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.predictionVolume, axis=2), predictionName='ink-average-no-train')

        # zero out the volume
        self.predictionVolume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)



    def savePredictionImage(self, args, iteration, predictValues=None, predictionName='ink', depth=0):
        if predictValues is None:
            #predictionImageInk = 65535 * ( (self.predictionImageInk.copy() - np.min(self.predictionImageInk)) / (np.amax(self.predictionImageInk) - np.min(self.predictionImageInk)) )
            predictionImage = (65535 * self.predictionImage).astype(np.uint16)
        else:
            predictionImage = (65535 * predictValues).astype(np.uint16)

        output_path = args["output_path"]
        try:
            os.makedirs(output_path + "/{}/".format(predictionName))
        except:
            pass

        # save the ink and surface predictions
        tiff.imsave(output_path + "/{}/prediction-iteration{}-depth{}.tif".format(predictionName, iteration, depth), predictionImage)
        tiff.imsave(output_path + "/training-{}.tif".format(iteration), self.trainingImage)

        # zero them out for the next predictions
        self.predictionImageInk = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.predictionImageSurf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)


    def savePredictionMetrics(self, args, iteration, minutes):
        output_path = args["output_path"] + '/'
        all_confusion = confusion_matrix(self.all_truth, self.all_preds)
        test_confusion = confusion_matrix(self.test_truth, self.test_preds)
        all_confusion_norm = all_confusion.astype('float') / all_confusion.sum(axis=1)[:, np.newaxis]
        test_confusion_norm = test_confusion.astype('float') / test_confusion.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix for ALL points: \n{}".format(all_confusion_norm))
        print("Normalized confusion matrix for TEST points: \n{}".format(test_confusion_norm))

        #calculate metrics
        all_precision = precision_score(self.all_truth, self.all_preds)
        all_recall = recall_score(self.all_truth, self.all_preds)
        test_precision = precision_score(self.test_truth, self.test_preds)
        test_recall = recall_score(self.test_truth, self.test_preds)

        #save results in csv
        column_names = 'iteration, minutes, true positive papyrus, false positive ink, false positive papyrus, true positive ink, precision, recall'
        self.test_results_norm.append([iteration] + [minutes] + test_confusion_norm.reshape(4).tolist() + [test_precision] + [test_recall])
        self.all_results_norm.append([iteration] + [minutes] + all_confusion_norm.reshape(4).tolist() + [all_precision] + [all_recall])
        np.savetxt(output_path + "confusion-all.csv", self.all_results_norm, fmt='%1.4f', header=column_names, delimiter=',')
        np.savetxt(output_path + "confusion-test.csv", self.test_results_norm, fmt='%1.4f', header=column_names, delimiter=',')

        # save description of this training session
        description = ""
        for arg in sorted(args.keys()):
            description += arg+": " + str(args[arg]) + "\n"
        np.savetxt(output_path +'description.txt', [description], delimiter=' ', fmt="%s")
        shutil.copy('model.py', output_path + 'network_model.txt')

        # zero-out predictions & images so next output is correct
        self.all_truth = []
        self.all_preds = []
        self.test_truth = []
        self.test_preds = []


    def moveToNextPositiveSample(self, args):
        while np.mean(self.coordinate_pool[self.train_index][2]) == 0:
            self.train_index += 1
        return


    def moveToNextNegativeSample(self, args):
        while np.mean(self.coordinate_pool[self.train_index][2]) == 1:
            self.train_index += 1
        return


    def totalPredictions(self, args):
        #TODO three-dimensions
        xSlides = (self.volume.shape[0] - args["x_Dimension"]) / args["overlap_step"]
        ySlides = (self.volume.shape[1] - args["y_Dimension"]) / args["overlap_step"]
        return int(xSlides * ySlides) * args["predict_depth"]
