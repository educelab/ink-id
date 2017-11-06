import numpy as np
import pdb
import os
from PIL import Image
import math
import datetime
import tifffile as tiff
import ops
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from scipy.ndimage.interpolation import rotate
import shutil
import time

class Volume:
    def __init__(self, args, volume_number):
        # Part 1: volume metadataf
        self.volume_args = args['volumes'][volume_number]
        self.volume_number = volume_number
        self.train_bounds = self.volume_args['train_bounds']
        self.train_portion = self.volume_args['train_portion']


        # Part 2: volume data
        data_files = os.listdir(self.volume_args['data_path'])
        data_files.sort()
        volume = []
        for f in data_files:
            slice_data = np.array(Image.open(self.volume_args['data_path']+f))
            volume.append(slice_data)
        self.volume = np.array(volume)
        if args["wobble_volume"]:
            self.wobbled_axes = []
            self.wobbled_angle = 0.0
            self.wobbled_volume = np.array(volume)
        print("  Volume {} shape: {}".format(self.volume_number, self.volume.shape))

        if len(self.volume_args['ground_truth']) > 0:
            self.ground_truth = tiff.imread(self.volume_args['ground_truth'])
        elif 'inked' in self.volume_args['data_path']:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535
        elif 'blank' in self.volume_args['data_path']:
            self.ground_truth = np.zeros((self.volume.shape[0:2]), dtype=np.uint16)
        else:
            self.ground_truth = np.ones((self.volume.shape[0:2]), dtype=np.uint16) * 65535

        if len(self.volume_args['surface_data']) > 0:
            self.surface_image = tiff.imread(self.volume_args['surface_data'])
        elif self.volume.shape[2] > args["z_dimension"]:
            print("  Approximating surface for volume {}...".format(self.volume_number))
            surf_start_time = time.time()
            self.surface_image = ops.generateSurfaceApproximation(args, self.volume)
            print("  Surface approximation took {:.2f} minutes.".format((time.time()-surf_start_time)/60))
        else:
            self.surface_image = np.zeros((self.volume.shape[0:2]))

        self.surface_mask_image = np.ones((self.volume.shape[0], self.volume.shape[1]), dtype=np.int)
        if len(self.volume_args['surface_mask']) > 0:
            self.surface_mask_image = tiff.imread(self.volume_args['surface_mask'])


        # Part 3: prediction data
        self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)
        self.prediction_image_ink = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_image_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.prediction_plus_surf = np.zeros((self.volume.shape[0:2]), dtype=np.float32)
        self.training_image = np.zeros(self.prediction_image_ink.shape, dtype=np.uint16)
        try:
            self.prediction_overlap_step = self.volume_args["prediction_overlap_step"]
        except:
            print("  No prediction_overlap_step for {}".format(self.volume_args['name']))


        # Part 4: prediction metadata and more
        self.max_truth = np.iinfo(self.ground_truth.dtype).max
        self.surface_mask = self.surface_mask_image / np.iinfo(self.surface_mask_image.dtype).max
        self.all_truth, self.all_preds = [], []
        self.test_truth, self.test_preds = [], []
        self.test_results, self.test_results_norm = [], []
        self.all_results, self.all_results_norm = [], []
        self.output_path = args["output_path"]+"/{}/".format(self.volume_args['name'])
        self.coordinate_pool = []
        self.train_index = 0
        self.epoch = 0



    def getTrainingBatch(self, args, n_samples):
        if len(self.coordinate_pool) == 0: # initialization
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], self.train_bounds, self.train_portion)
            print("Generating coordinate pool for volume {}...".format(self.volume_number))
            self.coordinate_pool = ops.generateCoordinatePool(args, self.volume, rowBounds, colBounds, self.ground_truth, self.surface_mask, self.train_bounds, self.train_portion)
            np.random.shuffle(self.coordinate_pool)
            print("Coordinate pool for volume {} is ready...".format(self.volume_number))
        if self.train_index + n_samples >= len(self.coordinate_pool): # end of epoch
            self.incrementEpoch(args)

        trainingSamples = np.zeros((n_samples, args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((n_samples, args["n_classes"]), dtype=np.float32)
        rowStep = int(args["y_dimension"]/2)
        colStep = int(args["x_dimension"]/2)

        # populate the samples and labels
        for i in range(n_samples):
            if args["balance_samples"] and (i > n_samples / 2):
                if np.sum(groundTruth[:,1] / i) > .5:
                    # more than 50% ink samples
                    self.moveToNextNegativeSample(args)
                else:
                    # fewer than 50% ink samples
                    self.moveToNextPositiveSample(args)

            rowCoord, colCoord, label, augment_seed = self.coordinate_pool[self.train_index]
            zCoord = max(0, self.surface_image[rowCoord, colCoord] - args["surface_cushion"])

            if args["use_jitter"]:
                zCoord = np.maximum(0, zCoord +  np.random.randint(args["jitter_range"][0], args["jitter_range"][1]))

            if args["add_random"] and label < .1 and np.random.randint(args["random_step"]) == 0:
                # make this non-ink sample random data labeled as non-ink
                sample = ops.getRandomBrick(args, self.volume, colCoord, rowCoord)
                groundTruth[i] = [1.0,0.0]
                continue

            if args["wobble_volume"]:
                zCoord = ops.adjustDepthForWobble(args, rowCoord, colCoord, zCoord, self.wobbled_angle, self.wobbled_axes, self.volume.shape)
                sample = self.wobbled_volume[rowCoord-rowStep:rowCoord+rowStep, colCoord-colStep:colCoord+colStep, zCoord:zCoord+args["z_dimension"]]
            else:
                sample = self.volume[rowCoord-rowStep:rowCoord+rowStep, colCoord-colStep:colCoord+colStep, zCoord:zCoord+args["z_dimension"]]

            if args["add_augmentation"]:
                sample = ops.augmentSample(args, sample, augment_seed)
                # change the augment seed for the next time around
                self.coordinate_pool[self.train_index][3] = (augment_seed+1) % 4

            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            groundTruth[i, int(label)] = 1.0
            self.training_image[rowCoord,colCoord] = int(65534/2) +  int((65534/2)*label)
            # if label_avg is greater than .9*255, then groundTruth=[0, 1]
            self.train_index += 1

        return trainingSamples, groundTruth, self.epoch



    def getTestBatch(self, args, n_samples):
        print("  Generating test set for volume {}...".format(self.volume_number))
        # allocate an empty array with appropriate size
        trainingSamples = np.zeros((n_samples, args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        groundTruth = np.zeros((n_samples, args["n_classes"]), dtype=np.float32)

        # this gets the "other half" of the data not in the train portion
        # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
        rowBounds, colBounds = ops.bounds(args, self.volume.shape, (self.train_bounds+2)%4, 1-self.train_portion)

        for i in range(n_samples):
            rowCoordinate, colCoordinate, zCoordinate, label_avg = ops.findRandomCoordinate(args, colBounds, rowBounds, self.ground_truth, self.surface_image, self.surface_mask, self.volume.shape, testSet=True)

            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_dimension"], \
                        colCoordinate:colCoordinate+args["x_dimension"], zCoordinate:zCoordinate+args["z_dimension"]])

            if label_avg > (.9 * self.max_truth):
                gt = [0.0,1.0]
                self.training_image[rowCoordinate,colCoordinate] = int(65534)
            else:
                gt = [1.0,0.0]
                self.training_image[rowCoordinate,colCoordinate] = int(65534/2)

            trainingSamples[i, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            groundTruth[i] = gt

        return trainingSamples, groundTruth



    def getPredictionSample3D(self, args, startingCoordinates, overlap_step):
        rowCoordinate = startingCoordinates[0]
        colCoordinate = startingCoordinates[1]
        depthCoordinate = startingCoordinates[2]

        predictionSamples = np.zeros((args["prediction_batch_size"], args["x_dimension"], args["y_dimension"], args["z_dimension"]), dtype=np.float32)
        coordinates = np.zeros((args["prediction_batch_size"], 3), dtype=np.int)

        sample_count = 0
        while sample_count < args["prediction_batch_size"]:
            if (colCoordinate + args["x_dimension"]) > self.volume.shape[1]:
                colCoordinate = 0
                rowCoordinate += overlap_step
            if (rowCoordinate + args["y_dimension"]) > self.volume.shape[0]:
                colCoordinate = 0
                rowCoordinate = 0
                depthCoordinate += 1
            if depthCoordinate >= args["predict_depth"]:
                break

            # don't predict on it if it's not on the fragment
            if not ops.isOnSurface(args, rowCoordinate, colCoordinate, self.surface_mask):
                colCoordinate += overlap_step
                continue

            # grab the sample and place it in output
            center_row = rowCoordinate+int(args["y_dimension"]/2)
            center_col = colCoordinate+int(args["x_dimension"]/2)
            zCoordinate = max(0,  self.surface_image[center_row, center_col] - args["surface_cushion"])

            if args["predict_depth"] > 1:
                #TODO this z-mapping mapping will eventually be something more intelligent
                zCoordinate += (depthCoordinate)
                #zCoordinate = depthCoordinate * int((self.volume.shape[2] - args["z_dimension"]) / args["predict_depth"])

            sample = (self.volume[rowCoordinate:rowCoordinate+args["y_dimension"], \
                    colCoordinate:colCoordinate+args["x_dimension"], zCoordinate:zCoordinate+args["z_dimension"]])
            predictionSamples[sample_count, 0:sample.shape[0], 0:sample.shape[1], 0:sample.shape[2]] = sample
            # populate the "prediction plus surface" with the initial surface value
            olap = overlap_step
            self.prediction_plus_surf[center_row-olap:center_row+olap, \
                    center_col-olap:center_col+olap] = np.max(self.volume[center_row, center_col]) #, max(0,min(self.volume.shape[2]-1,zCoordinate))]
            coordinates[sample_count] = [rowCoordinate, colCoordinate, depthCoordinate]

            # increment variables for next iteration
            colCoordinate += overlap_step
            sample_count += 1

        return (predictionSamples), (coordinates), [rowCoordinate, colCoordinate, depthCoordinate]



    def reconstruct3D(self, args, predictionValues, coordinates):
        center_step = int(round(self.prediction_overlap_step / 2))
        for i in range(coordinates.shape[0]):
            rowpoint = coordinates[i,0] + (int(args["x_dimension"] / 2))
            colpoint = coordinates[i,1] + (int(args["y_dimension"] / 2))
            zpoint = coordinates[i,2]
            predictionValue = predictionValues[i,1]

            self.all_preds.append(np.argmax(predictionValues[i,:]))
            if(self.ground_truth[rowpoint,colpoint]) > .9*self.max_truth:
                self.all_truth.append(1.0)
            else:
                self.all_truth.append(0.0)

            if(center_step > 0):
                self.prediction_volume[rowpoint-center_step:rowpoint+center_step, colpoint-center_step:colpoint+center_step, zpoint] = predictionValue
                self.prediction_plus_surf[rowpoint-center_step:rowpoint+center_step, colpoint-center_step:colpoint+center_step] *= predictionValue

            else:
                self.prediction_volume[rowpoint, colpoint, zpoint] = predictionValue
                self.prediction_plus_surf[rowpoint, colpoint] *= predictionValue


            if ops.isInTestSet(args, rowpoint, colpoint, self.volume.shape, self.train_bounds, self.train_portion):
                self.test_preds.append(self.all_preds[-1])
                self.test_truth.append(self.all_truth[-1])



    def savePrediction3D(self, args, iteration):
        # save individual pictures
        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.prediction_volume[:,:,d], predictionName='ink', depth=d)
        # save the average prediction across depths if depth is more than one
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.prediction_volume, axis=2), predictionName='ink-average')

        # save the output for samples not trained on
        if args["use_grid_training"]:
            # zero out the appropriate column
            if args["grid_test_square"] % 2 == 0:
                # test is on left side
                self.prediction_volume[ :, int(self.volume.shape[1] / 2):] = 0
            else:
                # test is on right side
                self.prediction_volume[ :, :int(self.volume.shape[1] / 2)] = 0

            n_rows = int(args["grid_n_squares"] / 2)
            voxels_per_row = int(self.volume.shape[0] / n_rows)
            start_row_number = int(args["grid_test_square"] / 2)
            end_row_number = start_row_number + 1
            self.prediction_volume[:(start_row_number*voxels_per_row), :] = 0
            self.prediction_volume[(end_row_number*voxels_per_row):, :] = 0

        else:
            rowBounds, colBounds = ops.bounds(args, [self.volume.shape[0], self.volume.shape[1]], self.train_bounds, self.train_portion)
            self.prediction_volume[rowBounds[0]:rowBounds[1], colBounds[0]:colBounds[1]] = 0

        for d in range(args["predict_depth"]):
            self.savePredictionImage(args, iteration, predictValues=self.prediction_volume[:,:,d], predictionName='ink-no-train', depth=d)
        if args["predict_depth"] > 1:
            self.savePredictionImage(args, iteration, predictValues = np.mean(self.prediction_volume, axis=2), predictionName='ink-average-no-train')

        # zero out the volume
        self.prediction_volume = np.zeros((self.volume.shape[0], self.volume.shape[1], args["predict_depth"]), dtype=np.float32)



    def savePredictionImage(self, args, iteration, predictValues=None, predictionName='ink', depth=0):
        if predictValues is None:
            #predictionImageInk = 65535 * ( (self.prediction_image_ink.copy() - np.min(self.prediction_image_ink)) / (np.amax(self.prediction_image_ink) - np.min(self.prediction_image_ink)) )
            predictionImage = (65535 * self.predictionImage).astype(np.uint16)
        else:
            predictionImage = (65535 * predictValues).astype(np.uint16)

        mn = np.min(self.prediction_plus_surf)
        mx = np.min(self.prediction_plus_surf)
        predictionPlusSurfImage = (65535 * (self.prediction_plus_surf - mn) / (mx-mn)).astype(np.uint16)

        try:
            os.makedirs(self.output_path + "/{}/".format(predictionName))
        except:
            pass

        # save the ink and surface predictions
        tiff.imsave(self.output_path + "/{}/prediction-iteration{}-depth{}.tif".format(predictionName, iteration, depth), predictionImage)
        # this doesn't work yet:
        # tiff.imsave(self.output_path + "/{}/predictionPlusSurf-iteration{}-depth{}.tif".format(predictionName, iteration, depth), predictionPlusSurfImage)
        tiff.imsave(self.output_path + "/training-{}.tif".format(iteration), self.training_image)

        # zero them out for the next predictions
        self.prediction_image_ink = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)
        self.prediction_image_surf = np.zeros((self.volume.shape[0], self.volume.shape[1]), dtype=np.float32)



    def savePredictionMetrics(self, args, iteration, minutes):
        print("\n\n\tMetrics for volume {}:".format(self.volume_number))
        all_confusion = confusion_matrix(self.all_truth, self.all_preds)
        test_confusion = confusion_matrix(self.test_truth, self.test_preds)
        all_confusion_norm = all_confusion.astype('float') / all_confusion.sum(axis=1)[:, np.newaxis]
        test_confusion_norm = test_confusion.astype('float') / test_confusion.sum(axis=1)[:, np.newaxis]
        print("\tNormalized confusion matrix for ALL points: \n{}".format(all_confusion_norm))
        print("\tNormalized confusion matrix for TEST points: \n{}".format(test_confusion_norm))

        #calculate metrics
        all_precision = precision_score(self.all_truth, self.all_preds)
        all_recall = recall_score(self.all_truth, self.all_preds)
        all_f1 = f1_score(self.all_truth, self.all_preds)
        test_precision = precision_score(self.test_truth, self.test_preds)
        test_recall = recall_score(self.test_truth, self.test_preds)
        test_f1 = f1_score(self.test_truth, self.test_preds)

        test_confusion_norm_list = [0,0,0,0]
        all_confusion_norm_list = [0,0,0,0]
        try:
            test_confusion_norm_list = test_confusion_norm.reshape(4).tolist()
            all_confusion_norm_list = all_confusion_norm.reshape(4).tolist()
        except:
            pass
        #save results in csv
        column_names = 'iteration, minutes, true positive papyrus, false positive ink, false positive papyrus, true positive ink, precision, recall, f1'
        self.test_results_norm.append([iteration] + [minutes] + test_confusion_norm_list + [test_precision] + [test_recall] + [test_f1])
        self.all_results_norm.append([iteration] + [minutes] + all_confusion_norm_list + [all_precision] + [all_recall] + [all_f1])
        np.savetxt(self.output_path + "confusion-all.csv", self.all_results_norm, fmt='%1.4f', header=column_names, delimiter=',')
        np.savetxt(self.output_path + "confusion-test.csv", self.test_results_norm, fmt='%1.4f', header=column_names, delimiter=',')

        # save description of this training session
        description = ""
        for arg in sorted(args.keys()):
            description += arg+": " + str(args[arg]) + "\n"
        np.savetxt(self.output_path +'description.txt', [description], delimiter=' ', fmt="%s")
        shutil.copy('model.py', self.output_path + 'network_model.txt')

        # zero-out predictions & images so next output is correct
        self.all_truth = []
        self.all_preds = []
        self.test_truth = []
        self.test_preds = []



    def wobbleVolume(self, args):
        wobble_start_time = time.time()
        self.wobbled_angle =  ((2*args["wobble_max_degrees"])*np.random.random_sample()) - args["wobble_max_degrees"]
        print("Wobbling volume {:.2f} degrees...".format(self.wobbled_angle))
        self.wobbled_axes = np.random.choice(3, 2, replace=False)
        self.wobbled_volume = rotate(self.volume, self.wobbled_angle, self.wobbled_axes, order=2, mode='nearest', reshape=False)
        print("Wobbling took {:.2f} minutes".format((time.time() - wobble_start_time)/60))



    def moveToNextPositiveSample(self, args):
        if self.train_index >= len(self.coordinate_pool):
            self.incrementEpoch(args)

        while self.coordinate_pool[self.train_index][2] == 0:
            if self.train_index + 1 == len(self.coordinate_pool):
                self.incrementEpoch(args)
            else:
                self.train_index += 1



    def moveToNextNegativeSample(self, args):
        if self.train_index >= len(self.coordinate_pool):
            self.incrementEpoch(args)

        while self.coordinate_pool[self.train_index][2] == 1:
            if self.train_index + 1 == len(self.coordinate_pool):
                self.incrementEpoch(args)
            else:
                self.train_index += 1



    def incrementEpoch(self, args):
        print("finished epoch")
        self.train_index = 0
        self.training_image = np.zeros(self.prediction_image_ink.shape, dtype=np.uint16)
        self.epoch += 1
        np.random.shuffle(self.coordinate_pool)



    def totalPredictions(self, args, overlap_step):
        #TODO don't predict off the fragment
        xSlides = (self.volume.shape[0] - args["x_dimension"]) / overlap_step
        ySlides = (self.volume.shape[1] - args["y_dimension"]) / overlap_step
        return int(xSlides * ySlides) * args["predict_depth"]
