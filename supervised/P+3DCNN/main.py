import tensorflow as tf
import numpy as np
import pdb
# import tifffile as tiff
import sys

import data
import model

args = {
    "trainingDataPath": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/",
    "mulitpower": "true",
    "groundTruthFile": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/GroundTruth-CarbonInk.png",
    "savePredictionPath": "/home/volcart/supervised-results/multipower-single-channel/",
    "saveModelPath": "/home/volcart/prelim-InkDetection/src/results/models/",
    "numChannels": 1,
    "numVolumes": 6,
    "cropX_low": int(sys.argv[1]),
    "cropX_high": int(sys.argv[2]),
    "cropY_low": int(sys.argv[3]),
    "cropY_high": int(sys.argv[4]),
    "x_Dimension": 25,
    "y_Dimension": 25,
    "z_Dimension": 71,
    "stride": 1,
    "predictBatchSize": 120, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 36,
    "dropout": 0.75,
    "trainingIterations": 30001,
    "predictStep": 1000,
    "displayStep": 10,
    "singleScanPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/"
}

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numChannels"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, args["saveModelPath"]+"model-epoch-4000.ckpt")
    epoch = 0
    while epoch < args["trainingIterations"]:
        # if epoch % args["grabNewSamples"] == 0:
        #     # trainingSamples, groundTruth = volume.getTrainingSample(args)
        #     trainingSamples, groundTruth = volume.getTrainingSample_MultipowerSingleChannel(args)
        #
        # pdb.set_trace()
        # randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
        # batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
        # batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

        # sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

        trainingSamples, groundTruth = volume.getTrainingSample_MultipowerSingleChannel(args)
        sess.run(optimizer, feed_dict={x: trainingSamples, y: groundTruth, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            acc, evaluatedLoss = sess.run([accuracy, loss], feed_dict={x: trainingSamples, y: groundTruth, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

        if epoch % args["predictStep"] == 0 and epoch > 0:
            # volume.emptyPredictionVolume(args)
            volume.initMultiplePredictionImages(args)
            total_num_predictions = volume.totalPredictions(args)

            startingCoordinates = [0,0,0]
            # predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)

            count = 1
            pdb.set_trace()
            while ((count-1)*args["predictBatchSize"]) < total_num_predictions:
                print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
                predictionValues = []
                for i in range(predictionSamples.shape[4]):
                    predictionValues.append(sess.run(pred, feed_dict={x: predictionSamples[:,:,:,:,i:i+1], keep_prob: 1.0}))
                volume.reconstruct_MulipowerSingleChannel(args, predictionValues, coordinates)
                # predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)
                count += 1
                volume.savePredictionImages(args, epoch)
            volume.savePredictionImage(args, epoch)
            # save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
            # print("Model saved in file: %s" % save_path)

        epoch = epoch + 1
