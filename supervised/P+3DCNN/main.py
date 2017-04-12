import tensorflow as tf
import numpy as np
import pdb
import sys
import json

import data
import model

args = {
    "trainingDataPath": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/",
    "mulitpower": "true",
    "groundTruthFile": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/GroundTruth-CarbonInk.png",
    "savePredictionPath": sys.argv[5],
    "saveModelPath": sys.argv[7],
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
    "batchSize": 36, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "dropout": 0.75,
    "trainingIterations": 30001,
    "predictStep": 2000,
    "displayStep": 100,
    "saveModelStep": 5000,
    "singleScanPath": "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/layers/full-layers/after-rotate/",
    "experimentType": sys.argv[6],
    "scalingFactor": float(sys.argv[8])
}

with open(args["savePredictionPath"]+'info.txt', 'w') as outfile:
    json.dump(args, outfile, indent=4)

if args["experimentType"] == "multipower-single-channel":
    x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numChannels"]])
elif args["experimentType"] == "multipower-multinetwork":
    x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"], args["numVolumes"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

if args["experimentType"] == "multipower-single-channel":
    pred, loss = model.buildModel(x, y, args)
elif args["experimentType"] == "multipower-multinetwork":
    pred, loss = model.buildMultiNetworkModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, args["saveModelPath"]+"run1/model-epoch-10000.ckpt") # NOTE: uncomment and change path to restore a graph model
    epoch = 9999
    while epoch < args["trainingIterations"]:

        if args["experimentType"] == "multipower-single-channel":
            trainingSamples, groundTruth = volume.getTrainingSample_MultipowerSingleChannel(args)
        elif args["experimentType"] == "multipower-multinetwork":
            trainingSamples, groundTruth = volume.getTrainingSample(args)

        sess.run(optimizer, feed_dict={x: trainingSamples, y: groundTruth, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            acc, evaluatedLoss = sess.run([accuracy, loss], feed_dict={x: trainingSamples, y: groundTruth, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

        if epoch % args["predictStep"] == 0 and epoch > 0:

            total_num_predictions = volume.totalPredictions(args)
            startingCoordinates = [0,0,0]
            count = 1

            if args["experimentType"] == "multipower-single-channel":
                volume.initPredictionImages(args)
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)
                while ((count-1)*args["predictBatchSize"]) < total_num_predictions:
                    print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
                    predictionValues = []
                    for i in range(predictionSamples.shape[4]):
                        predictionValues.append(sess.run(pred, feed_dict={x: predictionSamples[:,:,:,:,i:i+1], keep_prob: 1.0}))
                    volume.reconstruct_MulipowerSingleChannel(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, nextCoordinates)
                    count += 1
                    volume.savePredictionImages(args, epoch)
                volume.savePredictionImages(args, epoch)

            elif args["experimentType"] == "multipower-multinetwork":
                volume.emptyPredictionImage(args)
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, startingCoordinates)
                while ((count-1)*args["predictBatchSize"]) < total_num_predictions:
                    print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_num_predictions))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                    volume.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample_MultipowerSingleChannel(args, nextCoordinates)
                    count += 1
                    volume.savePredictionImage(args, epoch)
                volume.savePredictionImage(args, epoch)


        # NOTE: uncomment to save model
        if epoch % args["saveModelStep"] == 0 and epoch > 0:
            save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
            print("Model saved in file: %s" % save_path)

        epoch = epoch + 1
