import tensorflow as tf
import numpy as np
import pdb
# import tifffile as tiff
import sys

import data
import model

args = {
    "trainingDataPath": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/",
    "mulitpower": True,
    "groundTruthFile": "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/GroundTruth-CarbonInk.png",
    "savePredictionPath": "/home/volcart/supervised-results/multipower/",
    "saveModelPath": "/home/volcart/prelim-InkDetection/src/results/models/",
    "numChannels": 6,
    "x_Dimension": 25,
    "y_Dimension": 25,
    "z_Dimension": 71,
    "stride": 2,
    "numCubes": 100,
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 30,
    "dropout": 0.75,
    "trainingIterations": 50001,
    "grabNewSamples": 40,
    "predictStep": 5000,
    "displayStep": 10
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
    avgOutputVolume = []
    while epoch < args["trainingIterations"]:
        if epoch % args["grabNewSamples"] == 0:
            trainingSamples, groundTruth = volume.getTrainingSample(args)

        randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
        batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
        batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

        sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

        if epoch % args["predictStep"] == 0 and epoch > 0:
            volume.emptyPredictionVolume(args)

            startingCoordinates = [0,0,0]
            predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)

            count = 1
            while predictionSamples.shape[0] == args["numCubes"]: # TODO what about the special case where the volume is a perfect multiple of the numCubes?
                print("Predicting cubes " + str(count * args["numCubes"]))
                predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                volume.reconstruct(args, predictionValues, coordinates)
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                count += 1
                volume.savePredictionImage(args, epoch)
            predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
            volume.reconstruct(args, predictionValues, coordinates)
            volume.savePredictionImage(args, epoch)
            # save_path = saver.save(sess, args["saveModelPath"]+"model-epoch-"+str(epoch)+".ckpt")
            # print("Model saved in file: %s" % save_path)


        epoch = epoch + 1
