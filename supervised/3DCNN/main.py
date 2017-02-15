import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys

import data
import model

args = {
    "trainingDataPath": "/home/volcart/prelim-InkDetection/src/tmp/",
    "groundTruthFile": "/home/volcart/prelim-InkDetection/src/thresh-gt-mask.tif",
    "savePredictionPath": sys.argv[2],
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "subVolumeStepSize": int(sys.argv[3]),
    "z_Dimension": 90,
    "n_Classes": 2,
    "learningRate": 0.0001,
    "batchSize": 30,
    "dropout": 0.75,
    "trainingIterations": 10001,
    "predictStep": 75,
    "displayStep": 1
}

dataSamples, predictionSamples, dataGroundTruth, predictionGroundTruth, coordinates, volumeShape = data.inputData(args)

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
pred = tf.nn.softmax(pred)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    avgOutputVolume = []
    while epoch < args["trainingIterations"]:
        randomBatch = np.random.randint(dataSamples.shape[0] - args["batchSize"])
        batchX = dataSamples[randomBatch:randomBatch+args["batchSize"]]
        batchY = dataGroundTruth[randomBatch:randomBatch+args["batchSize"]]

        sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            print "Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc))

        if epoch % args["predictStep"] == 0 and epoch > 9000:

            trainingSet = sess.run(pred, feed_dict={x: dataSamples, keep_prob: 1.0})
            predictionSet = sess.run(pred, feed_dict={x:predictionSamples, keep_prob: 1.0})
            predictions = np.concatenate((trainingSet, predictionSet), 0)

            outputVolume = np.zeros((volumeShape[0], volumeShape[1]))
            for i in range(coordinates.shape[0]):
                if (predictions[i,0] > 0.5):
                    outputVolume[coordinates[i,0]:coordinates[i,0]+args["x_Dimension"], coordinates[i,1]:coordinates[i,1]+args["y_Dimension"]] = 1.0

            avgOutputVolume.append(outputVolume)
            if len(avgOutputVolume) > 1:
                npAvgOV = np.array(avgOutputVolume)
                npAvgOV = np.sum(npAvgOV, axis=0)
                npAvgOV = (npAvgOV - np.min(npAvgOV)) / (np.amax(npAvgOV) - np.min(npAvgOV))

                f = args["savePredictionPath"] + "out-" + str(epoch) + ".tif"
                tiff.imsave(f, np.array(npAvgOV*65535, dtype=np.uint16))


        epoch = epoch + 1
