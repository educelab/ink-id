import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys

import data
import model

if len(sys.argv) < 3:
    print("Missing arguments")
    print("Usage: main.py [xy Dimension]... [ overlap step]...")
    exit()

args = {
    "trainingDataPath": "/home/jack/devel/volcart/small-fragment-data/tmp/",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/YZ-aligned-mask-2.tif",
    "savePredictionPath": "/home/jack/devel/volcart/predictions/3dcnn/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": 90,
    "overlapStep": int(sys.argv[2]),
    "numCubes" : 250,
    "n_Classes": 3,
    "learningRate": 0.001,
    "batchSize": 30,
    "predictBatchSize": 5000,
    "dropout": 0.75,
    "trainingIterations": 30001,
    "predictStep": 30000,
    "displayStep": 20,
    "grabNewSamples": 100
}


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    avgOutputVolume = []
    while epoch < args["trainingIterations"]:
            if epoch % args["grabNewSamples"] == 0:
                trainingSamples, groundTruth = volume.getTrainingSample(args)

            #TODO: add periodic shuffle
            randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
            batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
            batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

            if epoch % args["displayStep"] == 0:
                acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                evaluatedLoss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(acc)))

            if epoch % args["predictStep"] == 0 and epoch > 0:
                startingCoordinates = [0,0,0]
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)

                count = 1
                while predictionSamples.shape[0] == args["numCubes"]: # TODO what about the special case where the volume is a perfect multiple of the numCubes?
                    print("Predicting cubes {}".format(str(count * args["numCubes"])))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                    volume.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                    count += 1
                volume.savePredictionImage(args)

            epoch = epoch + 1
