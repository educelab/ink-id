'''
main.py:
    - main program for the unsupervised VAE
        - input training data
        - build model
        - train
        - capture neuron output
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"

import tensorflow as tf
import pdb
import sys
import numpy as np

import data as data
import model as VAE

'''
    ARGS from experimentScript.py
        1. data path
        2. x_Dimension & y_Dimension & z_Dimension
        3. saveSamplesPath
'''

args = {
    "trainingDataPath": sys.argv[1],
    "x_Dimension": int(sys.argv[2]),
    "y_Dimension": int(sys.argv[2]),
    "z_Dimension": int(sys.argv[2]),
    "overlapStep": 2,
    "numCubes": 250,
    "learningRate": 0.001,
    "batchSize": 30,
    "dropout": 0.75,
    "trainingIterations": 30001,
    "grabNewSamples": 100,
    "analyzeStep": 30000,
    "displayStep": 5,
    "saveSamplesPath": sys.argv[3],
    "saveVideoPath": "/home/volcart/UnsupervisedResults/HercFragment/VAE/",
    "ffmpegFileListPath": "/home/volcart/VAE_Layers/concatFileList.txt",
}

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

l1, l2, l3, l4, l5, l6, finalLayer, loss = VAE.buildModel(x, args)
correct_prediction = tf.equal(tf.argmax(x,1), tf.argmax(finalLayer,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
init = tf.global_variables_initializer()

volume = data.Volume(args)
layers = [[l1], [l2], [l3], [l4], [l5], [l6]]
for i in range(len(layers)):
    for j in range(layers[i][0].get_shape().as_list()[4]):
        layers[i].append(data.PredictionVolume(args, volume))
sampleRates = [1, 2, 4, 8, 4, 2]

with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    while epoch < args["trainingIterations"]:
        if epoch % args["grabNewSamples"] == 0:
            trainingSamples = volume.getTrainingSample(args)

        randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
        batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]

        _ = sess.run(optimizer, feed_dict={x: batchX, keep_prob: args["dropout"]})

        if epoch % args["displayStep"] == 0:
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, keep_prob: 1.0})
            evaluatedAccuracy = sess.run(accuracy, feed_dict={x: batchX, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Accuracy: " + str(np.mean(evaluatedAccuracy)))

        if epoch % args["analyzeStep"] == 0 and epoch > 0:
            for i in range(len(layers)):
                print("Predicting layer number: " + str(i))
                for j in range(1,len(layers[i])):
                    print("Predicting sample number: " + str(j))
                    startingCoordinates = [0,0,0]
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)
                    while predictionSamples.shape[0] == args["numCubes"]: # TODO what about the special case where the volume is a perfect multiple of the numCubes?
                        predictionValues = sess.run(layers[i][0], feed_dict={x: predictionSamples, keep_prob: 1.0})
                        layers[i][j].reconstruct(args, predictionValues[:,:,:,:,j-1], coordinates, sampleRates[i])
                        predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                    predictionValues = sess.run(layers[i][0], feed_dict={x: predictionSamples, keep_prob: 1.0})
                    layers[i][j].reconstruct(args, predictionValues[:,:,:,:,j-1], coordinates, sampleRates[i])

                    layers[i][j].trimZeros()
                    layers[i][j].savePredictionSlices(args, i, j)
                    # layers[i][j].savePredictionVolume(args)
                    # layers[i][j].savePredictionVideo(args)

        epoch += 1
