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
        1. x_Dimension & y_Dimension
        2. subVolumeStepSize
        3. saveSamplePath
        4. saveVideoPath
'''

args = {
    "dataPath": "/home/volcart/Pictures/LampBlackTest-2016.volpkg/paths/20161205161113/flattened_i1/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": 50,
    "subVolumeStepSize": int(sys.argv[2]),
    "learningRate": 0.001,
    "batchSize": 10,
    "dropout": 0.75,
    "trainingIterations": 10001,
    "analyzeStep": 10000,
    "displayStep": 5,
    "saveSamplePath": sys.argv[3],
    "saveVideoPath": sys.argv[4],
    "ffmpegFileListPath": "/home/volcart/VAE_Layers/concatFileList.txt"
}

overlappingCoordinates, trainingData = data.readData(args)

x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

l1, l2, l3, l4, l5, l6, finalLayer, loss = VAE.buildModel(x, args)
correct_prediction = tf.equal(tf.argmax(x,1), tf.argmax(finalLayer,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    while epoch < args["trainingIterations"]:

        randomBatch = np.random.randint(trainingData.shape[0] - args["batchSize"])
        batchX = trainingData[randomBatch:randomBatch+args["batchSize"]]

        _ = sess.run(optimizer, feed_dict={x: batchX, keep_prob: args["dropout"]})
        if epoch % args["displayStep"] == 0:
            evaluatedLoss = sess.run(loss, feed_dict={x: batchX, keep_prob: 1.0})
            evaluatedAccuracy = sess.run(accuracy, feed_dict={x: batchX, keep_prob: 1.0})
            print("Epoch: " + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Accuracy: " + str(np.mean(evaluatedAccuracy)))

        if epoch % args["analyzeStep"] == 0:
            l1_Samples, l2_Samples, l3_Samples, l4_Samples, l5_Samples, l6_Samples = sess.run([l1, l2, l3, l4, l5, l6], feed_dict={x: trainingData, keep_prob: 1.0})
            data.saveSamples(args, overlappingCoordinates, [l1_Samples, l2_Samples, l3_Samples, l4_Samples, l5_Samples, l6_Samples])

        epoch += 1
