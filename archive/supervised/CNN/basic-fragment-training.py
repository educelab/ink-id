import tensorflow as tf
import numpy as np
import pdb
import sys, os, shutil
import json
from sklearn.metrics import precision_score

import data
import model
import ops

config = {
    # FLAGS
    "surface_segmentation": True,
    "multipower": False,
    "crop": False,
    "addRandom": True,
    "useJitter": True,
    "addAugmentation": True,

    # PATHS
    "trainingDataPath": "/home/jack/devel/volcart/small-fragment-slices/",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-gt.tif",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-surface.tif",
    "savePredictionPath": "/home/jack/devel/volcart/cnn-predictions/",
    "saveModelPath": "/home/jack/devel/volcart/models/",

    # DATA
    "numVolumes": 1,
    "x_Dimension": 64,
    "y_Dimension": 64,
    "z_Dimension": 64,
    "stride": 8,
    "scalingFactor": 1.0,
    "randomStep": 10,
    "randomRange" : 200,
    "surfaceCushion" : 20,
    "jitterRange" : [-3, 3],
    "cropX_low": 550,
    "cropX_high": 600,
    "cropY_low": 450,
    "cropY_high": 500,

    # MODEL
    "numChannels": 1,
    "n_Classes": 2,
    "neurons": [4, 8, 16, 32],
    "filter": [3,3,3],
    "learningRate": 0.0001,

    # SESSION
    "epochs": 2,
    "train_iterations": 1000,
    "batchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "predictBatchSize": 200, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "dropout": 0.75,
    "predictStep": 500,
    "displayStep": 20,
    "saveModelStep": 500,
    "graphStep": 1000,
}

try: os.makedirs(config["savePredictionPath"])
except: pass # directory already made

# save description of this training session
description = ""
for arg in sorted(config.keys()):
    description += arg+": " + str(config[arg]) + "\n"
np.savetxt(config["savePredictionPath"] +'info.txt', [description], delimiter=' ', fmt="%s")
shutil.copy('model.py', config["savePredictionPath"] + 'network_model.txt')

x = tf.placeholder(tf.float32, [None, config["x_Dimension"], config["y_Dimension"], config["z_Dimension"], config["numChannels"]])
y = tf.placeholder(tf.float32, [None, config["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildBaseModel(x, y, keep_prob, config)
optimizer = tf.train.AdamOptimizer(learning_rate=config["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1) - 1))

volume = data.Volume(config)

with tf.Session() as sess:

    # NOTE: if restoring a session, comment out lines tf.global_variables_initializer() and sess.run(init)
    saver = tf.train.Saver()
    # saver.restore(sess, config["saveModelPath"]+"model-iteration-240000.ckpt") # NOTE: uncomment and change path to restore a graph model

    init = tf.global_variables_initializer()
    sess.run(init)
    # train_writer = tf.summary.FileWriter('/tmp/tb/', sess.graph)

    epoch = 1
    iteration = 0

    train_accs = []
    train_losses = []
    train_precs = []
    test_accs = []
    test_losses = []
    test_precs = []

    test_coordinates = volume.getRandomTestCoordinates(config)
    testX, testY = volume.getSamples(config, test_coordinates, predictionSamples=True)

    while iteration < config["train_iterations"] + 1:
        coordinates = volume.getTrainingCoordinates(config)
        for i in range(0, coordinates.shape[0], config["batchSize"]):
            iteration += 1
            if i < (coordinates.shape[0] - config["batchSize"]):
                batchX, batchY = volume.getSamples(config, coordinates[i:i+config["batchSize"],:])
            else: # edge case
                batchX, batchY = volume.getSamples(config, coordinates[i:coordinates.shape[0],:])

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: config["dropout"]})

            if iteration % config["displayStep"] == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x:testX, y:testY, keep_prob: 1.0})
                train_loss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_loss = sess.run(loss, feed_dict={x: testX, y:testY, keep_prob: 1.0})
                train_preds = sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0})
                test_preds = sess.run(pred, feed_dict={x: testX, y:testY, keep_prob:1.0})
                train_prec = precision_score(np.argmax(batchY, 1), np.argmax(train_preds, 1))
                test_prec = precision_score(np.argmax(testY, 1), np.argmax(test_preds, 1))
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                train_precs.append(train_prec)
                test_precs.append(test_prec)

                print("Epoch: {}\tIteration: {}\tTotal # iterations: {}".format(epoch, iteration, int(coordinates.shape[0] / config["batchSize"])))
                print("  Train Loss: {:.3f}\tTest Loss: {:.3f}".format(train_loss, test_loss))
                print("  Train Acc:  {:.3f}\tTest Acc:  {:.3f}".format(train_acc, test_acc))
                print("  Train Prec: {:.3f}\tTest Prec: {:.3f}".format(train_precs[-1], test_precs[-1]))

            # preiodically save the graph and model
            if iteration % config["graphStep"] == 0 and i > 0:
                ops.graph(config, i, test_accs, test_losses, train_accs, train_losses, test_precs, train_precs)
            if iteration % config["saveModelStep"] == 0 and i > 0:
                try: os.makedirs(config["saveModelPath"])
                except: pass # directory already made
                save_path = saver.save(sess, config["saveModelPath"]+"model-iteration-"+str(i)+".ckpt")
                print("Model saved in file: %s" % save_path)

            # make periodic predictions
            if iteration % config["predictStep"] == 0 and i > 0:
                coordinates = volume.get3DPredictionCoordinates(config)
                volume.initPredictionVolumes(config, config["numVolumes"])

                predictionValues = []
                for i in range(0,coordinates.shape[0],config["predictBatchSize"]):
                    print("Predicting iteration: {}\t Total number of iterations: {}".format(i, coordinates.shape[0]))
                    if i < (coordinates.shape[0] - config["predictBatchSize"]):
                        batchX, batchY = volume.getSamples(config, coordinates[i:i+config["predictBatchSize"],:], predictionSamples=True)
                    else: # edge case, predict on last batch of samples
                        batchX, batchY = volume.getSamples(config, coordinates[i:coordinates.shape[0],:], predictionSamples=True)

                    predictionValues.append(sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0}))

                    if i < (coordinates.shape[0] - config["predictBatchSize"]):
                        volume.reconstruct3D(config, predictionValues, coordinates[i:i+config["predictBatchSize"],:])
                    else: # edge case
                        volume.reconstruct3D(config, predictionValues, coordinates[i:coordinates.shape[0],:])

                volume.savePredictionVolumes(config, epoch)

        epoch = epoch + 1


    # NOTE ----------------------------
        # make a single prediction after training has completed

    coordinates = volume.get3DPredictionCoordinates(config)
    volume.initPredictionVolumes(config, config["numVolumes"])

    predictionValues = []
    for i in range(0,coordinates.shape[0],config["predictBatchSize"]):
        print("Predicting iteration: {}\t Total number of iterations: {}".format(i, coordinates.shape[0]))
        if i < (coordinates.shape[0] - config["predictBatchSize"]):
            batchX, batchY = volume.getSamples(config, coordinates[i:i+config["predictBatchSize"],:])
        else: # edge case, predict on last batch of samples
            batchX, batchY = volume.getSamples(config, coordinates[i:coordinates.shape[0],:])

        predictionValues.append(sess.run(pred, feed_dict={x: batchX, keep_prob: 1.0}))

        if i < (coordinates.shape[0] - config["predictBatchSize"]):
            volume.reconstruct3D(config, predictionValues, coordinates[i:i+config["predictBatchSize"],:])
        else: # edge case
            volume.reconstruct3D(config, predictionValues, coordinates[i:coordinates.shape[0],:])

    volume.savePredictionVolumes(config, epoch)
