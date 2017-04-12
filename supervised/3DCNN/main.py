import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys
import matplotlib.pyplot as plt
import datetime
import data
import model
import time
import ops
from sklearn.metrics import precision_score

if len(sys.argv) < 6:
    print("Missing arguments")
    print("Usage: main.py  [xy Dimension]... [z Dimension]... [cushion]... [overlap step]... [layer1 neurons]... [data path]")
    exit()

print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###
    #"trainingDataPath": "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
    #"trainingDataPath" : "/home/jack/devel/volcart/small-fragment-data/nudge-0.50%/slices/"
    "trainingDataPath" : str(sys.argv[6]),
    #"surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/surf-output-21500/surface-points-21500.tif",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/polyfit-slices-degree32-cush16-thresh21500/surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif",
    "savePredictionPath": "/home/jack/devel/volcart/predictions/3dcnn/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": int(sys.argv[2]),
    "surfaceCushion" : int(sys.argv[3]),

    ### Network configuration ###
    "receptiveField" : [3,3,3],
    "learningRate": 0.0001,
    "batchSize": 10,
    "predictBatchSize": 200,
    "dropout": 0.5,
    "layer1_neurons": int(sys.argv[5]),
    "trainingIterations": 30001,
    "trainingEpochs": 2,
    "n_Classes": 2,

    ### Data configuration ###
    "numCubes" : 500,
    "addRandom" : False,
    "randomStep" : 10, # one in every randomStep non-ink samples will be a random brick
    "randomRange" : 200,
    "useJitter" : True,
    "jitterRange" : [-5, 5],
    "addAugmentation" : True,
    "train_portion" : .5,
    "trainBounds" : 3, # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
    "grabNewSamples": 20,
    "surfaceThresh": 20400,
    "restrictSurface": True,

    ### Output configuration ###
    "predictStep": 10000,
    "displayStep": 50,
    "overlapStep": int(sys.argv[4]),
    "predictDepth" : 1,
    "savePredictionFolder" : "/home/jack/devel/volcart/predictions/3dcnn/{}x{}x{}-{}-{}-{}h/".format(
            sys.argv[1], sys.argv[1], sys.argv[2],  #x, y, z
            datetime.datetime.today().timetuple()[1], # month
            datetime.datetime.today().timetuple()[2], # day
            datetime.datetime.today().timetuple()[3]), # hour
    "notes": "Set to four convolutional layers, down from six"
}


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_Classes"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, args)
ones_like_preds = tf.ones_like(pred)
zeros_like_trues = tf.zeros_like(y)

optimizer = tf.train.AdamOptimizer(learning_rate=args["learningRate"]).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
true_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1)))
false_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1) - 1))
ink_precision = tf.metrics.precision(tf.argmax(y,1), tf.argmax(pred,1))

init = tf.global_variables_initializer()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    predict_flag = False
    iteration = 0
    epoch = 0
    avgOutputVolume = []
    train_accs = []
    train_losses = []
    train_precs = []
    test_accs = []
    test_losses = []
    test_precs = []
    testX, testY = volume.getTrainingSample(args, testSet=True)
    while epoch < args["trainingEpochs"]:
            predict_flag = False

            batchX, batchY, epoch = volume.getTrainingBatch(args)

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

            if iteration % args["displayStep"] == 0:
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

                if ( (test_acc > .9) or (test_prec > .7) ) and iteration > 5000: # or (test_prec / args["numCubes"] < .05)
                    # make a full prediction if results are tentatively spectacular
                    predict_flag = True

                print("Iteration: {}\t\tEpoch: {}".format(iteration, epoch))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tInk Precision: {:.3f}".format(train_loss, train_acc, train_precs[-1]))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tInk Precision: {:.3f}".format(test_loss, test_acc, test_precs[-1]))


            if (predict_flag) or (iteration % args["predictStep"] == 0 and iteration > 0):
                print("{} training iterations took {:.2f} minutes".format( \
                    iteration, (time.time() - start_time)/60))
                startingCoordinates = [0,0,0]
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, startingCoordinates)

                count = 1
                total_predictions = volume.totalPredictions(args)
                total_prediction_batches = int(total_predictions / args["predictBatchSize"])
                print("Beginning predictions...")
                while ((count-1)*args["predictBatchSize"]) < total_predictions:
                    if (count % int(total_prediction_batches / 10) == 0):
                        #update UI at 10% intervals
                        print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_predictions))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                    volume.reconstruct3D(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample3D(args, nextCoordinates)
                    count += 1
                minutes = "{:.2f}".format( (time.time() - start_time) /60 )
                volume.savePrediction3D(args, iteration)
                volume.savePredictionMetrics(args, iteration, minutes)

                ops.graph(args, iteration, test_accs, test_losses, train_accs, train_losses, test_precs, train_precs)

            iteration = iteration + 1

print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
