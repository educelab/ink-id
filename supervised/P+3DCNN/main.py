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

if len(sys.argv) < 5:
    print("Missing arguments")
    print("Usage: main.py  [xy Dimension]... [z Dimension]... [cushion]... [overlap step]... [dropout probability]... [data path]")
    exit()

print("Initializing...")
start_time = time.time()

args = {
    "trainingDataPaths" : [
        "/home/jack/devel/volcart/CarbonPhantomFeb/CarbonPhantom_MP_40kV_200uA_4000ms_2",
        "/home/jack/devel/volcart/CarbonPhantomFeb/CarbonPhantom_MP_60kV_133uA_2000ms_2",
        "/home/jack/devel/volcart/CarbonPhantomFeb/CarbonPhantom_MP_70kV_112uA_1700ms_2",
        "/home/jack/devel/volcart/CarbonPhantomFeb/CarbonPhantom_MP_90kV_88uA_1150ms_2",
        "/home/jack/devel/volcart/CarbonPhantomFeb/CarbonPhantom_MP_1100kV_72uA_900ms_2",
    ]
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/polyfit-slices-degree32-cush16-thresh21500/surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif",
    "savePredictionPath": "/home/jack/devel/volcart/predictions/3dcnn/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": int(sys.argv[2]),
    "savePredictionFolder" : "/home/jack/devel/volcart/predictions/3dcnn/{}x{}x{}-{}-{}-{}h/".format(
        sys.argv[1], sys.argv[1], sys.argv[2],
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3]),
    "surfaceCushion" : int(sys.argv[3]),
    "overlapStep": int(sys.argv[4]),
    "receptiveField" : [3,3,3],
    "numCubes" : 250,
    "addRandom" : True,
    "randomRange" : 200,
    "jitterRange" : [-12, 12],
    "n_Classes": 2,
    "train_portion" : .5,
    "learningRate": 0.0001,
    "batchSize": 30,
    "predictBatchSize": 100,
    "dropout": float(sys.argv[5]),
    "trainingIterations": 40001,
    "predictStep": 1000,
    "displayStep": 20,
    "grabNewSamples": 20,
    "surfaceThresh": 20000,
    "notes": "Changed stride of second layer to 1"
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
# for true positive, 1 * (1 - 1) = 0
false_positives = tf.count_nonzero(tf.argmax(pred,1) * (tf.argmax(y,1) - 1))
init = tf.global_variables_initializer()

volume = data.Volume(args)

with tf.Session() as sess:
    sess.run(init)
    predict_flag = False
    epoch = 0
    avgOutputVolume = []
    train_accs = []
    train_losses = []
    train_fps = []
    test_accs = []
    test_losses = []
    test_fps = []
    testX, testY = volume.getTrainingSample(args, testSet=True)
    while epoch < args["trainingIterations"]:
            predict_flag = False

            if epoch % args["grabNewSamples"] == 0:
                trainingSamples, groundTruth = volume.getTrainingSample(args)

            if epoch % args["grabNewSamples"] % int(args["numCubes"]/4) == 0:
                # periodically shuffle input and labels in parallel
                all_pairs = list(zip(trainingSamples, groundTruth))
                np.random.shuffle(all_pairs)
                trainingSamples, groundTruth = zip(*all_pairs)
                trainingSamples = np.array(trainingSamples)
                groundTruth = np.array(groundTruth)


            randomBatch = np.random.randint(trainingSamples.shape[0] - args["batchSize"])
            batchX = trainingSamples[randomBatch:randomBatch+args["batchSize"]]
            batchY = groundTruth[randomBatch:randomBatch+args["batchSize"]]

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: args["dropout"]})

            if epoch % args["displayStep"] == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x:testX, y:testY, keep_prob: 1.0})
                train_loss = sess.run(loss, feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
                test_loss = sess.run(loss, feed_dict={x: testX, y:testY, keep_prob: 1.0})
                train_fp = sess.run(false_positives, feed_dict={x: batchX, y:batchY, keep_prob:1.0})
                test_fp = sess.run(false_positives, feed_dict={x:testX, y:testY, keep_prob:1.0})
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                test_fps.append(test_fp / args["numCubes"])
                train_fps.append(train_fp / args["batchSize"])

                if (test_fp / args["numCubes"] < .05) or (test_acc > .9):
                    # fewer than 5% false positives, make a full prediction
                    predict_flag = True

                print("Epoch: {}".format(epoch))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tFp: {}".format(train_loss, train_acc, train_fp))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tFp: {}".format(test_loss, test_acc, test_fp))
                # + str(epoch) + "  Loss: " + str(np.mean(evaluatedLoss)) + "  Acc: " + str(np.mean(train_acc)))

            if (predict_flag) or (epoch % args["predictStep"] == 0 and epoch > 0):
                print("{} training iterations took {:.2f} minutes".format( \
                    epoch, (time.time() - start_time)/60))
                startingCoordinates = [0,0]
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, startingCoordinates)

                count = 1
                total_predictions = volume.totalPredictions(args)
                total_prediction_batches = int(total_predictions / args["predictBatchSize"])
                print("Beginning predictions...")
                while ((count-1)*args["predictBatchSize"]) < total_predictions:
                    if (count % int(total_prediction_batches / 10) == 0):
                        #update UI at 10% intervals
                        print("Predicting cubes {} of {}".format((count * args["predictBatchSize"]), total_predictions))
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, keep_prob: 1.0})
                    volume.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionSample(args, nextCoordinates)
                    count += 1
                volume.savePredictionImage(args, epoch)

                plt.figure(1)
                plt.clf()
                plt.subplot(311) # losses
                axes = plt.gca()
                axes.set_ylim([0,np.median(test_losses)+1])
                xs = np.arange(len(train_accs))
                plt.plot(train_losses, 'k.')
                plt.plot(test_losses, 'g.')
                plt.subplot(312) # accuracies
                plt.plot(train_accs, 'k.')
                plt.plot(xs, np.poly1d(np.polyfit(xs, train_accs, 1))(xs), color='k')
                plt.plot(test_accs, 'g.')
                plt.plot(xs, np.poly1d(np.polyfit(xs, test_accs, 1))(xs), color='g')
                plt.subplot(313) # false positives
                plt.plot(train_fps, 'k.')
                plt.plot(test_fps, 'g.')
                plt.savefig(args["savePredictionFolder"]+"plots-{}.png".format(epoch))
                #plt.show()

            epoch = epoch + 1

print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
