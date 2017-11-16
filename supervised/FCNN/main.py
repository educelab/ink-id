import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys
import datetime
import data
import model
import time
import ops
import os
from sklearn.metrics import precision_score


print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###
    ### Input configuration ###
    "volumes": [
        {
            "name": "lunate-sigma",
            "microns_per_voxel":5,
            "data_path": "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
            "ground_truth":"/home/jack/devel/volcart/small-fragment-gt.tif",
            "surface_mask":"/home/jack/devel/volcart/small-fragment-outline.tif",
            "surface_data":"/home/jack/devel/volcart/small-fragment-smooth-surface.tif",
            "train_portion":.6,
            "train_bounds":3,# bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
            "use_in_training":True,
            "use_in_test_set":True,
            "make_prediction":True,
            "prediction_overlap_step":4
        },

    ],

    "z_dimension": 48,
    "xy_dimension_range": [64,128],
    ### Back off from the surface point some distance
    "surface_cushion" : 12,
    ### Network configuration ###
    "learning_rate": 0.001,
    "minibatch_size": 24,

    "dropout": 0.7,
    "training_iterations": 10000,
    "training_epochs": 1,
    "n_classes": 2,

    ### Data configuration ###
    "random_range" : 200,
    "use_jitter" : True,
    "jitter_range" : [-8, 8],
    "add_augmentation" : True,
    "train_portion" : .6, # Percent of division between train and predict regions
    "balance_samples" : True,
    "train_quadrants" : -1, # parameters: 0=test top left (else train) || 1=test top right || 2=test bottom left || 3=test bottom right
    "train_bounds" : 3, # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
    "surface_threshold": 20400,
    "restrict_surface": False,

    ### Output configuration ###
    "predict_step": 1000, # make a prediction every x steps
    "overlap_step": 4, # during prediction, predict on one sample for each _ by _ voxel square
    "display_step": 50, # output stats every x steps
    "predict_depth" : 1,
    "output_path": "/home/jack/devel/fall17/predictions/3dcnn/{}-{}-{}h".format(
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3]),
    "notes": "Testing effect of 1x1x7 'z-bar' convolutions, while minimizing parameters"
}


x = tf.placeholder(tf.float32, [None, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]])
y = tf.placeholder(tf.int32, [None, args["x_Dimension"], args["y_Dimension"]])
keep_prob = tf.placeholder(tf.float32)

pred, loss = model.buildModel(x, y, keep_prob, args)

optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(loss)
predictions = tf.cast(tf.argmax(pred, axis=3), tf.int32)
correct_pred = tf.equal(predictions, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#false_positives = tf.equal(tf.argmax(y,1) + 1, tf.argmax(pred, 1))
#false_positive_rate = tf.reduce_mean(tf.cast(false_positives, tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('xentropy-loss', loss)
#tf.summary.scalar('false_positive_rate', false_positive_rate)

merged = tf.summary.merge_all()
volume = data.Volume(args)

# create summary writer directory
if tf.gfile.Exists(args["output_path"]):
    tf.gfile.DeleteRecursively(args["output_path"])
tf.gfile.MakeDirs(args["output_path"])


# automatically dump "sess" once the full loop finishes
x = tf.placeholder(tf.float32, [None, max(args["xy_dimension_range"]), max(args["xy_dimension_range"]), args["z_dimension"]])
y = tf.placeholder(tf.float32, [None, max(args["xy_dimension_range"]), max(args["xy_dimension_range"]))
drop_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)
pred, loss = model.buildModel(x, y, drop_rate, args, training_flag)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
false_positives = tf.equal(tf.argmax(y,1) + 1, tf.argmax(pred, 1))
false_positive_rate = tf.reduce_mean(tf.cast(false_positives, tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.histogram('prediction_values', pred[:,1])
tf.summary.scalar('xentropy-loss', loss)
tf.summary.histogram('prediction_values', pred[:,1])
tf.summary.scalar('false_positive_rate', false_positive_rate)


merged = tf.summary.merge_all()
saver = tf.train.Saver()
best_test_f1 = 0.0
best_f1_iteration = 0
best_test_precision = 0.0
best_precision_iteration = 0
volume = data.Volume(args)

# create summary writer directory
if tf.gfile.Exists(args["output_path"]):
    tf.gfile.DeleteRecursively(args["output_path"])
tf.gfile.MakeDirs(args["output_path"])


# automatically dump "sess" once the full loop finishes
with tf.Session() as sess:
    print("Beginning train session...")
    print("Output directory: {}".format(args["output_path"]))

    train_writer = tf.summary.FileWriter(args["output_path"] + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args["output_path"] + '/test')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    predict_flag = False
    iteration = 0
    iterations_since_prediction = 0
    epoch = 0
    predictions_made = 0
    avgOutputVolume = []
    train_accs = []
    train_losses = []
    train_precs = []
    test_accs = []
    test_losses = []
    test_precs = []
    testX, testY = volume.getTestBatch(args)

    try:
        while iteration < args["training_iterations"]:
        #while iteration < args["training_iterations"]:

            predict_flag = False

            random_range = args["xy_dimension_range"][1] - args["xy_dimension_range"][0]
            x_dim = np.random.randint(random_range) + args["xy_dimension_range"][0]
            y_dim = np.random.randint(random_range) + args["xy_dimension_range"][0]
            batchX, batchY, epoch = volume.getMiniBatch(args, x_dim, y_dim)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batchX, y: batchY, drop_rate:args["dropout"], training_flag:True})
            train_writer.add_summary(summary, iteration)

            if iteration % args["display_step"] == 0:
                train_acc, train_loss, train_preds = \
                    sess.run([accuracy, loss, pred], feed_dict={x: batchX, y: batchY, drop_rate: 0.0, training_flag:False})
                test_acc, test_loss, test_preds, test_summary, = \
                    sess.run([accuracy, loss, pred, merged], feed_dict={x: testX, y: testY, drop_rate:0.0, training_flag:False})
                train_prec = precision_score(np.argmax(batchY, 1), np.argmax(train_preds, 1))
                test_prec = precision_score(np.argmax(testY, 1), np.argmax(test_preds, 1))
                test_f1 = fbeta_score(np.argmax(testY, 1), np.argmax(test_preds, 1), beta=args["fbeta_weight"])

                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                train_precs.append(train_prec)
                test_precs.append(test_prec)

                test_writer.add_summary(test_summary, iteration)


                print("Iteration: {}\t\tEpoch: {}".format(iteration, epoch))
                print("Train Loss: {:.3f}\tTrain Acc: {:.3f}\tInk Precision: {:.3f}".format(train_loss, train_acc, train_precs[-1]))
                print("Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tInk Precision: {:.3f}".format(test_loss, test_acc, test_precs[-1]))

                if (test_f1 > best_test_f1):
                    print("\tAchieved new peak f1 score! Saving model...\n")
                    best_test_f1 = test_f1
                    best_f1_iteration = iteration
                    save_path = saver.save(sess, args["output_path"] + '/models/model.ckpt', )

                if (test_acc > .9) and (test_prec > .7) and (iterations_since_prediction > 100): #or (test_prec > .8)  and (predictions_made < 4): # or (test_prec / args["numCubes"] < .05)
                    # make a full prediction if results are tentatively spectacular
                    predict_flag = True


            if (predict_flag) or (iteration % args["predict_step"] == 0 and iteration > 0):
                iterations_since_prediction = 0
                predictions_made += 1
                print("{} training iterations took {:.2f} minutes".format( \
                    iteration, (time.time() - start_time)/60))
                startingCoordinates = [0,0,0]
                predictionSamples, coordinates, nextCoordinates = volume.getPredictionBatch(args, startingCoordinates)

                print("Beginning predictions on volume...")
                while nextCoordinates is not None:
                    #TODO add back the output
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, training_flag:False})
                    volume.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volume.getPredictionBatch(args, nextCoordinates)
                minutes = ( (time.time() - start_time) /60 )
                volume.saveAllPredictions(args, iteration)
                volume.saveAllPredictionMetrics(args, iteration, minutes)

            if args["wobble_volume"] and iteration >= args["wobble_step"] and (iteration % args["wobble_step"]) == 0:
                # ex. wobble at iteration 1000, or after the prediction for the previous wobble
                volume.wobbleVolumes(args)
            iteration += 1
            iterations_since_prediction += 1


    except KeyboardInterrupt:
        # still make last prediction if interrupted
        pass

    # make one last prediction after everything finishes
    # use the model that performed best on the test set :)
    saver.restore(sess, args["output_path"] + '/models/model.ckpt')
    startingCoordinates = [0,0,0]
    predictionSamples, coordinates, nextCoordinates = volume.getPredictionBatch(args, startingCoordinates)
    count = 1
    print("Beginning predictions from best model (iteration {})...".format(best_f1_iteration))



print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
