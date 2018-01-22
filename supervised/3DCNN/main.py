import tensorflow as tf
import numpy as np
import pdb
import tifffile as tiff
import sys
import datetime
import data
import multidata
import model
import time
import ops
import os
from sklearn.metrics import precision_score, fbeta_score


print("Initializing...")
start_time = time.time()

args = {
    ### Input configuration ###
    "volumes": [

        {
            "name": "lunate-sigma",
            "microns_per_voxel":5,
            "data_path":'/home/jack/devel/volcart/lunate-sigma/training-slices/',
            "ground_truth":'/home/jack/devel/volcart/lunate-sigma/small-fragment-random-truth-33p.tif',
            "surface_mask":'/home/jack/devel/volcart/lunate-sigma/small-fragment-surface.tif',
            "surface_data":'/home/jack/devel/volcart/lunate-sigma/small-fragment-outline.tif',
            "train_portion":.5,
            "train_bounds":3,# bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
            "use_in_training":True,
            "use_in_test_set":True,
            "make_prediction":True,
            "prediction_overlap_step":4,
            "predict_depth":1,
            "scale_factor":1,

        },


    ],

    # hackish way to make dimensions even
    "x_dimension": 96,
    "y_dimension": 96,
    "z_dimension": 48,

    ### Back off from the surface point some distance
    "surface_cushion" : 8,

    ### Network configuration ###
    "use_multitask_training": False,
    "shallow_learning_rate":.001,
    "learning_rate": .0001,
    "batch_size": 30,
    "prediction_batch_size": 500,
    "filter_size" : [3,3,3],
    "dropout": 0.5,
    "neurons": [16, 8, 4, 2],
    #"neurons": [int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1])],
    "training_iterations": 100000,
    "training_epochs": 10, #int(3 / pow(float(sys.argv[1]), 2)),
    "n_classes": 2,
    "pos_weight": .5,
    "batch_norm_momentum": .9,
    "fbeta_weight": 0.2,

    ### Data configuration ###
    "wobble_volume" : False,
    "wobble_step" : 1000,
    "wobble_max_degrees" : 2,
    "num_test_cubes" : 200,
    "add_random" : False,
    "random_step" : 10, # one in every randomStep non-ink samples will be a random brick
    "random_range" : 200,
    "use_jitter" : True,
    "jitter_range" : [-3, 3],
    "add_augmentation" : True,
    "balance_samples" : True,
    "use_grid_training": False,
    "grid_n_squares":4,
    "grid_test_square": 3,
    "surface_threshold": 20400,
    "restrict_surface": True,
    "truth_cutoff_low": .2,
    "truth_cutoff_high": .8,

    ### Output configuration ###
    "predict_step": 5000, # make a prediction every x steps
    "display_step": 20, # output stats every x steps
    "predict_depth" : 2,
    "output_path": "/home/jack/devel/spring18/3dcnn-predictions/{}-{}-{}h".format(
        datetime.datetime.today().timetuple()[1],
        datetime.datetime.today().timetuple()[2],
        datetime.datetime.today().timetuple()[3],),
    "notes": ""
}


x = tf.placeholder(tf.float32, [None, args["x_dimension"], args["y_dimension"], args["z_dimension"]])
y = tf.placeholder(tf.float32, [None, args["n_classes"]])
drop_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)


if args["use_multitask_training"]:
    pred, shallow_loss, loss = model.buildMultitaskModel(x, y, drop_rate, args, training_flag)
    shallow_optimizer = tf.train.AdamOptimizer(learning_rate=args["shallow_learning_rate"]).minimize(shallow_loss)
else:
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

'''Summary images need revision, running them on every iteration really slows it down
sample_view_z = tf.reshape(tf.reduce_mean(x[0], axis=2), [1, args["x_dimension"], args["y_dimension"], 1])
sample_view_y = tf.reshape(tf.reduce_mean(x[0], axis=0), [1, args["x_dimension"], args["z_dimension"], 1])
sample_view_x = tf.reshape(tf.reduce_mean(x[0], axis=1), [1, args["y_dimension"], args["z_dimension"], 1])
tf.summary.image('z-project', sample_view_z, max_outputs=1)
tf.summary.image('y-project', sample_view_y, max_outputs=1)
tf.summary.image('x-project', sample_view_x, max_outputs=1)
'''
if args["use_multitask_training"]:
    tf.summary.scalar('xentropy-shallow-loss', loss)
tf.summary.scalar('false_positive_rate', false_positive_rate)


merged = tf.summary.merge_all()
saver = tf.train.Saver()
best_test_f1 = 0.0
best_f1_iteration = 0
best_test_precision = 0.0
best_precision_iteration = 0
best_test_acc = 0.0
best_acc_iteration = 0
best_model_path = ''
volumes = multidata.VolumeSet(args)

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
    testX, testY = volumes.getTestBatch(args)

    try:
        #while iteration < args["training_iterations"]:
        while epoch < args["training_epochs"]:

            predict_flag = False

            batchX, batchY, epoch = volumes.getTrainingBatch(args)
            if args["use_multitask_training"]:
                summary, _, _ = sess.run([merged, optimizer, shallow_optimizer], feed_dict={x: batchX, y: batchY, drop_rate:args["dropout"], training_flag:True})
            else:
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
                    best_model_path = saver.save(sess, args["output_path"] + '/models/model', global_step=iteration)

                if (test_acc > .9) and (test_prec > .7) and (iterations_since_prediction > 100): #or (test_prec > .8)  and (predictions_made < 4): # or (test_prec / args["numCubes"] < .05)
                    # make a full prediction if results are tentatively spectacular
                    predict_flag = True


            if (predict_flag) or (iteration % args["predict_step"] == 0 and iteration > 0):
                save_path = saver.save(sess, args["output_path"] + '/models/model', global_step=iteration)
                iterations_since_prediction = 0
                predictions_made += 1
                print("{} training iterations took {:.2f} minutes".format( \
                    iteration, (time.time() - start_time)/60))
                startingCoordinates = [0,0,0]
                predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, startingCoordinates)

                print("Beginning predictions on volumes...")
                while nextCoordinates is not None:
                    #TODO add back the output
                    predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, training_flag:False})
                    volumes.reconstruct(args, predictionValues, coordinates)
                    predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, nextCoordinates)
                minutes = ( (time.time() - start_time) /60 )
                volumes.saveAllPredictions(args, iteration)
                volumes.saveAllPredictionMetrics(args, iteration, minutes)

            if args["wobble_volume"] and iteration >= args["wobble_step"] and (iteration % args["wobble_step"]) == 0:
                # ex. wobble at iteration 1000, or after the prediction for the previous wobble
                volumes.wobbleVolumes(args)
            iteration += 1
            iterations_since_prediction += 1


    except KeyboardInterrupt:
        # still make last prediction if interrupted
        pass

    # make one last prediction after everything finishes
    # use the model that performed best on the test set :)
    saver.restore(sess, "{}".format(best_model_path))
    startingCoordinates = [0,0,0]
    predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, startingCoordinates, v_olap=2)
    count = 1
    print("Beginning predictions from best model (iteration {})...".format(best_f1_iteration))
    while nextCoordinates is not None:
        #TODO add back the output
        predictionValues = sess.run(pred, feed_dict={x: predictionSamples, drop_rate: 0.0, training_flag:False})
        volumes.reconstruct(args, predictionValues, coordinates)
        predictionSamples, coordinates, nextCoordinates = volumes.getPredictionBatch(args, nextCoordinates, v_olap=2)
    minutes = ( (time.time() - start_time) /60 )
    volumes.saveAllPredictions(args, best_f1_iteration)
    volumes.saveAllPredictionMetrics(args, best_f1_iteration, minutes)



print("full script took {:.2f} minutes".format((time.time() - start_time)/60))
