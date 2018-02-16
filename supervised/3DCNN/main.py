"""
Train an ink classifier and produce predicted output for a volume.
"""

import argparse
import datetime
import os
import time

from sklearn.metrics import precision_score, fbeta_score
import tensorflow as tf
import numpy as np

from volumeset import VolumeSet
import model


def main():
    """Run the training and prediction process."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', metavar='path', required=True,
                        help='path to volume data (slices directory)')
    parser.add_argument('--groundtruth', metavar='path', required=True,
                        help='path to ground truth image')
    parser.add_argument('--surfacemask', metavar='path', required=True,
                        help='path to surface mask image')
    parser.add_argument('--surfacedata', metavar='path', required=True,
                        help='path to surface data')
    parser.add_argument('--gridtestsquare', metavar='num', default=1, type=int,
                        help='index of grid test square for this k-fold run')
    parser.add_argument('--outputdir', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    params = {
        'volumes': [
            {
                'name': 'lunate-sigma',
                'microns_per_voxel': 5,
                'data_path': args.data,
                'ground_truth': args.groundtruth,
                'surface_mask': args.surfacemask,
                'surface_data': args.surfacedata,
                'train_portion': .6,
                # train_bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
                'train_bounds': 3,
                'use_in_training': True,
                'use_in_test_set': True,
                'make_prediction': True,
                'prediction_overlap_step': 4,
            },

        ],

        'x_dimension': 96,
        'y_dimension': 96,
        'z_dimension': 48,

        # Back off from the surface point some distance
        'surface_cushion': 10,

        # Network configuration
        'use_multitask_training': False,
        'shallow_learning_rate': .001,
        'learning_rate': .001,
        'batch_size': 30,
        'prediction_batch_size': 400,
        'filter_size': [3, 3, 3],
        'dropout': 0.5,
        'neurons': [16, 8, 4, 2],
        'training_iterations': 1000000,
        'training_epochs': 2,
        'n_classes': 2,
        'pos_weight': .5,
        'batch_norm_momentum': .9,
        'fbeta_weight': 0.3,

        # Data configuration
        'wobble_volume': False,
        'wobble_step': 1000,
        'wobble_max_degrees': 2,
        'num_test_cubes': 400,
        'add_random': False,
        'random_step': 10,  # one in every randomStep non-ink samples will be a random brick
        'random_range': 200,
        'use_jitter': True,
        'jitter_range': [-4, 4],
        'add_augmentation': True,
        'balance_samples': False,
        'use_grid_training': True,
        'grid_n_squares': 10,
        'grid_test_square': args.gridtestsquare,
        'surface_threshold': 20400,
        'restrict_surface': True,
        'truth_cutoff_low': .15,
        'truth_cutoff_high': .85,

        # Output configuration
        'predict_step': 10000, # make a prediction every x steps
        'overlap_step': 4, # during prediction, predict on one sample for each _ by _ voxel square
        'display_step': 100, # output stats every x steps
        'predict_depth' : 1,
        'output_path': os.path.join(args.outputdir, '3dcnn-predictions/{}-{}-{}h'.format(
            datetime.datetime.today().timetuple()[1],
            datetime.datetime.today().timetuple()[2],
            datetime.datetime.today().timetuple()[3])),
        'notes': '',
    }

    x = tf.placeholder(tf.float32, [None, params['x_dimension'], params['y_dimension'], params['z_dimension']])
    y = tf.placeholder(tf.float32, [None, params['n_classes']])
    drop_rate = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)

    if params['use_multitask_training']:
        pred, shallow_loss, loss = model.build_multitask_model(x, y, drop_rate, params, training_flag)
        shallow_optimizer = tf.train.AdamOptimizer(learning_rate=params['shallow_learning_rate']).minimize(shallow_loss)
    else:
        pred, loss = model.build_model(x, y, drop_rate, params, training_flag)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    false_positives = tf.equal(tf.argmax(y, 1) + 1, tf.argmax(pred, 1))
    false_positive_rate = tf.reduce_mean(tf.cast(false_positives, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('prediction_values', pred[:, 1])
    tf.summary.scalar('xentropy-loss', loss)
    tf.summary.histogram('prediction_values', pred[:, 1])

    if params['use_multitask_training']:
        tf.summary.scalar('xentropy-shallow-loss', loss)
    tf.summary.scalar('false_positive_rate', false_positive_rate)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=None)
    best_test_f1 = 0.0
    best_f1_iteration = 0
    volumes = VolumeSet(params)

    # create summary writer directory
    if tf.gfile.Exists(params['output_path']):
        tf.gfile.DeleteRecursively(params['output_path'])
    tf.gfile.MakeDirs(params['output_path'])

    # automatically dump 'sess' once the full loop finishes
    with tf.Session() as sess:
        print('Beginning train session...')
        print('Output directory: {}'.format(params['output_path']))

        train_writer = tf.summary.FileWriter(params['output_path'] + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(params['output_path'] + '/test')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        predict_flag = False
        iteration = 0
        iterations_since_prediction = 0
        epoch = 0
        predictions_made = 0
        train_accs = []
        train_losses = []
        train_precs = []
        test_accs = []
        test_losses = []
        test_precs = []
        train_minutes = []
        test_x, test_y = volumes.getTestBatch(params)

        try:
            while epoch < params['training_epochs']:
            # while iteration < params['training_iterations']:

                predict_flag = False

                batch_x, batch_y, epoch = volumes.getTrainingBatch(params)
                if params['use_multitask_training']:
                    summary, _, _ = sess.run([merged, optimizer, shallow_optimizer],
                                             feed_dict={x: batch_x, y: batch_y,
                                                        drop_rate: params['dropout'],
                                                        training_flag: True})
                else:
                    summary, _ = sess.run([merged, optimizer],
                                          feed_dict={x: batch_x, y: batch_y,
                                                     drop_rate: params['dropout'],
                                                     training_flag: True})
                train_writer.add_summary(summary, iteration)

                if iteration % params['display_step'] == 0:
                    train_acc, train_loss, train_preds = sess.run([accuracy, loss, pred],
                                                                  feed_dict={x: batch_x, y: batch_y, drop_rate: 0.0, training_flag:False})
                    test_acc, test_loss, test_preds, test_summary = sess.run([accuracy, loss, pred, merged],
                                                                             feed_dict={x: test_x, y: test_y, drop_rate:0.0, training_flag:False})
                    train_prec = precision_score(np.argmax(batch_y, 1), np.argmax(train_preds, 1))
                    test_prec = precision_score(np.argmax(test_y, 1), np.argmax(test_preds, 1))
                    test_f1 = fbeta_score(np.argmax(test_y, 1), np.argmax(test_preds, 1), beta=params['fbeta_weight'])

                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    test_losses.append(test_loss)
                    train_losses.append(train_loss)
                    train_precs.append(train_prec)
                    test_precs.append(test_prec)
                    train_minutes.append([iteration, ((time.time() - start_time)/60 )])

                    test_writer.add_summary(test_summary, iteration)


                    print('Iteration: {}\t\tEpoch: {}'.format(iteration, epoch))
                    print('Train Loss: {:.3f}\tTrain Acc: {:.3f}\tInk Precision: {:.3f}'.format(train_loss, train_acc, train_precs[-1]))
                    print('Test Loss: {:.3f}\tTest Acc: {:.3f}\t\tInk Precision: {:.3f}'.format(test_loss, test_acc, test_precs[-1]))

                    if (test_f1 > best_test_f1):
                        print('\tAchieved new peak f1 score! Saving model...\n')
                        best_test_f1 = test_f1
                        best_f1_iteration = iteration
                        saver.save(sess, params['output_path'] + '/models/best-model.ckpt') 

                    if (test_acc > .9) and (test_prec > .7) and (iterations_since_prediction > 100): #or (test_prec > .8)  and (predictions_made < 4): # or (test_prec / params['numCubes'] < .05)
                        # make a full prediction if results are tentatively spectacular
                        predict_flag = True


                if (predict_flag) or (iteration % params['predict_step'] == 0 and iteration > 0):
                    np.savetxt(params['output_path']+'/times.csv', np.array(train_minutes), fmt='%.3f', delimiter=',', header='iteration,minutes')
                    prediction_start_time = time.time()
                    iterations_since_prediction = 0
                    predictions_made += 1
                    print('{} training iterations took {:.2f} minutes'.format( \
                        iteration, (time.time() - start_time)/60))
                    starting_coordinates = [0,0,0]
                    prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, starting_coordinates)

                    print('Beginning predictions on volumes...')
                    while next_coordinates is not None:
                        #TODO add back the output
                        prediction_values = sess.run(pred, feed_dict={x: prediction_samples, drop_rate: 0.0, training_flag:False})
                        volumes.reconstruct(params, prediction_values, coordinates)
                        prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, next_coordinates)
                    minutes = ( (time.time() - prediction_start_time) /60 )
                    volumes.saveAllPredictions(params, iteration)
                    volumes.saveAllPredictionMetrics(params, iteration, minutes)
                    saver.save(sess, params['output_path'] + '/models/model.ckpt', global_step=iteration)

                if params['wobble_volume'] and iteration >= params['wobble_step'] and (iteration % params['wobble_step']) == 0:
                    # ex. wobble at iteration 1000, or after the prediction for the previous wobble
                    volumes.wobbleVolumes(params)
                iteration += 1
                iterations_since_prediction += 1


        except KeyboardInterrupt:
            # still make last prediction if interrupted
            pass

        # make one last prediction after everything finishes
        # use the model that performed best on the test set
        saver.restore(sess, params['output_path'] + '/models/best-model.ckpt')
        starting_coordinates = [0,0,0]
        prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, starting_coordinates)
        print('Beginning predictions from best model (iteration {})...'.format(best_f1_iteration))
        while next_coordinates is not None:
            #TODO add back the output
            prediction_values = sess.run(pred, feed_dict={x: prediction_samples, drop_rate: 0.0, training_flag:False})
            volumes.reconstruct(params, prediction_values, coordinates)
            prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, next_coordinates)
        minutes = ( (time.time() - start_time) /60 )
        volumes.saveAllPredictions(params, best_f1_iteration)
        volumes.saveAllPredictionMetrics(params, best_f1_iteration, minutes)

    print('full script took {:.2f} minutes'.format((time.time() - start_time)/60))


if __name__ == '__main__':
    main()
