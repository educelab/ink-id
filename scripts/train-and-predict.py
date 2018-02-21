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

from inkid.volumes import VolumeSet
import inkid.model
import inkid.ops


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
    parser.add_argument('--gridtestsquare', metavar='num', default=0, type=int,
                        help='index of grid test square for this k-fold run')
    parser.add_argument('--outputdir', metavar='path', default='out',
                        help='path to output directory')

    args = parser.parse_args()

    # Load default parameters
    params = inkid.ops.load_parameters_from_json('default_parameters.json')

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['ground_truth'] = args.groundtruth
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    params['grid_test_square'] = args.gridtestsquare
    params['output_path'] = os.path.join(args.outputdir, '3dcnn-predictions/{}-{}-{}h'.format(
            datetime.datetime.today().timetuple()[1],
            datetime.datetime.today().timetuple()[2],
            datetime.datetime.today().timetuple()[3]))

    x = tf.placeholder(tf.float32, [None, params['x_dimension'], params['y_dimension'], params['z_dimension']])
    y = tf.placeholder(tf.float32, [None, params['n_classes']])
    drop_rate = tf.placeholder(tf.float32)
    training_flag = tf.placeholder(tf.bool)

    if params['use_multitask_training']:
        pred, shallow_loss, loss = inkid.model.build_multitask_model(x, y, drop_rate, params, training_flag)
        shallow_optimizer = tf.train.AdamOptimizer(learning_rate=params['shallow_learning_rate']).minimize(shallow_loss)
    else:
        pred, loss = inkid.model.build_model(x, y, drop_rate, params, training_flag)

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
            # while iteration < params['training_iterations']:
            while epoch < params['training_epochs']:

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
                                                                  feed_dict={x: batch_x, y: batch_y, drop_rate: 0.0, training_flag: False})
                    test_acc, test_loss, test_preds, test_summary = sess.run([accuracy, loss, pred, merged],
                                                                             feed_dict={x: test_x, y: test_y, drop_rate: 0.0, training_flag: False})
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
                        builder = tf.saved_model.builder.SavedModelBuilder(params['output_path'])
                        builder.add_meta_graph_and_variables(sess, ['SERVING'])

                    if (test_acc > .9) and (test_prec > .7) and (iterations_since_prediction > 100): #or (test_prec > .8)  and (predictions_made < 4): # or (test_prec / params['numCubes'] < .05)
                        # make a full prediction if results are tentatively spectacular
                        predict_flag = True

                if (predict_flag) or (iteration % params['predict_step'] == 0 and iteration > 0):
                    np.savetxt(params['output_path']+'/times.csv', np.array(train_minutes), fmt='%.3f', delimiter=',', header='iteration,minutes')
                    prediction_start_time = time.time()
                    iterations_since_prediction = 0
                    predictions_made += 1
                    print('{} training iterations took {:.2f} minutes'.format(
                        iteration, (time.time() - start_time)/60))
                    starting_coordinates = [0, 0, 0]
                    prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, starting_coordinates)

                    print('Beginning predictions on volumes...')
                    while next_coordinates is not None:
                        # TODO add back the output
                        prediction_values = sess.run(pred, feed_dict={x: prediction_samples, drop_rate: 0.0, training_flag: False})
                        volumes.reconstruct(params, prediction_values, coordinates)
                        prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, next_coordinates)
                    minutes = ((time.time() - prediction_start_time) / 60)
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
        starting_coordinates = [0, 0, 0]
        prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, starting_coordinates)
        print('Beginning predictions from best model (iteration {})...'.format(best_f1_iteration))
        while next_coordinates is not None:
            # TODO add back the output
            prediction_values = sess.run(pred, feed_dict={x: prediction_samples, drop_rate: 0.0, training_flag: False})
            volumes.reconstruct(params, prediction_values, coordinates)
            prediction_samples, coordinates, next_coordinates = volumes.getPredictionBatch(params, next_coordinates)
        minutes = ((time.time() - start_time) / 60)
        volumes.saveAllPredictions(params, best_f1_iteration)
        volumes.saveAllPredictionMetrics(params, best_f1_iteration, minutes)

    print('full script took {:.2f} minutes'.format((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
