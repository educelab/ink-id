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
    params = inkid.ops.load_default_parameters()

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['ground_truth'] = args.groundtruth
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    params['grid_test_square'] = args.gridtestsquare
    # TODO folder name
    params['output_path'] = os.path.join(args.outputdir, '3dcnn-predictions/{}-{}-{}h'.format(
            datetime.datetime.today().timetuple()[1],
            datetime.datetime.today().timetuple()[2],
            datetime.datetime.today().timetuple()[3]))


    # create summary writer directory
    if tf.gfile.Exists(params['output_path']):
        tf.gfile.DeleteRecursively(params['output_path'])
    tf.gfile.MakeDirs(params['output_path'])

    with tf.Session() as sess:
        print('Beginning train session...')
        print('Output directory: {}'.format(params['output_path']))

        volumes = VolumeSet(params)

        # TODO give it the number of epochs or number of samples
        training_dataset = volumes.get_training_dataset(params['batch_size'])
        evaluation_dataset = volumes.get_evaluation_dataset(params['num_test_subvolumes'])
        prediction_dataset = volumes.get_prediction_dataset(params['prediction_batch_size'])

        training_iterator = training_dataset.make_one_shot_iterator()
        evaluation_iterator = evaluation_dataset.make_one_shot_iterator()
        prediction_iterator = prediction_dataset.make_one_shot_iterator()

        training_handle = sess.run(training_iterator.string_handle())
        evaluation_handle = sess.run(evaluation_iterator.string_handle())
        prediction_handle = sess.run(prediction_iterator.string_handle())
        
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
        next_input, next_label = iterator.get_next()

        drop_rate = tf.placeholder(tf.float32)
        training_flag = tf.placeholder(tf.bool)

        inputs, labels, pred, loss, accuracy, precision, fbeta, false_positive_rate = inkid.model.build_3dcnn(
            next_input, next_label, drop_rate, params, training_flag, params['fbeta_weight'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss)

        tf.summary.scalar('xentropy-loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('fbeta', fbeta)
        tf.summary.scalar('false_positive_rate', false_positive_rate)
        tf.summary.histogram('prediction_values', pred[:, 1])

        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=None)
    
        predict_flag = False
        iteration = 0
        iterations_since_prediction = 0

        best_test_fbeta = 0.0
        best_fbeta_iteration = 0

        train_writer = tf.summary.FileWriter(params['output_path'] + '/train', sess.graph)
        eval_writer = tf.summary.FileWriter(params['output_path'] + '/eval')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
            while True:
                predict_flag = False

                summary, _, train_accuracy, train_loss, train_preds, train_labels, train_precision = sess.run([merged, optimizer, accuracy, loss, pred, labels, precision],
                                                                                                              feed_dict={drop_rate: params['drop_rate'],
                                                                                                                         training_flag: True,
                                                                                                                         handle: training_handle})
                train_writer.add_summary(summary, iteration)
        
                if iteration % params['display_step'] == 0:
                    print("New display step")
                    eval_accuracy, eval_loss, eval_preds, eval_summary, eval_precision, eval_fbeta = sess.run([accuracy, loss, pred, merged, precision, fbeta],
                                                                             feed_dict={drop_rate: 0.0,
                                                                                        training_flag: False,
                                                                                        handle: evaluation_handle})
                    eval_writer.add_summary(eval_summary, iteration)

                    print('Iteration: {}'.format(iteration))
                    print('Train Loss: {:.3f}\tTrain Acc: {:.3f}\tTrain Precision: {:.3f}'.format(train_loss, train_accuracy, train_precision))
                    print('Eval Loss: {:.3f}\tEval Acc: {:.3f}\t\tEval Precision: {:.3f}'.format(eval_loss, eval_accuracy, eval_precision))
                    print('Eval FBeta Score:', eval_fbeta)

                    if (eval_fbeta > best_test_fbeta):
                        print('\tAchieved new peak f1 score! Saving model...\n')
                        best_test_fbeta = eval_fbeta
                        best_fbeta_iteration = iteration
                        saver.save(sess, params['output_path'] + '/models/best-model.ckpt')
                        # builder = tf.saved_model.builder.SavedModelBuilder(params['output_path'])
                        # builder.add_meta_graph_and_variables(sess, ['SERVING'])
                        # builder.save()

                    if (eval_accuracy > .9) and (eval_precision > .7) and (iterations_since_prediction > 100): #or (test_prec > .8)  and (predictions_made < 4): # or (test_prec / params['numCubes'] < .05)
                        # make a full prediction if results are tentatively spectacular
                        predict_flag = True

                if (predict_flag) or (iteration % params['predict_step'] == 0 and iteration > 0):
                    np.savetxt(params['output_path']+'/times.csv', np.array(train_minutes), fmt='%.3f', delimiter=',', header='iteration,minutes')
                    prediction_start_time = time.time()
                    iterations_since_prediction = 0
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
