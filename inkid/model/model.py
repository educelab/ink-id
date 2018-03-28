"""
Functions for building the tf model.
"""
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers

# https://stackoverflow.com/a/47043377
class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, eval_input_fn, predict_input_fn, predict_every_n_steps, volume_set, args):
        self._estimator = estimator
        self._eval_input_fn = eval_input_fn
        self._predict_input_fn = predict_input_fn
        self._predict_every_n_steps = predict_every_n_steps
        self._volume_set = volume_set
        self._args = args

    def after_save(self, session, global_step):
        eval_results = self._estimator.evaluate(self._eval_input_fn)
        print('Evaluation results:\n\t%s' % eval_results)

        iteration = global_step - 1
        # if iteration > 0 and iteration % self._predict_every_n_steps == 0:
        if True:
            predictions = self._estimator.predict(self._predict_input_fn)
            for prediction in predictions:
                self._volume_set.reconstruct(self._args, np.array([prediction['probabilities']]), np.array([[prediction['XYZcoordinate'][0], prediction['XYZcoordinate'][1], 0]]))
            self._volume_set.saveAllPredictions(self._args, iteration)

class Model3dcnn:
    def __init__(self, drop_rate, subvolume_shape, batch_norm_momentum, filters):
        self._drop_rate = drop_rate
        self._subvolume_shape = subvolume_shape
        self._filters = filters
        self._input_shape = [-1, subvolume_shape[0], subvolume_shape[1], subvolume_shape[2], 1]
        self._batch_norm_momentum = batch_norm_momentum

        # self.batch_norm1 = layers.BatchNormalization(
        #     scale=False, axis=4, momentum=batch_norm_momentum)
        # self.batch_norm2 = layers.BatchNormalization(
        #     scale=False, axis=4, momentum=batch_norm_momentum)
        # self.batch_norm3 = layers.BatchNormalization(
        #     scale=False, axis=4, momentum=batch_norm_momentum)
        # self.batch_norm4 = layers.BatchNormalization(
        #     scale=False, axis=4, momentum=batch_norm_momentum)
        # self.conv3d = partial(
        #     slim.convolution, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='valid')

    def __call__(self, inputs, training):
        inputs = (tf.reshape(
            inputs,
            [-1,
             self._subvolume_shape[0],
             self._subvolume_shape[1],
             self._subvolume_shape[2],
             1
            ]
        ))
        conv1 = layers.batch_normalization(slim.convolution(inputs, self._filters[0], [3, 3, 3],
                                                            stride=[2, 2, 2], padding='valid'),
                                           training=training,
                                           scale=False,
                                           axis=4,
                                           momentum=self._batch_norm_momentum)
        conv2 = layers.batch_normalization(slim.convolution(conv1, self._filters[1], [3, 3, 3],
                                                            stride=[2, 2, 2], padding='valid'),
                                           training=training,
                                           scale=False,
                                           axis=4,
                                           momentum=self._batch_norm_momentum)
        conv3 = layers.batch_normalization(slim.convolution(conv2, self._filters[2], [3, 3, 3],
                                                            stride=[2, 2, 2], padding='valid'),
                                           training=training,
                                           scale=False,
                                           axis=4,
                                           momentum=self._batch_norm_momentum)
        conv4 = layers.batch_normalization(slim.convolution(conv3, self._filters[3], [3, 3, 3],
                                                            stride=[2, 2, 2], padding='valid'),
                                           training=training,
                                           scale=False,
                                           axis=4,
                                           momentum=self._batch_norm_momentum)

        logits = layers.dropout(slim.fully_connected(slim.flatten(conv4),
                                                  2,
                                                  activation_fn=None),
                                rate=self._drop_rate,
                                training=training)

        # tf.summary.histogram('dropout', tf.nn.softmax(net))

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=net))

        # return tf.nn.softmax(net), loss
        return logits
        # y = tf.reshape(inputs, self._input_shape)
        # y = self.batch_norm1(self.conv3d(y, num_outputs=self._filters[0]), training=training)
        # y = self.batch_norm2(self.conv3d(y, num_outputs=self._filters[1]), training=training)
        # y = self.batch_norm3(self.conv3d(y, num_outputs=self._filters[2]), training=training)
        # y = self.batch_norm4(self.conv3d(y, num_outputs=self._filters[3]), training=training)
        # y = layers.dropout(slim.fully_connected(slim.flatten(y), 2, activation_fn=None),
        #                    rate=self._drop_rate)
        # return y


def model_fn_3dcnn(features, labels, mode, params):
    model = Model3dcnn(
        params['drop_rate'],
        params['subvolume_shape'],
        params['batch_norm_momentum'],
        params['filters']
    )

    inputs = features['Subvolume']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(inputs, training=False)
        predictions = {
            'volumeID': features['VolumeID'],
            'XYZcoordinate': features['XYZCoordinate'],
            'class': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        logits = model(inputs, training=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

        epsilon = 1e-5
        predicted = tf.argmax(logits, 1)
        actual = tf.argmax(labels, 1)
        true_positives = tf.count_nonzero(predicted * actual, dtype=tf.float32)
        true_negatives = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
        false_positives = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
        false_negatives = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
        accuracy = tf.divide(
            true_positives + true_negatives,
            true_positives + true_negatives + false_positives + false_negatives
        )
        precision = tf.divide(
            true_positives,
            true_positives + false_positives + epsilon
        )
        recall = tf.divide(
            true_positives,
            true_positives + false_negatives + epsilon
        )
        # https://en.wikipedia.org/wiki/F1_score
        fbeta_weight = 0.3 # TODO use parameter
        fbeta_squared = tf.constant(fbeta_weight ** 2.0)
        fbeta = (1 + fbeta_squared) * tf.divide(
            (precision * recall),
            (fbeta_squared * precision) + recall + epsilon
        )

        # accuracy = tf.metrics.accuracy(
        #     labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))

        tf.identity(true_positives, name='train_true_positives')
        tf.identity(true_negatives, name='train_true_negatives')
        tf.identity(false_positives, name='train_false_positives')
        tf.identity(false_negatives, name='train_false_negatives')
        tf.identity(accuracy, name='train_accuracy')
        tf.identity(precision, name='train_precision')
        tf.identity(recall, name='train_recall')
        tf.identity(fbeta, name='train_fbeta_score')

        tf.summary.scalar('train_true_positives', true_positives)
        tf.summary.scalar('train_true_negatives', true_negatives)
        tf.summary.scalar('train_false_positives', false_positives)
        tf.summary.scalar('train_false_negatives', false_negatives)
        tf.summary.scalar('train_accuracy', accuracy)
        tf.summary.scalar('train_precision', precision)
        tf.summary.scalar('train_recall', recall)
        tf.summary.scalar('train_fbeta_score', fbeta)
        
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(inputs, training=False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
                'precision': tf.metrics.precision(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
                'recall': tf.metrics.recall(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
                'fbeta_score': fbeta_score(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1),
                    beta=params['fbeta_weight']
                ),
            })

# https://stackoverflow.com/a/45654762
def fbeta_score(labels, predictions, beta=0.5):
    precision, precision_update_op = tf.metrics.precision(labels, predictions)
    recall, recall_update_op = tf.metrics.recall(labels, predictions)
    epsilon = 1e-5
    score = (1 + beta**2) * tf.divide(
        (precision * recall),
        (beta**2 * precision) + recall + epsilon
    )
    return (score, tf.group(precision_update_op, recall_update_op))


def build_3dcnn(inputs, labels, drop_rate, args, training_flag, fbeta_weight):
    subvolumes = inputs['Subvolume']
    subvolumes = (tf.reshape(
        subvolumes,
        [-1, args["subvolume_dimension_x"], args["subvolume_dimension_y"], args["subvolume_dimension_z"], 1]))
    conv1 = layers.batch_normalization(slim.convolution(subvolumes, args["neurons"][0], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, args["neurons"][1], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, args["neurons"][2], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, args["neurons"][3], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])

    logits = layers.dropout(slim.fully_connected(slim.flatten(conv4),
                                              2,
                                              activation_fn=None),
                         rate=drop_rate)

    pred = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # https://stackoverflow.com/a/43960730
    predicted = tf.argmax(pred, 1)
    actual = tf.argmax(labels, 1)
    true_positives = tf.count_nonzero(predicted * actual, dtype=tf.float32)
    true_negatives = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
    false_positives = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
    false_negatives = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
    precision = tf.divide(true_positives, (true_positives + false_positives))
    recall = tf.divide(true_positives, (true_positives + false_negatives))
    # https://en.wikipedia.org/wiki/F1_score
    fbeta_squared = tf.constant(fbeta_weight ** 2.0)
    fbeta_score = (1 + fbeta_squared) * tf.divide(
        (precision * recall),
        (fbeta_squared * precision) + recall
    )

    return inputs, labels, pred, loss, accuracy, precision, fbeta_score, false_positives
    
        
def build_model(inputs, labels, drop_rate, args, training_flag):
    """Build a model. This is the original implementation."""
    inputs = (tf.reshape(
        inputs,
        [-1,
         args["subvolume_shape"][0],
         args["subvolume_shape"][1],
         args["subvolume_shape"][2],
         1
        ]
    ))
    conv1 = layers.batch_normalization(slim.convolution(inputs, args["filters"][0], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv2 = layers.batch_normalization(slim.convolution(conv1, args["filters"][1], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv3 = layers.batch_normalization(slim.convolution(conv2, args["filters"][2], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])
    conv4 = layers.batch_normalization(slim.convolution(conv3, args["filters"][3], [3, 3, 3],
                                                        stride=[2, 2, 2], padding='valid'),
                                       training=training_flag,
                                       scale=False,
                                       axis=4,
                                       momentum=args["batch_norm_momentum"])

    net = layers.dropout(slim.fully_connected(slim.flatten(conv4),
                                              2,
                                              activation_fn=None),
                         rate=drop_rate)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=net))

    return tf.nn.softmax(net), loss
