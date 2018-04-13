"""
Functions for building the tf model.
"""
from functools import partial

import numpy as np
import tensorflow as tf

import inkid.ops
import inkid.metrics

class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    """Run some logic every time a checkpoint is saved.

    This is a bit of a Trojan horse that allows us to run evaluations,
    predictions, or arbitrary logic in the middle of a training
    run. An instance of this class is passed to the estimator when
    .train() is called. We also define in the RunConfig of the
    estimator how often we want it to save a checkpoint. So it will
    save a checkpoint that often, and then the after_save() method
    below is called. By passing the estimator itself to this class
    when it is initialized, we can call .evaluate() or .predict() on
    the estimator from here. Once done, the training process will
    continue unaware that any of this happened.

    https://stackoverflow.com/a/47043377

    """
    def __init__(self, estimator, eval_input_fn, predict_input_fn, predict_every_n_steps, region_set, predictions_dir):
        """Initialize the listener.

        Notably we pass the estimator itself to this class so that we
        can use it later.

        """
        self._estimator = estimator
        self._eval_input_fn = eval_input_fn
        self._predict_input_fn = predict_input_fn
        self._predict_every_n_steps = predict_every_n_steps
        self._region_set = region_set
        self._predictions_dir = predictions_dir
        
    def after_save(self, session, global_step):
        """Run our custom logic after the estimator saves a checkpoint."""
        eval_results = self._estimator.evaluate(self._eval_input_fn)

        iteration = global_step - 1
        predictions = self._estimator.predict(
            self._predict_input_fn,
            predict_keys=[
                'region_id',
                'ppm_xy',
                'probabilities',
            ],
        )
        for prediction in predictions:
            self._region_set.reconstruct_predicted_ink_classes(
                np.array([prediction['region_id']]),
                np.array([prediction['probabilities']]),
                np.array([prediction['ppm_xy']]),
            )
        self._region_set.save_predictions(self._predictions_dir, iteration)
        self._region_set.reset_predictions()


class Model3dcnn(object):
    """Defines the network architecture."""
    def __init__(self, drop_rate, subvolume_shape, batch_norm_momentum, filters):
        """Initialize the layers as members with state."""
        self._input_shape = [-1, subvolume_shape[0], subvolume_shape[1], subvolume_shape[2], 1]

        # To save some space below, this creates a tf.layers.Conv3D
        # that is still missing the 'filters' argument, so it can be
        # called multiple times below but we only need to specify
        # 'filters' since the other arguments are the same for each
        # convolutional layer.
        convolution_layer = partial(
            tf.layers.Conv3D,
            kernel_size=[3, 3, 3],
            strides=(2, 2, 2),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1, 1),
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
        )
        
        self.conv1 = convolution_layer(filters=filters[0])
        self.batch_norm1 = tf.layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)

        self.conv2 = convolution_layer(filters=filters[1])
        self.batch_norm2 = tf.layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)

        self.conv3 = convolution_layer(filters=filters[2])
        self.batch_norm3 = tf.layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)

        self.conv4 = convolution_layer(filters=filters[3])
        self.batch_norm4 = tf.layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)

        self.fc = tf.layers.Dense(2)
        self.dropout = tf.layers.Dropout(drop_rate)

        
    def __call__(self, inputs, training):
        """Chain the layers together when this class is 'called'."""
        y = tf.reshape(inputs, self._input_shape)
        y = self.conv1(y)
        y = self.batch_norm1(y, training=training)
        y = self.conv2(y)
        y = self.batch_norm2(y, training=training)
        y = self.conv3(y)
        y = self.batch_norm3(y, training=training)
        y = self.conv4(y)
        y = self.batch_norm4(y, training=training)
        y = tf.layers.flatten(y)
        y = self.fc(y)
        y = self.dropout(y, training=training)
        
        return y


def model_fn_3dcnn(features, labels, mode, params):
    """Define the model_fn for the Tensorflow Estimator.

    Depending on what mode is passed (train, evaluate, or predict)
    perform the necessary actions.

    The graph is built again every time .train(), .evaluate() or
    .predict() are called. In each case the estimator will first check
    the model directory to see if checkpoints have been saved, and
    then it will load the latest checkpoint weights into the
    graph. This is why it works for us to run an evaluation or
    prediction in the middle of training, because they are run right
    after checkpoints have been saved. This functionality is all built
    into the Tensorflow Estimator.

    https://github.com/tensorflow/tensorflow/issues/13895

    """
    model = Model3dcnn(
        params['drop_rate'],
        params['subvolume_shape'],
        params['batch_norm_momentum'],
        params['filters']
    )

    inputs = features['Input']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(inputs, training=False)
        # Here we specify all of the possible outputs from calling
        # .predict(), which returns a dictionary with these keys for
        # each prediction. So by passing predict_keys to .predict(),
        # we can select some of these and not return the others.
        predictions = {
            'region_id': features['RegionID'],
            'ppm_xy': features['PPM_XY'],
            'class': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
            'inputs': inputs,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
        )

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
        positives = true_positives + false_positives
        negatives = true_negatives + false_negatives
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
        fbeta_weight = 0.3
        fbeta_squared = tf.constant(fbeta_weight ** 2.0)
        fbeta = (1 + fbeta_squared) * tf.divide(
            (precision * recall),
            (fbeta_squared * precision) + recall + epsilon
        )

        tf.identity(true_positives, name='train_true_positives')
        tf.identity(true_negatives, name='train_true_negatives')
        tf.identity(false_positives, name='train_false_positives')
        tf.identity(false_negatives, name='train_false_negatives')
        tf.identity(positives, name='train_positives')
        tf.identity(negatives, name='train_negatives')
        tf.identity(accuracy, name='train_accuracy')
        tf.identity(precision, name='train_precision')
        tf.identity(recall, name='train_recall')
        tf.identity(fbeta, name='train_fbeta_score')

        tf.summary.scalar('train_true_positives', true_positives)
        tf.summary.scalar('train_true_negatives', true_negatives)
        tf.summary.scalar('train_false_positives', false_positives)
        tf.summary.scalar('train_false_negatives', false_negatives)
        tf.summary.scalar('train_positives', positives)
        tf.summary.scalar('train_negatives', negatives)
        tf.summary.scalar('train_accuracy', accuracy)
        tf.summary.scalar('train_precision', precision)
        tf.summary.scalar('train_recall', recall)
        tf.summary.scalar('train_fbeta_score', fbeta)

        # These three lines are very important despite being a little
        # opaque. Without them, batch normalization does not really
        # work at all, and the model will appear to train successfully
        # but this will not transfer to any evaluation or prediction
        # runs.
        # https://github.com/tensorflow/tensorflow/issues/16455
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op
        )

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
                'fbeta_score': inkid.metrics.fbeta_score(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1),
                    beta=params['fbeta_weight']
                ),
                'total_positives': inkid.metrics.total_positives(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
                'total_negatives': inkid.metrics.total_negatives(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
            })
