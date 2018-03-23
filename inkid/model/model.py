"""
Functions for building the tf model.
"""
from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import layers

# https://stackoverflow.com/a/47043377
class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn, predict_every_n_steps):
        self._estimator = estimator
        self._input_fn = input_fn
        self._predict_every_n_steps = predict_every_n_steps

    def after_save(self, session, global_step):
        eval_results = self._estimator.evaluate(self._input_fn)
        print('Evaluation results:\n\t%s' % eval_results)

        iteration = global_step - 1
        if iteration > 0 and iteration % self._predict_every_n_steps == 0:
            pass # TODO predict
            

class Model3dcnn:
    def __init__(self, drop_rate, subvolume_shape, batch_norm_momentum, filters):
        self._drop_rate = drop_rate
        self._subvolume_shape = subvolume_shape
        self._filters = filters
        self._input_shape = [-1, subvolume_shape[0], subvolume_shape[1], subvolume_shape[2], 1]

        self.batch_norm1 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm2 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm3 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.batch_norm4 = layers.BatchNormalization(
            scale=False, axis=4, momentum=batch_norm_momentum)
        self.conv3d = partial(
            slim.convolution, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='valid')

    def __call__(self, inputs, training):
        y = tf.reshape(inputs, self._input_shape)
        y = self.batch_norm1(self.conv3d(y, num_outputs=self._filters[0]), training=training)
        y = self.batch_norm2(self.conv3d(y, num_outputs=self._filters[1]), training=training)
        y = self.batch_norm3(self.conv3d(y, num_outputs=self._filters[2]), training=training)
        y = self.batch_norm4(self.conv3d(y, num_outputs=self._filters[3]), training=training)
        y = layers.dropout(slim.fully_connected(slim.flatten(y), 2, activation_fn=None),
                           rate=self._drop_rate)
        return y


def model_fn_3dcnn(features, labels, mode, params):
    model = Model3dcnn(params['drop_rate'],
                       params['subvolume_shape'],
                       params['batch_norm_momentum'],
                       params['filters'])
    
    subvolume = features
    if isinstance(subvolume, dict):
        subvolume = features['Subvolume']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(subvolume, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
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
        logits = model(subvolume, training=True)
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
        fbeta_score = (1 + fbeta_squared) * tf.divide(
            (precision * recall),
            (fbeta_squared * precision) + recall + epsilon
        )

        # accuracy = tf.metrics.accuracy(
        #     labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))

        tf.identity(accuracy, name='train_accuracy')
        tf.identity(precision, name='train_precision')
        tf.identity(recall, name='train_recall')
        tf.identity(fbeta_score, name='train_fbeta_score')
        tf.summary.scalar('train_accuracy', accuracy)
        tf.summary.scalar('train_precision', precision)
        tf.summary.scalar('train_recall', recall)
        tf.summary.scalar('train_fbeta_score', fbeta_score)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(subvolume, training=False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

        # TODO deduplicate
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
        fbeta_score = (1 + fbeta_squared) * tf.divide(
            (precision * recall),
            (fbeta_squared * precision) + recall + epsilon
        )

        

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                # TODO add F1 and precision
                'accuracy': tf.metrics.accuracy(
                    labels=tf.argmax(labels, 1),
                    predictions=tf.argmax(logits, 1)
                ),
            })

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
