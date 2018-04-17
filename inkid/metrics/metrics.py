import tensorflow as tf


def total_positives(labels, predictions):
    true_positives, true_positives_update_op = tf.metrics.true_positives(labels, predictions)
    false_positives, false_positives_update_op = tf.metrics.false_positives(labels, predictions)
    return (
        true_positives + false_positives,
        tf.group(true_positives_update_op, false_positives_update_op)
    )


def total_negatives(labels, predictions):
    true_negatives, true_negatives_update_op = tf.metrics.true_negatives(labels, predictions)
    false_negatives, false_negatives_update_op = tf.metrics.false_negatives(labels, predictions)
    return (
        true_negatives + false_negatives,
        tf.group(true_negatives_update_op, false_negatives_update_op)
    )


# https://stackoverflow.com/a/45654762
def fbeta_score(labels, predictions, beta=0.3):
    precision, precision_update_op = tf.metrics.precision(labels, predictions)
    recall, recall_update_op = tf.metrics.recall(labels, predictions)
    epsilon = 1e-5
    score = (1 + beta**2) * tf.divide(
        (precision * recall),
        (beta**2 * precision) + recall + epsilon
    )
    return (score, tf.group(precision_update_op, recall_update_op))
