#!/usr/bin/env python3

import tensorflow as tf


def FocalLoss(gamma=2.0):
    def focal_loss(y_true, y_pred, sample_weight=None):

        # Trim down the indexes to only those that have a class label
        idx = tf.squeeze(tf.where(tf.reduce_any(tf.greater(y_true, 0.0), axis=-1)), axis=-1)
        y_true = tf.gather(y_true, idx)
        y_pred = tf.gather(y_pred, idx)

        # Balance the class weights
        C = tf.math.reduce_sum(y_true, axis=0, keepdims=True)  # Work out how many samples of each class
        # Divide each by the number of samples (makes each column sum to 1)
        W = tf.math.divide_no_nan(y_true, C)
        # Sum across to remove the 0s from non active classes
        W = tf.math.reduce_sum(W, axis=-1)
        # Normalise by number of classes to make total weight sum to 1
        W = tf.math.divide(W, tf.math.count_nonzero(C, dtype=y_true.dtype))
        # Make the average value be 1.0 for the batch
        W = tf.math.multiply(W, tf.cast(tf.size(idx), dtype=tf.float32))

        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        loss = tf.multiply(W, tf.reduce_sum(-tf.math.pow((1.0 - p_t), gamma) * tf.math.log(p_t), axis=-1))

        return loss

    return focal_loss
