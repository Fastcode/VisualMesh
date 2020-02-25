#!/usr/bin/env python3

import tensorflow as tf


def FocalLoss(gamma=2.0):
    def focal_loss(y_true, y_pred, sample_weight=None):

        # Trim down the indexes to only those that have a class label
        idx = tf.squeeze(tf.where(tf.reduce_any(tf.greater(y_true, 0.0), axis=-1)), axis=-1)
        y_true = tf.gather(y_true, idx)
        y_pred = tf.gather(y_pred, idx)

        # Calculate the class weights required to balance the output
        C = tf.math.reduce_sum(y_true, axis=0, keepdims=True)
        C = tf.divide(tf.reduce_max(C), C)

        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        loss = tf.reduce_sum(tf.multiply(C, -tf.math.pow((1.0 - p_t), gamma) * tf.math.log(p_t)), axis=-1)

        return loss

    return focal_loss
