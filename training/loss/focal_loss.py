# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
