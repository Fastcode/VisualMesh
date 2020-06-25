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

import math

import tensorflow as tf


def SeekerLoss():
    def seeker_loss(y_true, y_pred, sample_weight=None):

        # Y should be between -1 and 1
        y_true = tf.clip_by_value(y_true, -1.0, 1.0)

        # Two separate loss functions for when points are close or far
        # When the distance is far we don't care about the sign anymore
        close = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=-1)
        far = tf.reduce_mean(tf.math.squared_difference(tf.math.abs(y_true), tf.math.abs(y_pred)))

        # Select which loss function to use based on if the points are close
        loss = tf.where(tf.reduce_all(y_true <= math.tanh(1.0), axis=-1), close, far)

        return loss

    return seeker_loss
