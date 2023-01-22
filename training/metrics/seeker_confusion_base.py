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

from .confusion_base import ConfusionBase


class SeekerConfusionBase(ConfusionBase):
    def __init__(self, name, threshold, **kwargs):
        super(SeekerConfusionBase, self).__init__(name, size=2, **kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Y true contains a list of all the possible answers, we are ranked on how close we are to the closest one
        # This code works out the closest point to our prediction for each case so we can use that for training
        y_true = tf.gather_nd(
            y_true,
            tf.stack(
                [
                    tf.range(tf.shape(y_true)[0], dtype=tf.int32),
                    tf.math.argmin(
                        tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(y_pred, axis=-2), y_true), axis=-1),
                        axis=-1,
                        output_type=tf.int32,
                    ),
                ],
                axis=1,
            ),
        )

        # Our labels for this confusion matrix are based on if the points are near enough
        y_true = tf.where(tf.reduce_all(tf.math.abs(y_true) <= self.threshold, axis=-1), 1.0, 0.0)
        y_pred = tf.where(tf.reduce_all(tf.math.abs(y_pred) <= self.threshold, axis=-1), 1.0, 0.0)

        super(SeekerConfusionBase, self).update_state(
            tf.stack([y_true, 1.0 - y_true], axis=1),
            tf.stack([y_pred, 1.0 - y_pred], axis=1),
            sample_weight=sample_weight,
        )
