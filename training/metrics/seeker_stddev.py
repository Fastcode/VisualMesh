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


class SeekerStdDev(tf.keras.metrics.Metric):
    def __init__(self, name, threshold, **kwargs):
        super(SeekerStdDev, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        self.n = self.add_weight(name="n", shape=(), initializer="zeros", dtype=tf.int64)
        self.s = self.add_weight(name="s", shape=(), initializer="zeros", dtype=tf.float64)

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

        y_true = tf.clip_by_value(y_true, -1.0, 1.0)

        # For standard deviation we only want to look at how we went when we were predicting that we were close
        idx = tf.squeeze(tf.where(tf.reduce_all(tf.abs(y_pred) <= self.threshold, axis=-1)), axis=-1)
        y_true = tf.cast(tf.gather(y_true, idx), dtype=tf.float64)
        y_pred = tf.cast(tf.gather(y_pred, idx), dtype=tf.float64)

        # Get the properties we care about from the sample we just received
        n = tf.shape(y_true, out_type=tf.int64)[0]
        s = tf.math.reduce_sum(tf.math.squared_difference(y_pred, y_true))

        # First time we just assign the b numbers
        self.s.assign_add(s)
        self.n.assign_add(n)

    def result(self):
        return tf.math.sqrt(self.s / tf.cast(self.n - 1, tf.float64))

    def reset_states(self):
        self.n.assign(tf.zeros_like(self.n))
        self.s.assign(tf.zeros_like(self.s))
