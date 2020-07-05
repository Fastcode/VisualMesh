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

        self.n_a = self.add_weight(name="n_a", shape=(), initializer="zeros", dtype=tf.int32)
        self.x_a = self.add_weight(name="x_a", shape=(), initializer="zeros", dtype=tf.float32)
        self.s_a = self.add_weight(name="s_a", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # For standard deviation we only want to look at how we went for the truth samples
        idx = tf.squeeze(tf.where(tf.reduce_all(y_true <= self.threshold, axis=-1)), axis=-1)
        y_true = tf.gather(y_true, idx)
        y_pred = tf.gather(y_pred, idx)

        # Get the difference value we will be getting stddev from
        # We're going to assume that both n and m are the same
        b = tf.reshape(y_pred - y_true, (-1,))

        # Get the properties we care about from the sample we just received
        x_b = tf.math.reduce_mean(b)
        n_b = tf.size(b)
        s_b = tf.math.reduce_sum(tf.square(b - x_b))

        # Combine this new data with the existing data
        n_ab = self.n_a + n_b
        x_ab = (self.x_a * tf.cast(self.n_a, self.dtype) + x_b * tf.cast(n_b, self.dtype)) / tf.cast(n_ab, self.dtype)
        s_ab = self.s_a + s_b + (x_b - self.x_a) ** 2 * tf.cast(self.n_a * n_b, self.dtype) / tf.cast(n_ab, self.dtype)

        # First time we just assign the b numbers
        self.x_a.assign(tf.where(self.n_a == 0, x_b, x_ab))
        self.s_a.assign(tf.where(self.n_a == 0, s_b, s_ab))
        self.n_a.assign(n_ab)

    def result(self):
        return tf.math.sqrt(self.s_a / tf.cast(self.n_a - 1, self.dtype))

    def reset_states(self):
        self.n_a.assign(tf.zeros_like(self.n_a))
        self.x_a.assign(tf.zeros_like(self.x_a))
        self.s_a.assign(tf.zeros_like(self.s_a))
