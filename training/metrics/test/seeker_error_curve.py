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
import math
from .curve import Curve
from .bucket import x_bucket


class SeekerErrorCurve(Curve):
    def __init__(self, scale, **kwargs):
        def x_axis(X, c):
            return X[:, 0]

        def y_axis(X, c):
            return tf.math.sqrt(tf.math.divide(X[:, 1], tf.cast(c[:, 0], dtype=X.dtype)))

        super(SeekerErrorCurve, self).__init__(
            x_axis=x_axis, y_axis=y_axis, sort_axis=x_axis, bucket_axis=x_bucket, **kwargs
        )

        self.scale = scale

    def update_state(self, y_true, y_pred):

        # Clip the truth values from -1 to 1
        y_true = tf.clip_by_value(y_true, -1.0, 1.0)
        y_pred = y_pred

        # Predicted distance to the point
        d_pred = tf.linalg.norm(y_pred, axis=-1)

        # Plot as predicted distance increases
        # For this we assume the mean is 0, meaning this is variance
        X = tf.stack([d_pred, tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=-1)], axis=-1)
        c = tf.expand_dims(tf.ones_like(d_pred, dtype=tf.int64), axis=-1)

        self.update(X, c)
