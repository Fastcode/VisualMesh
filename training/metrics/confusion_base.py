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


class ConfusionBase(tf.keras.metrics.Metric):
    def __init__(self, name, size, **kwargs):
        super(ConfusionBase, self).__init__(name=name, **kwargs)

        self.confusion = self.add_weight(name="confusion", shape=(size, size), initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Build up an index list that maps each class
        idx = tf.stack(
            [
                tf.argmax(input=y_true, axis=-1, output_type=self.confusion.dtype),
                tf.argmax(input=y_pred, axis=-1, output_type=self.confusion.dtype),
            ],
            axis=-1,
        )

        # Trim down the indexes to only those that have a class label
        idx = tf.gather(idx, tf.squeeze(tf.where(tf.reduce_any(tf.greater(y_true, 0.0), axis=-1)), axis=-1))

        # Add them into the corresponding locations
        self.confusion.scatter_nd_add(idx, tf.ones_like(idx[:, 0], dtype=self.confusion.dtype))

    def reset_states(self):
        self.confusion.assign(tf.zeros_like(self.confusion))
