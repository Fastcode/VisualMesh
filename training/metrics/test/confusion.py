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

import os

import tensorflow as tf


class Confusion(tf.keras.metrics.Metric):
    def __init__(self, name, classes, **kwargs):
        super(Confusion, self).__init__(name=name, **kwargs)

        self.classes = classes

        self.confusion = self.add_weight(
            name="confusion", shape=(len(classes), len(classes)), initializer="zeros", dtype=tf.int32
        )

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

    def result(self):
        return 0

    def _write(self, f, txt):
        print("{}".format(txt))
        f.write("{}\n".format(txt))

    def save(self, output_path):

        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        with open("{}.txt".format(base_path), "w") as f:

            for i, c in enumerate(self.classes):

                # For all labels where idx was predicted (all positives)
                p = tf.reduce_sum(self.confusion[:, i])
                # For all predictions where idx was labelled
                tp_fn = tf.reduce_sum(self.confusion[i, :])

                # Save the metrics
                name = c["name"]
                self._write(f, "{}".format(name.title()))
                self._write(f, "\tPrecision: {}".format(self.confusion[i, i] / p))
                self._write(f, "\tRecall: {}".format(self.confusion[i, i] / tp_fn))

                self._write(f, "\tPredicted {} samples are really:".format(name.title()))
                for j, k in enumerate(self.classes):
                    self._write(f, "\t\t{}: {:.3f}%".format(k["name"].title(), 100 * (self.confusion[j, i] / p)))
                self._write(f, "\tReal {} samples are predicted as:".format(name.title()))
                for j, k in enumerate(self.classes):
                    self._write(f, "\t\t{}: {:.3f}%".format(k["name"].title(), 100 * (self.confusion[i, j] / tp_fn)))
