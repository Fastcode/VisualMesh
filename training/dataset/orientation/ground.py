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

from .random_rotation import random_rotation


class Ground:
    def __init__(self, **config):
        self.augmentations = {} if "augmentations" not in config else config["augmentations"]

    def features(self):
        return {
            "Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
        }

    def __call__(self, Hoc, **features):

        # If we have a height augmentation, apply it
        if "height" in self.augmentations:
            v = self.augmentations["height"]

            Hoc = tf.tensor_scatter_nd_add(
                Hoc,
                [[3, 2]],
                tf.expand_dims(tf.random.truncated_normal(shape=(), mean=v["mean"], stddev=v["stddev"]), 0),
            )

        if "rotation" in self.augmentations:
            v = self.augmentations["rotation"]

            # Apply a random axis angle rotation
            Hoc = tf.matmul(random_rotation(v["mean"], v["stddev"]), Hoc)

        # We can just read the cameras ground orientation directly from camera/Hoc
        return {"Hoc": Hoc}
