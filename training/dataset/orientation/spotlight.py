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


class Spotlight:
    def __init__(self, **config):
        pass

    def features(self):
        return {
            "Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
            "targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }

    def __call__(self, Hoc, targets, **features):

        # Pick a random target
        rOCc = targets[tf.random.uniform((), 0, tf.shape(targets)[0], tf.int32)]

        # Get our axes and height
        z, h = tf.linalg.normalize(-rOCc)
        y, _ = tf.linalg.normalize(tf.linalg.cross(z, Hoc[2, :3]))
        x, _ = tf.linalg.normalize(tf.linalg.cross(y, z))

        # Assemble Hoc
        Roc = tf.stack([x, y, z], axis=0)
        spotlight_Hoc = tf.concat(
            [tf.pad(Roc, [[0, 1], [0, 0]]), tf.convert_to_tensor([[0], [0], h, [1]], Roc.dtype)], axis=1
        )

        return {"Hoc": spotlight_Hoc}
