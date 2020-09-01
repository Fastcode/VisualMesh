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

from .random_rotation import random_axis, random_rotation


class Spotlight:
    def __init__(self, **config):
        self.augmentations = {} if "augmentations" not in config else config["augmentations"]

    def features(self):
        return {
            "Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
            "spotlight/targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }

    def __call__(self, Hoc, **features):

        targets = features["spotlight/targets"]

        # Extract the world z that we will use to orient our mesh
        world_z = Hoc[2, :3]

        # If we are going to apply a random rotation we only need to do it to world z
        if "rotation" in self.augmentations:
            v = self.augmentations["rotation"]

            # Apply a random axis angle rotation
            world_z = tf.squeeze(
                tf.matmul(random_rotation(v["mean"], v["stddev"])[:3, :3], tf.expand_dims(world_z, -1)), -1
            )

        # Pick a random target
        rOCc = targets[tf.random.uniform((), 0, tf.shape(targets)[0], tf.int32)]

        # Apply a random augmentation to the position
        if "position" in self.augmentations:
            v = self.augmentations["position"]
            rOCc = rOCc + random_axis() * tf.random.truncated_normal((), v["mean"], v["stddev"])

        # Get our axes and height
        z, h = tf.linalg.normalize(-rOCc)

        # If a minimum and/or maximum distance is given, apply them to the position
        if "position" in self.augmentations:
            v = self.augmentations["position"]
            if "min" in v:
                h = tf.math.maximum(tf.constant(v["min"], dtype=h.dtype), h)
            if "max" in v:
                h = tf.math.minimum(tf.constant(v["max"], dtype=h.dtype), h)

        y, _ = tf.linalg.normalize(tf.linalg.cross(z, world_z))
        x, _ = tf.linalg.normalize(tf.linalg.cross(y, z))

        # Assemble Hoc
        Roc = tf.stack([x, y, z], axis=0)
        spotlight_Hoc = tf.concat(
            [tf.pad(Roc, [[0, 1], [0, 0]]), tf.convert_to_tensor([[0], [0], h, [1]], Roc.dtype)], axis=1
        )

        return {"Hoc": spotlight_Hoc}
