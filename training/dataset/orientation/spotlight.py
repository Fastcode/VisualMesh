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

from .random_rotation import random_axis, random_rotation


class Spotlight:
    def __init__(self, **config):
        self.augmentations = {} if "augmentations" not in config else config["augmentations"]

    def features(self):
        f = {
            "Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
            "spotlight/targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }
        if "position" in self.augmentations and "generate" in self.augmentations["position"]:
            f["lens/fov"] = tf.io.FixedLenFeature([], tf.float32)
        return f

    def __call__(self, Hoc, **features):

        targets = features["spotlight/targets"]
        fov = features["lens/fov"]

        # If we are told to generate random positions then append a random position to the list of targets
        if "position" in self.augmentations:
            p = self.augmentations["position"]
            if "generate" in p and "limits" in p:

                # Random distance
                r = tf.random.uniform((), p["limits"][0], p["limits"][1])

                # The random theta value is generated based on the lens fov
                theta = tf.random.uniform((), 0, fov)
                psi = tf.random.uniform((), -math.pi, math.pi)

                # Add on this value
                targets = tf.concat(
                    [
                        targets,
                        tf.expand_dims(
                            tf.stack(
                                [
                                    r * tf.math.cos(theta),
                                    r * tf.math.cos(psi) * tf.math.sin(theta),
                                    r * tf.math.sin(psi) * tf.math.sin(theta),
                                ],
                                axis=0,
                            ),
                            axis=0,
                        ),
                    ],
                    axis=0,
                )

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
            if "limits" in v:
                h = tf.clip_by_value(
                    h, tf.constant(v["limits"][0], dtype=h.dtype), tf.constant(v["limits"][1], dtype=h.dtype)
                )

        y, _ = tf.linalg.normalize(tf.linalg.cross(z, world_z))
        x, _ = tf.linalg.normalize(tf.linalg.cross(y, z))

        # Assemble Hoc
        Roc = tf.stack([x, y, z], axis=0)
        spotlight_Hoc = tf.concat(
            [tf.pad(Roc, [[0, 1], [0, 0]]), tf.convert_to_tensor([[0], [0], h, [1]], Roc.dtype)], axis=1
        )

        return {"Hoc": spotlight_Hoc}
