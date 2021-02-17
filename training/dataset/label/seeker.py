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
from training.op import difference_visual_mesh, map_visual_mesh, unmap_visual_mesh
from training.projection import project


class Seeker:
    def __init__(self, scale, mesh, geometry, **config):
        self.scale = scale

        # Grab our relevant fields
        self.mesh_model = mesh["model"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]

    def features(self):
        return {
            "seeker/targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }

    def __call__(self, image, Hoc, V, valid, **features):

        targets = features["seeker/targets"]

        # Transform our targets into observation plane space
        uOCo, _ = tf.linalg.normalize(tf.einsum("ij,kj->ki", Hoc[:3, :3], targets), axis=-1)

        # Remove any points that are above the camera in observation plane space as they can never be projected down
        uOCo = tf.gather(uOCo, tf.squeeze(tf.where(uOCo[:, 2] < 0), axis=-1))

        args = {"model": self.mesh_model, "height": Hoc[2, 3], "geometry": self.geometry, "radius": self.radius}

        # Get our vectors in nm coordinates
        mesh_nm = unmap_visual_mesh(V, **args)
        target_nm = unmap_visual_mesh(uOCo, **args)

        # Replicate out the points so they are the same size
        n_nodes = tf.shape(mesh_nm)[0]
        n_targets = tf.shape(target_nm)[0]
        m = tf.reshape(tf.tile(mesh_nm, (1, n_targets)), (-1, 2))  # [a,b,c] -> [a,a,a,b,b,b,c,c,c]
        t = tf.tile(target_nm, (n_nodes, 1))  # [a,b,c] -> [a,b,c,a,b,c,a,b,c]

        # Calculate the difference that we will be predicting for each of the values
        diff = tf.reshape(difference_visual_mesh(t, m, **args), (n_nodes, n_targets, 2))

        # Divide distance by scale so a value from 0->scale becomes 0->1
        distance = diff / self.scale

        # Work out if any of the points are on screen so we can throw out samples which we can't see
        px = tf.cast(
            tf.round(
                project(
                    targets,
                    tf.shape(image)[:2],
                    features["lens/projection"],
                    features["lens/focal_length"],
                    features["lens/centre"],
                    features["lens/k"],
                )
            ),
            tf.int32,
        )
        on_screen = tf.reduce_any(
            tf.logical_and(
                tf.logical_and(px[:, 0] >= 0, px[:, 0] < tf.shape(image)[0]),
                tf.logical_and(px[:, 1] >= 0, px[:, 1] < tf.shape(image)[1]),
            )
        )

        return {"Y": distance, "valid": tf.logical_and(tf.logical_and(valid, on_screen), tf.shape(uOCo)[0] > 0)}
