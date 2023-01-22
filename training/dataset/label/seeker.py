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
from training.op import difference_visual_mesh, map_visual_mesh, unmap_visual_mesh
from training.projection import project


class Seeker:
    def __init__(self, scale, ratio, mesh, geometry, **config):
        self.scale = scale

        # Grab our relevant fields
        self.mesh_model = mesh["model"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]
        self.min_ratio = ratio[0]
        self.max_ratio = ratio[1]

    def features(self):
        return {
            "seeker/targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }

    def __call__(self, image, Hoc, V, valid, **features):

        targets = features["seeker/targets"]
        projection = features["lens/projection"]
        focal_length = features["lens/focal_length"]
        centre = features["lens/centre"]
        k = features["lens/k"]
        dims = tf.shape(image)[:2]

        # If a target isn't on screen we will be expecting the network to predict on things it can't see
        # This will result in lower performance as it tries to learn how to do this for the training dataset
        px = tf.cast(tf.round(project(targets, dims, projection, focal_length, centre, k)), tf.int32)
        on_screen = tf.reduce_all(tf.logical_and(px >= 0, px < tf.expand_dims(dims, 0)), axis=-1)
        targets = tf.gather(targets, tf.squeeze(tf.where(on_screen), axis=-1))

        # Transform our remaining targets into observation plane space
        uOCo, l = tf.linalg.normalize(tf.einsum("ij,kj->ki", Hoc[:3, :3], targets), axis=-1)

        # Check if this signal is close enough in distance to our observation plane to be considered
        height = Hoc[2, 3]
        ratio = tf.squeeze(l, axis=-1) / height
        uOCo = tf.gather(
            uOCo, tf.squeeze(tf.where(tf.logical_and(ratio > self.min_ratio, ratio < self.max_ratio)), axis=-1)
        )

        # Remove any points that are above the camera in observation plane space as they can never be projected down
        uOCo = tf.gather(uOCo, tf.squeeze(tf.where(uOCo[:, 2] < 0), axis=-1))

        args = {"model": self.mesh_model, "height": height, "geometry": self.geometry, "radius": self.radius}

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

        # If we have no points after doing all this, add a random point
        # We can use it to ensure that the network learns how to predict when there are no points
        def random_distance():
            theta = tf.random.uniform([tf.shape(distance)[0], 1], -math.pi, math.pi)
            return math.sqrt(2.0) * tf.stack([tf.cos(theta), tf.sin(theta)], axis=-1)

        distance = tf.cond(tf.size(distance) > 0, lambda: distance, random_distance)

        return {"Y": distance}
