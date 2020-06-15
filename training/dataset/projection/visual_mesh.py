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

import re

import tensorflow as tf

import training.op as op


class VisualMesh:
    def __init__(self, mesh, geometry, **config):

        # Grab our relevant fields
        self.mesh_model = mesh["model"]
        self.cached_meshes = mesh["cached_meshes"]
        self.max_distance = mesh["max_distance"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]
        self.n_intersections = geometry["intersections"]
        self.intersection_tolerance = geometry["intersection_tolerance"]

        # The number of neighbours is given in the name of the mesh type
        self.n_neighbours = int(re.search(r"\d+$", self.mesh_model).group()) + 1

    def features(self):
        return {
            "lens/projection": tf.io.FixedLenFeature([], tf.string),
            "lens/focal_length": tf.io.FixedLenFeature([], tf.float32),
            "lens/centre": tf.io.FixedLenFeature([2], tf.float32),
            "lens/k": tf.io.FixedLenFeature([2], tf.float32),
            "lens/fov": tf.io.FixedLenFeature([], tf.float32),
        }

    def __call__(self, image, Hoc, **features):

        # Run the visual mesh to get our values
        C, V, G, I = op.project_visual_mesh(
            image_dimensions=tf.shape(image)[:2],
            lens_projection=features["lens/projection"],
            lens_focal_length=features["lens/focal_length"],
            lens_centre=features["lens/centre"],
            lens_distortion=features["lens/k"],
            lens_fov=features["lens/fov"],
            cam_to_observation_plane=Hoc,
            mesh_model=self.mesh_model,
            cached_meshes=self.cached_meshes,
            max_distance=self.max_distance,
            geometry=self.geometry,
            radius=self.radius,
            n_intersections=self.n_intersections,
            intersection_tolerance=self.intersection_tolerance,
        )

        # We actually do know the shape of G but tensorflow makes it a little hard to do in the c++ op
        # We reshape here to ensure the size is known for later shape inferences
        G = tf.reshape(G, (-1, self.n_neighbours))

        # We also cut off the null point on the end and replace it with the lowest int for easier combining later
        G = G[:-1]
        G = tf.where(G == tf.shape(G)[0], G.dtype.min, G)

        return {"C": C, "V": V, "G": G, "I": I}
