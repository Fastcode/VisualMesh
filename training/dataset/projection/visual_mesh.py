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
from training.op import lookup_visual_mesh
from training.projection import project


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

    def __call__(self, image, Hoc, valid, **features):

        # Lookup vectors in the visual mesh
        V, G = lookup_visual_mesh(
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

        # Project them to pixel coordinates
        C = project(
            tf.einsum("ij,ki->kj", Hoc[:3, :3], V),
            tf.shape(image)[:2],
            features["lens/projection"],
            features["lens/focal_length"],
            features["lens/centre"],
            features["lens/k"],
        )

        # We actually do know the shape of G but tensorflow makes it a little hard to do in the c++ op
        # We reshape here to ensure the size is known for later shape inferences
        G = tf.reshape(G, (-1, self.n_neighbours))

        return {
            "C": C,
            "V": V,
            "G": G,
            "lens/projection": features["lens/projection"],
            "lens/focal_length": features["lens/focal_length"],
            "lens/centre": features["lens/centre"],
            "lens/k": features["lens/k"],
            "lens/fov": features["lens/fov"],
            "valid": tf.logical_and(valid, tf.size(C) > 0),
        }
