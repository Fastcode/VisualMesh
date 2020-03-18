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
import os
import re

import tensorflow as tf

# If we are in docker look for the visual mesh op in /visualmesh/training/visualmesh_op.so
if (
    os.path.exists("/.dockerenv")
    or os.path.isfile("/proc/self/cgroup")
    and any("docker" in line for line in open("/proc/self/cgroup"))
) and os.path.isfile("/visualmesh/training/dataset/visualmesh_op.so"):
    VisualMesh = tf.load_op_library("/visualmesh/training/dataset/visualmesh_op.so").visual_mesh
# Otherwise check to see if we built it and it should be right next to this file
elif os.path.isfile(os.path.join(os.path.dirname(__file__), "visualmesh_op.so")):
    VisualMesh = tf.load_op_library(os.path.join(os.path.dirname(__file__), "visualmesh_op.so")).visual_mesh
else:
    raise Exception("Please build the tensorflow visual mesh op before running")


class VisualMeshDataset:
    def __init__(self, paths, mesh, geometry, batch_size, prefetch, variants={}, features={}):
        self.paths = paths
        self.batch_size = batch_size
        self.mesh_type = mesh["type"]
        self.cached_meshes = mesh["cached_meshes"]
        self.max_distance = mesh["max_distance"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]
        self.n_intersections = geometry["intersections"]
        self.intersection_tolerance = geometry["intersection_tolerance"]
        self.prefetch = prefetch
        self._variants = variants

        # We have the default features that are required for Visual Mesh projection
        # and then the extras that are for the specific dataset type
        self.features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "camera/projection": tf.io.FixedLenFeature([], tf.string),
            "camera/focal_length": tf.io.FixedLenFeature([], tf.float32),
            "camera/centre": tf.io.FixedLenFeature([2], tf.float32),
            "camera/k": tf.io.FixedLenFeature([2], tf.float32),
            "camera/fov": tf.io.FixedLenFeature([], tf.float32),
            "camera/Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
            **features,
        }

        # The number of neighbours is given in the name of the mesh type
        self.n_neighbours = int(re.search(r"\d+$", self.mesh_type).group()) + 1

    def _load_example(self, proto):
        return tf.io.parse_single_example(serialized=proto, features=self.features)

    def _project_mesh(self, args):

        # Grab Hoc so we can manipulate it
        Hoc = args["camera/Hoc"]

        # Adjust our height and orientation
        if "mesh" in self._variants:
            v = self._variants["mesh"]

            # If we have a height variation, apply it
            if "height" in v:
                tf.tensor_scatter_nd_add(
                    Hoc,
                    [[3, 2]],
                    tf.expand_dims(
                        tf.random.truncated_normal(shape=(), mean=v["height"]["mean"], stddev=v["height"]["stddev"]), 0
                    ),
                )
            if "rotation" in v:
                # Make 3 random euler angles
                rotation = tf.random.truncated_normal(
                    shape=[3], mean=v["rotation"]["mean"], stddev=v["rotation"]["stddev"],
                )
                # Cos and sin for everyone!
                ca = tf.cos(rotation[0])
                sa = tf.sin(rotation[0])
                cb = tf.cos(rotation[1])
                sb = tf.sin(rotation[1])
                cc = tf.cos(rotation[2])
                sc = tf.sin(rotation[2])

                # Convert these into a rotation matrix
                rot = tf.convert_to_tensor(
                    [
                        [cc * ca, -cc * sa * cb + sc * sb, cc * sa * sb + sc * cb, 0.0],
                        [sa, ca * cb, -ca * sb, 0.0],
                        [-sc * ca, sc * sa * cb + cc * sb, -sc * sa * sb + cc * cb, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=Hoc.dtype,
                )

                # Apply the rotation
                Hoc = tf.matmul(rot, Hoc)

        # Run the visual mesh to get our values
        px, G, global_indices = VisualMesh(
            tf.shape(args["X"])[:2],
            args["camera/projection"],
            args["camera/focal_length"],
            args["camera/centre"],
            args["camera/k"],
            args["camera/fov"],
            Hoc,
            self.mesh_type,
            self.cached_meshes,
            self.max_distance,
            self.geometry,
            self.radius,
            self.n_intersections,
            self.intersection_tolerance,
            name="ProjectVisualMesh",
        )

        # We actually do know the shape of G but tensorflow makes it a little hard to do
        # We reshape here to ensure the size is known
        G = tf.reshape(G, (-1, self.n_neighbours))

        return {"px": px, "G": G, "global_indices": global_indices}

    def _interpolate_gather(self, img, px):

        # Bilinearly interpolate our image based on our floating point pixel coordinate
        y_0 = tf.floor(px[:, 0])
        x_0 = tf.floor(px[:, 1])
        y_1 = y_0 + 1
        x_1 = x_0 + 1

        # Calculate the weights of how much the x and y account for for each of the 4 corners
        y_w = px[:, 0] - y_0
        x_w = px[:, 1] - x_0

        # Get the coordinates of the four closest pixels to this point
        p_idx = [
            tf.cast(tf.stack([a, b], axis=-1), tf.int32) for a, b in [(y_0, x_0), (y_0, x_1), (y_1, x_0), (y_1, x_1),]
        ]

        # Gather the pixel values from the image
        p_val = [tf.gather_nd(img, idx) for idx in p_idx]

        # Weight each of the pixel values based on their relative distance
        p_weighted = [
            tf.multiply(val, tf.expand_dims(w, axis=-1))
            for val, w in zip(
                p_val,
                [
                    tf.multiply(1 - y_w, 1 - x_w),
                    tf.multiply(1 - y_w, x_w),
                    tf.multiply(y_w, 1 - x_w),
                    tf.multiply(y_w, x_w),
                ],
            )
        ]

        # Add all the weighted values to get the final interpolated value
        return tf.add_n(p_weighted)

    def _apply_variants(self, X):
        # Make the shape of X back into an imageish shape for the functions
        X = tf.expand_dims(X, axis=0)

        # Apply the variants that were listed
        v = self._variants["image"]
        if "brightness" in v and v["brightness"]["stddev"] > 0:
            X = tf.image.adjust_brightness(
                X, tf.random.truncated_normal(shape=(), mean=v["brightness"]["mean"], stddev=v["brightness"]["stddev"])
            )
        if "contrast" in v and v["contrast"]["stddev"] > 0:
            X = tf.image.adjust_contrast(
                X, tf.random.truncated_normal(shape=(), mean=v["contrast"]["mean"], stddev=v["contrast"]["stddev"])
            )
        if "hue" in v and v["hue"]["stddev"] > 0:
            X = tf.image.adjust_hue(
                X, tf.random.truncated_normal(shape=(), mean=v["hue"]["mean"], stddev=v["hue"]["stddev"])
            )
        if "saturation" in v and v["saturation"]["stddev"] > 0:
            X = tf.image.adjust_saturation(
                X, tf.random.truncated_normal(shape=(), mean=v["saturation"]["mean"], stddev=v["saturation"]["stddev"])
            )
        if "gamma" in v and v["gamma"]["stddev"] > 0:
            X = tf.image.adjust_gamma(
                X, tf.random.truncated_normal(shape=(), mean=v["gamma"]["mean"], stddev=v["gamma"]["stddev"])
            )

        # Remove the extra dimension we added
        return tf.squeeze(X, axis=0)

    def _reduce_batch(self, protos):

        Xs = []
        Ys = []
        Ws = []
        Gs = []
        pxs = []
        ns = []
        images = []
        n_elems = tf.zeros(shape=(), dtype=tf.int32)

        for i in range(self.batch_size):

            # Load the example from the proto
            data = self._load_example(protos[i])

            # Load the image so we can get it's size
            data["X"] = tf.image.convert_image_dtype(tf.image.decode_image(data["image"], channels=3), tf.float32)

            # Project the visual mesh for this example
            data.update(self._project_mesh(data))

            # Select and interpolate the pixels to get X
            data["X"] = self._interpolate_gather(data["X"], data["px"])

            # Apply any visual augmentations we may want to the image
            if "image" in self._variants:
                data["X"] = self._apply_variants(data["X"])

            # Expand the classes for this value
            Y, W = self.get_labels(data)

            # Number of elements in this component
            n = tf.shape(data["G"])[0] - 1

            # Cut off the null point and replace with -1s
            G = data["G"]
            G = tf.where(G[:-1] == n, -1, G[:-1]) + n_elems

            # Move along our number of elements to offset the later batch graphs to be internally consistent
            n_elems = n_elems + n

            # Add all the elements to the lists
            Xs.append(data["X"])
            Ys.append(Y)
            Ws.append(W)
            Gs.append(G)
            pxs.append(data["px"])
            ns.append(n)
            images.append(data["image"])

        # Add on the null point for X and G
        Xs.append(tf.constant([[-1.0, -1.0, -1.0]], dtype=tf.float32))
        Gs.append(tf.fill([1, self.n_neighbours], n_elems))

        X = tf.concat(Xs, axis=0)
        Y = tf.concat(Ys, axis=0)
        W = tf.concat(Ws, axis=0)
        G = tf.concat(Gs, axis=0)
        n = tf.stack(ns, axis=-1)
        px = tf.concat(pxs, axis=0)
        image = tf.stack(images, axis=0)

        # Fix the null point for G
        G = tf.where(G == -1, n_elems, G)

        # Return the results
        return {"X": X, "G": G, "Y": Y, "W": W, "n": n, "px": px, "image": image}

    def build(self, stats=False):

        # Make the statistics aggregator
        if stats:
            aggregator = tf.data.experimental.StatsAggregator()

        # Load our dataset records and batch them while they are still compressed
        dataset = tf.data.TFRecordDataset(self.paths)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        # Apply our reduction function to project/squash our dataset into a batch
        dataset = dataset.map(self._reduce_batch, num_parallel_calls=self.prefetch)

        # Prefetch some elements to ensure training smoothness
        dataset = dataset.prefetch(self.prefetch)

        # Add the statistics
        if stats:
            options = tf.data.Options()
            options.experimental_stats.aggregator = aggregator
            dataset = dataset.with_options(options)

        return (dataset, aggregator.get_summary()) if stats else dataset


# Convert a dataset into a format that will be accepted by keras fit
def keras_dataset(args):
    # Return in the format (x, y, weights)
    return ((args["X"], args["G"]), args["Y"])
