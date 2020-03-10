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
    def __init__(self, input_files, classes, model, batch_size, prefetch, variants):
        self.input_files = input_files
        self.classes = classes
        self.batch_size = batch_size
        self.mesh_type = model["mesh"]["type"]
        self.cached_meshes = model["mesh"]["cached_meshes"]
        self.max_distance = model["mesh"]["max_distance"]
        self.geometry = tf.constant(model["geometry"]["shape"], dtype=tf.string, name="GeometryType")
        self.radius = model["geometry"]["radius"]
        self.n_intersections = model["geometry"]["intersections"]
        self.intersection_tolerance = model["geometry"]["intersection_tolerance"]
        self.prefetch = prefetch
        self._variants = variants

        # The number of neighbours is given in the name of the mesh type
        self.n_neighbours = int(re.search(r"\d+$", self.mesh_type).group()) + 1

    def _load_example(self, proto):
        example = tf.io.parse_single_example(
            serialized=proto,
            features={
                "image": tf.io.FixedLenFeature([], tf.string),
                "mask": tf.io.FixedLenFeature([], tf.string),
                "lens/projection": tf.io.FixedLenFeature([], tf.string),
                "lens/focal_length": tf.io.FixedLenFeature([], tf.float32),
                "lens/centre": tf.io.FixedLenFeature([2], tf.float32),
                "lens/k": tf.io.FixedLenFeature([2], tf.float32),
                "lens/fov": tf.io.FixedLenFeature([], tf.float32),
                "mesh/orientation": tf.io.FixedLenFeature([3, 3], tf.float32),
                "mesh/height": tf.io.FixedLenFeature([], tf.float32),
            },
        )

        return {
            "image": tf.image.decode_image(example["image"], channels=3),
            "mask": tf.image.decode_png(example["mask"], channels=4),
            "projection": example["lens/projection"],
            "focal_length": example["lens/focal_length"],
            "lens_centre": example["lens/centre"],
            "fov": example["lens/fov"],
            "lens_distortion": example["lens/k"],
            "orientation": example["mesh/orientation"],
            "height": example["mesh/height"],
            "raw": example["image"],
        }

    def _project_mesh(self, args):

        height = args["height"]
        orientation = args["orientation"]

        # Adjust our height and orientation
        if "mesh" in self._variants:
            v = self._variants["mesh"]
            if "height" in v:
                height = height + tf.random.truncated_normal(
                    shape=(), mean=v["height"]["mean"], stddev=v["height"]["stddev"],
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
                rot = [
                    cc * ca,
                    -cc * sa * cb + sc * sb,
                    cc * sa * sb + sc * cb,
                    sa,
                    ca * cb,
                    -ca * sb,
                    -sc * ca,
                    sc * sa * cb + cc * sb,
                    -sc * sa * sb + cc * cb,
                ]  # yapf: disable
                rot = tf.reshape(tf.stack(rot), [3, 3])

                # Apply the rotation
                orientation = tf.matmul(rot, orientation)

        # Construct the camera to observation plane transformation matrix
        Hoc = tf.zeros(shape=(4, 4))
        Hoc[0:3, 0:3] = orientation
        Hoc[2, 3] = height

        # Run the visual mesh to get our values
        pixels, neighbours = VisualMesh(
            tf.shape(input=args["image"])[:2],
            args["projection"],
            args["focal_length"],
            args["lens_centre"],
            args["lens_distortion"],
            args["fov"],
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

        # Bilinearly interpolate our image based on our floating point pixel coordinate
        y_0 = tf.floor(pixels[:, 0])
        x_0 = tf.floor(pixels[:, 1])
        y_1 = y_0 + 1
        x_1 = x_0 + 1

        # Weights for the x and y axis
        y_w = pixels[:, 0] - y_0
        x_w = pixels[:, 1] - x_0

        # Pixel coordinates to values to weighted values to X
        p_idx = [
            tf.cast(tf.stack([a, b], axis=-1), tf.int32) for a, b in [(y_0, x_0), (y_0, x_1), (y_1, x_0), (y_1, x_1),]
        ]
        p_val = [tf.image.convert_image_dtype(tf.gather_nd(args["image"], idx), tf.float32) for idx in p_idx]
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
        X = tf.add_n(p_weighted)

        # For the segmentation just use the nearest neighbour
        Y = tf.gather_nd(args["mask"], tf.cast(tf.round(pixels), tf.int32))

        return X, Y, neighbours, pixels

    def _expand_classes(self, Y):

        # Expand the classes from colours into individual columns
        W = tf.image.convert_image_dtype(Y[:, 3], tf.float32)  # Alpha channel
        cs = []
        for c in self.classes:
            cs.append(
                tf.where(
                    tf.logical_and(
                        tf.reduce_any(
                            tf.stack(
                                [tf.reduce_all(input_tensor=tf.equal(Y[:, :3], [v]), axis=-1) for v in c["colours"]],
                                axis=-1,
                            ),
                            axis=-1,
                        ),
                        tf.greater(W, 0.0),
                    ),
                    1.0,
                    0.0,
                )
            )
        Y = tf.stack(cs, axis=-1)

        return Y, W

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
        raws = []
        n_elems = tf.zeros(shape=(), dtype=tf.int32)

        for i in range(self.batch_size):

            # Load the example from the proto
            example = self._load_example(protos[i])

            # Project the visual mesh for this example
            X, Y, G, px = self._project_mesh(example)

            # We actually do know the shape of G but tensorflow makes it a little hard to do
            # We reshape here to ensure the size is known
            G = tf.reshape(G, (-1, self.n_neighbours))

            # Apply any visual augmentations we may want
            if "image" in self._variants:
                X = self._apply_variants(X)

            # Expand the classes for this value
            Y, W = self._expand_classes(Y)

            # Number of elements in this component
            n = tf.shape(input=Y)[0]

            # Cut off the null point and replace with -1s
            G = G[:-1] + n_elems
            G = tf.where(G == n + n_elems, -1, G)

            # Move along our number of elements
            n_elems = n_elems + n

            # Add all the elements to the lists
            Xs.append(X)
            Ys.append(Y)
            Ws.append(W)
            Gs.append(G)
            pxs.append(px)
            raws.append(example["raw"])
            ns.append(n)

        # Add on the null point for X and G
        # This is a hack 5 is number of neighbours + 1
        Xs.append(tf.constant([[-1.0, -1.0, -1.0]], dtype=tf.float32))
        Gs.append(tf.fill([1, self.n_neighbours], n_elems))

        X = tf.concat(Xs, axis=0)
        Y = tf.concat(Ys, axis=0)
        W = tf.concat(Ws, axis=0)
        G = tf.concat(Gs, axis=0)
        n = tf.stack(ns, axis=-1)
        px = tf.concat(pxs, axis=0)
        raw = tf.stack(raws, axis=0)

        # Fix the null point for G
        G = tf.where(G == -1, n_elems, G)

        # Return the results
        return {"X": X, "Y": Y, "W": W, "n": n, "G": G, "px": px, "raw": raw}

    def build(self, stats=False):

        # Make the statistics aggregator
        if stats:
            aggregator = tf.data.experimental.StatsAggregator()

        # Load our dataset records and batch them while they are still compressed
        dataset = tf.data.TFRecordDataset(self.input_files)
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
