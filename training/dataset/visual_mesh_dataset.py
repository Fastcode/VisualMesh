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
import math


class VisualMeshDataset:
    def __init__(self, paths, batch_size, view, example, orientation, projection, label, keys):
        self.paths = paths
        self.view = view
        self.example = example
        self.orientation = orientation
        self.projection = projection
        self.label = label
        self.keys = keys
        self.batch_size = batch_size

        # Autotune how many elements we prefetch and map by
        self.prefetch = tf.data.experimental.AUTOTUNE

        # First we need to know what features we need for each stage so we can load them from the dataset
        self.features = {}
        for p in self.view.prefixes():

            # Get the features needed
            for k, t in [
                *self.orientation.features().items(),
                *self.projection.features().items(),
                *self.example.features().items(),
                *self.label.features().items(),
            ]:
                key = "{}{}".format(p, k)
                if key not in self.features or self.features[key] == t:
                    self.features[key] = t
                else:
                    raise RuntimeError("Incompatible features for {} ({} vs {})".format(key, self.features[key], t))

        # Now we build the in order to go from the keys we want to the ones that are in the dataset
        self.dataset_features = {}
        for k, t in self.features.items():
            key = k if k not in self.keys else self.keys[k]
            if key not in self.dataset_features or self.dataset_features[key] == t:
                self.dataset_features[key] = t
            else:
                raise RuntimeError("Incompatible features for {} ({} vs {})".format(key, self.features[key], t))

    def _map(self, proto):

        # Load the data and convert it to our internal format
        features = tf.io.parse_single_example(serialized=proto, features=self.dataset_features)
        features = {k: features[k if k not in self.keys else self.keys[k]] for k in self.features}

        views = {}
        for p in self.view.prefixes():

            result = {"valid": True}

            # Use orientation to work out where the mesh should be projected
            result.update(self.orientation(**{**{k: features[p + k] for k in self.orientation.features()}, **result}))

            # Read the input image as we need the dimensions
            result.update(self.example.input(**{**{k: features[p + k] for k in self.example.features()}, **result}))

            # Perform the actual projection to get vectors and coordinates
            result.update(self.projection(**{**{k: features[p + k] for k in self.projection.features()}, **result}))

            # Get the image output using the requested features and the projection information
            result.update(self.example(**{**{k: features[p + k] for k in self.example.features()}, **result}))

            # Get the label output using the requested features and the projection information
            result.update(self.label(**{**{k: features[p + k] for k in self.label.features()}, **result}))

            # Put it in the prefix results
            views[p] = result

        # Apply the multiview merging algorithm to the results
        return self.view.merge(views)

    def _reduce(self, batch):

        # Get n out from the ragged batch
        n = batch["n"]

        # We need the n offsets to reposition all the graph values
        cn = tf.math.cumsum(tf.math.reduce_sum(n, axis=1), exclusive=True)
        n_elems = tf.math.reduce_sum(n)

        # For X we must add the "offscreen" point as a -1,-1,-1 point one past the end of the list
        X = tf.concat(
            [
                tf.reshape(batch["X"].values, shape=[-1, 3], name="vmdataset/_reduce/reshape/X"),
                tf.constant([[-1.0, -1.0, -1.0]], dtype=tf.float32),
            ],
            axis=0,
            name="vmdataset/_reduce/concat/X",
        )

        # Using .values removes the outer "ragged" batch which is effectively concatenating
        Y = batch["Y"].values
        C = batch["C"].values
        V = batch["V"].values

        # If Y has multiple values per label we need to expand them out so the loss function doesn't cry
        if isinstance(Y, tf.RaggedTensor):
            Y = Y.to_tensor(default_value=math.nan)

        # Fixed size elements are in sensible shapes
        jpg = batch["jpg"]
        Hoc = batch["Hoc"]
        lens = {
            "projection": batch["lens/projection"],
            "focal_length": batch["lens/focal_length"],
            "centre": batch["lens/centre"],
            "k": batch["lens/k"],
            "fov": batch["lens/fov"],
        }

        # Add on our offset for each batch so that we get a proper result and then remove the ragged edge
        G = (batch["G"] + tf.expand_dims(tf.expand_dims(cn, -1), -1)).values

        # Replace the offscreen negative points with the offscreen point and append it to the end
        G = tf.concat(
            [tf.where(G < 0, n_elems, G, name="vmdataset/_reduce/where/G"), tf.fill([1, tf.shape(G)[-1]], n_elems),],
            axis=0,
            name="vmdataset/_reduce/concat/G",
        )

        return {
            "X": X,
            "Y": Y,
            "G": G,
            "n": n,
            "C": C,
            "V": V,
            "Hoc": Hoc,
            "jpg": jpg,
            "lens": lens,
        }

    def build(self):

        # Load the files
        dataset = tf.data.TFRecordDataset(self.paths)

        # Extract the data from the examples
        dataset = dataset.map(self._map, num_parallel_calls=self.prefetch)

        # Prefetch some elements to ensure training smoothness
        dataset = dataset.prefetch(self.prefetch)

        # Filter out any samples that have been flagged as invalid
        dataset = dataset.filter(lambda args: tf.reduce_all(args["valid"]))

        # Perform a ragged batch
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.batch_size))

        # Perform actions needed to convert the ragged batches into training examples
        dataset = dataset.map(self._reduce, num_parallel_calls=self.prefetch)

        # Prefetch some elements to ensure training smoothness
        dataset = dataset.prefetch(self.prefetch)

        return dataset
