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


class VisualMeshDataset:
    def __init__(self, paths, view, example, orientation, projection, label, keys):
        self.paths = paths
        self.view = view
        self.example = example
        self.orientation = orientation
        self.projection = projection
        self.label = label
        self.keys = keys

        # TODO these need to be provided as arguments fix these
        self.batch_size = 10
        self.prefetch = 1  # tf.data.experimental.AUTOTUNE

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
                    raise RuntimeError("Incompatible features for {} ({} vs {})".format(key, features[key], t))

        # Now we build the in order to go from the keys we want to the ones that are in the dataset
        self.dataset_features = {}
        for k, t in self.features.items():
            key = k if k not in self.keys else self.keys[k]
            if key not in self.dataset_features or self.dataset_features[key] == t:
                self.dataset_features[key] = t
            else:
                raise RuntimeError("Incompatible features for {} ({} vs {})".format(key, features[key], t))

    def _map_batch(self, batch):

        results = []
        for i in range(self.batch_size):
            # Load the data and convert it to our internal format
            features = tf.io.parse_single_example(serialized=batch[i], features=self.dataset_features)
            features = {k: features[k if k not in self.keys else self.keys[k]] for k in self.features}

            views = {}
            for p in self.view.prefixes():

                result = {}

                # Use orientation to work out where the mesh should be projected
                result.update(
                    self.orientation(**{**{k: features[p + k] for k in self.orientation.features()}, **result})
                )

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
            results.append(self.view.merge(views))

        return results

    def _reduce_batch(self, batch):
        batch = self._map_batch(batch)

        # We need the n offsets to reposition all the graph values
        n = tf.stack([b["n"] for b in batch], axis=0)
        cn = tf.math.cumsum(tf.math.reduce_sum(n, axis=1), exclusive=True)
        n_elems = tf.math.reduce_sum(n)

        # For X we must add the "offscreen" point as a -1,-1,-1 point one past the end of the list
        X = tf.concat([*[b["X"] for b in batch], tf.constant([[-1.0, -1.0, -1.0]], dtype=tf.float32)], axis=0)

        Y = tf.concat([b["Y"] for b in batch], axis=0)
        W = tf.concat([b["W"] for b in batch], axis=0)
        C = tf.concat([b["C"] for b in batch], axis=0)
        V = tf.concat([b["V"] for b in batch], axis=0)
        img = tf.stack([b["jpg"] for b in batch], axis=0)

        # For the graph we need to move along each element by where it is in the batch
        # We also must replace -1s with a new element off the end of the graph
        G = tf.concat(
            [
                *[tf.where(batch[i]["G"] == -1, n_elems, batch[i]["G"] + cn[i]) for i in range(self.batch_size)],
                tf.fill([1, self.projection.n_neighbours], n_elems),
            ],
            axis=0,
        )

        return {"X": X, "Y": Y, "W": W, "G": G, "n": n, "C": C, "V": V, "img": img}

    def build(self):

        # Load, batch and map the dataset
        dataset = tf.data.TFRecordDataset(self.paths)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._reduce_batch, num_parallel_calls=self.prefetch)

        # Prefetch some elements to ensure training smoothness
        dataset = dataset.prefetch(self.prefetch)

        return dataset
