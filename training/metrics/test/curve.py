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

import os

import numpy as np
import tensorflow as tf
from .bucket import curve_bucket

if True:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt


class Curve(tf.keras.metrics.Metric):
    def __init__(self, name, x_axis, y_axis, sort_axis, chart, bucket_axis=curve_bucket, n_points=1000):

        super(Curve, self).__init__(name=name)

        # Things needed to reduce the chart
        self.n_points = n_points
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._sort_axis = sort_axis
        self._bucket_axis = bucket_axis

        # Things needed to draw the chart
        self.chart = chart

        # Store values and counts
        self.X = None
        self.c = None

    def reduce(self, X, c):

        # Sort by our sorting axis
        idx = tf.argsort(self._sort_axis(X, c))
        X = tf.gather(X, idx)
        c = tf.gather(c, idx)

        # Calculate the x and y value for all points
        x_values = self._x_axis(X, c)
        y_values = self._y_axis(X, c)

        # Calculate the distance along the curve for each point so we can sample evenly
        curve = self._bucket_axis(x_values, y_values)

        # # We are going to reduce this curve to n_points. If there are large gaps that space will be wasted.
        # # Therefore we calculate the largest gap we can have and still fill all points, then we cut the distances to it.
        # # We work out what the total length of the line would be if we limited it to each value in increasing order.
        # # Then we find the largest value for which this would work and give us no gaps in the resulting curve
        # sorted_curve = tf.sort(curve, direction="ASCENDING")
        # expected_length = tf.cumsum(sorted_curve, exclusive=True) + (
        #     sorted_curve * tf.range(start=tf.size(sorted_curve), limit=0, delta=-1, dtype=sorted_curve.dtype)
        # )
        # # We cutoff at the smallest valid value, or the largest possible value (if no value would be valid)
        # cutoff = tf.reduce_max(tf.where((expected_length / sorted_curve) > self.n_points, sorted_curve, 0.0))
        # cutoff = tf.where(cutoff > 0.0, cutoff, sorted_curve[-1])
        # curve = tf.minimum(curve, cutoff)

        # The first distance along the curve is 0 distance
        curve = tf.pad(tf.math.cumsum(curve), [[1, 0]])

        # Reduce our curve distance to n_points values along the curve
        idx = tf.cast(tf.expand_dims(curve * (self.n_points / curve[-1]), axis=-1), dtype=tf.int32)
        idx = tf.clip_by_value(idx, 0, self.n_points - 1)

        # Sum up the number of samples for each point (weighting)
        weights = tf.expand_dims(tf.math.reduce_sum(c, axis=-1), axis=-1)

        # Scatter the points and weights into the appropriate buckets and then normalise the values
        X = tf.scatter_nd(idx, tf.multiply(X, tf.cast(weights, X.dtype)), (self.n_points, *X.shape[1:]))
        c = tf.scatter_nd(idx, c, (self.n_points, *c.shape[1:]))
        X = tf.math.divide(X, tf.cast(tf.reduce_sum(c, axis=-1, keepdims=True), X.dtype))

        # Remove any points that have a 0 count
        valid_idx = tf.squeeze(tf.where(tf.reduce_sum(c, axis=1)), axis=-1)
        X = tf.gather(X, valid_idx)
        c = tf.gather(c, valid_idx)

        return X, c

    def update(self, X, c):

        # Reduce down to a smaller number of values
        X, c = self.reduce(X, c)

        # We need to do this as a python function so we can get into eager mode
        # That lets us store the results of each, rather than trying to store them in a single variable
        def add_value(X, c):
            self.X = X if self.X is None else tf.concat([self.X, X], axis=0)
            self.c = c if self.c is None else tf.concat([self.c, c], axis=0)
            return tf.shape(self.X, out_type=tf.int64)[0]

        tf.py_function(add_value, (X, c), tf.int64)

    def result(self):
        return 0

    def reset_states(self):
        self.X.assign(tf.zeros_like(self.X))
        self.c.assign(tf.zeros_like(self.c))

    def save(self, output_path):
        # Reduce what we have down to simplified curves
        X, c = self.reduce(self.X, self.c)

        # Get our coordinates
        x_axis = self._x_axis(X, c)
        y_axis = self._y_axis(X, c)
        sort_axis = self._sort_axis(X, c)

        # Make the directories
        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # Plot the figure
        fig, ax = plt.subplots()
        ax.set_title(self.chart["title"])
        ax.set_xlabel(self.chart["x_label"])
        ax.set_ylabel(self.chart["y_label"])
        ax.plot(x_axis, y_axis)
        fig.savefig("{}.png".format(base_path))
        plt.close(fig)

        # Save the raw csv
        np.savetxt(
            "{}.csv".format(base_path),
            tf.stack([x_axis, y_axis, sort_axis], axis=-1).numpy(),
            comments="",
            header="{},{},{}".format(self.chart["x_label"], self.chart["y_label"], self.chart["sort_label"]),
            delimiter=",",
        )

    def curve(self):
        tf.stack([self._sort_axis(self.X, self.c), self._x_axis(self.X, self.c), self._y_axis(self.X, self.c)])
