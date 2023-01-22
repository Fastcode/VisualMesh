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

import cv2
import matplotlib.cm as cm
import numpy as np

import tensorflow as tf


class SeekerHourglass(tf.keras.metrics.Metric):
    def __init__(self, name="seeker_hourglass", n=500, **kwargs):
        super(SeekerHourglass, self).__init__(name=name, **kwargs)

        # We make sure n is an odd number so we always have a point at 0,0
        self.n = n * 2 + 1

        # This gets the previous grid before resetting this is a hack to get around training/validation
        self.last_grid = self.add_weight(name="grid", shape=(self.n, self.n), dtype=tf.int64, initializer="zeros")
        self.grid = self.add_weight(name="grid", shape=(self.n, self.n), dtype=tf.int64, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Y true contains a list of all the possible answers, we are ranked on how close we are to the closest one
        # This code works out the closest point to our prediction for each case so we can use that for training
        y_true = tf.gather_nd(
            y_true,
            tf.stack(
                [
                    tf.range(tf.shape(y_true)[0], dtype=tf.int32),
                    tf.math.argmin(
                        tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(y_pred, axis=-2), y_true), axis=-1),
                        axis=-1,
                        output_type=tf.int32,
                    ),
                ],
                axis=1,
            ),
        )

        # Clip the truth values from -1 to 1
        y_true = tf.clip_by_value(tf.reshape(y_true, shape=[-1]), -1.0, 1.0)
        y_pred = tf.reshape(y_pred, shape=[-1])

        # Calculate the coordinates into our histogram
        coords = tf.stack(
            [
                tf.cast((self.n - 1) * ((y_pred + 1.0) / 2.0), dtype=tf.int32),
                tf.cast((self.n - 1) * ((y_true + 1.0) / 2.0), dtype=tf.int32),
            ],
            axis=-1,
        )

        # Add it to the grid
        self.grid.assign_add(
            tf.scatter_nd(coords, tf.ones_like(y_true, dtype=self.grid.dtype), shape=tf.shape(self.grid))
        )

    def result(self):
        return self.grid

    def reset_states(self):
        self.grid.assign(tf.zeros_like(self.grid))

    def images(self, grid):
        line_width = int(max(1, self.n / 500))

        # Create the image
        grid = tf.cast(grid, tf.float64)
        grid = grid / tf.reduce_max(grid, axis=0, keepdims=True)
        grid = cm.viridis(grid)[:, :, :3]
        img = tf.image.convert_image_dtype(grid, tf.uint8)

        # Draw the diagonal truth line
        img = cv2.line(img.numpy(), (0, 0), (img.shape[0], img.shape[1]), (0, 0, 0), line_width)

        # Draw the boundary boxes for the loss function
        tl_50 = np.array(img.shape[:2]) // 4
        br_50 = np.array(img.shape[:2]) * 3 // 4
        img = cv2.rectangle(img, (tl_50[0], tl_50[1]), (br_50[0], br_50[1]), (255, 255, 255), line_width)

        tl_75 = np.array(img.shape[:2]) // 8
        br_75 = np.array(img.shape[:2]) * 7 // 8
        img = cv2.rectangle(img, (tl_75[0], tl_75[1]), (br_75[0], br_75[1]), (255, 0, 0), line_width)

        return tf.expand_dims(img, axis=0)

    def save(self, output_path):
        output_file = os.path.join(output_path, "{}.png".format(self.name))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(tf.image.encode_png(self.images(self.grid)[0, ...]).numpy())
