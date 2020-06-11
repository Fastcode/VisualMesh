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


class Image:
    def __init__(self, **config):
        # TODO variant config in here
        pass

    def _interpolate_gather(self, img, C):

        # Bilinearly interpolate our image based on our floating point pixel coordinate
        y_0 = tf.floor(C[:, 0])
        x_0 = tf.floor(C[:, 1])
        y_1 = y_0 + 1
        x_1 = x_0 + 1

        # Calculate the weights of how much the x and y account for for each of the 4 corners
        y_w = C[:, 0] - y_0
        x_w = C[:, 1] - x_0

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

    def features(self):
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
        }

    def input(self, image, **features):
        # TODO apply variants to the image
        return {
            "jpg": image,
            "image": tf.image.convert_image_dtype(tf.image.decode_image(image, channels=3), tf.float32),
        }

    def __call__(self, image, C, **features):
        # Get the pixels referenced by the image
        return {
            "X": self._interpolate_gather(image, C),
        }
